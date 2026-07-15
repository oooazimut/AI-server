from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

import httpx

from ai_server.models import AgentResult, ModelUsageRecord
from ai_server.settings import Settings, get_settings


@dataclass(frozen=True)
class ProviderBalanceSnapshot:
    provider: str
    status: str
    lines: list[str]
    available: bool | None = None
    error: str = ""


class BalanceClient(Protocol):
    async def snapshot(self) -> ProviderBalanceSnapshot: ...


class DeepSeekBalanceClient:
    def __init__(self, settings: Settings) -> None:
        self._lock = asyncio.Lock()
        self._api_key = settings.deepseek_api_key
        self._base_url = settings.deepseek_balance_base_url
        self._timeout_seconds = settings.deepseek_balance_timeout_seconds
        self._cache_seconds = settings.tech_footer_balance_cache_seconds
        self._cached_until: datetime | None = None
        self._cached_snapshot: ProviderBalanceSnapshot | None = None

    async def snapshot(self) -> ProviderBalanceSnapshot:
        now = datetime.now(UTC)
        if self._cached_snapshot and self._cached_until and now < self._cached_until:
            return self._cached_snapshot

        async with self._lock:
            now = datetime.now(UTC)
            if self._cached_snapshot and self._cached_until and now < self._cached_until:
                return self._cached_snapshot

            snapshot = await self._fetch()
            self._cached_snapshot = snapshot
            self._cached_until = now + timedelta(seconds=self._cache_seconds)
            return snapshot

    async def _fetch(self) -> ProviderBalanceSnapshot:
        if not self._api_key:
            return ProviderBalanceSnapshot(
                provider="deepseek",
                status="not_configured",
                lines=["DeepSeek: баланс недоступен, API-ключ не настроен."],
                available=None,
            )

        url = self._base_url.rstrip("/") + "/user/balance"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_seconds),
                trust_env=False,
            ) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return ProviderBalanceSnapshot(
                provider="deepseek",
                status="error",
                lines=[f"DeepSeek: баланс недоступен ({type(exc).__name__})."],
                available=None,
                error=str(exc),
            )

        lines = _format_deepseek_balance(payload)
        return ProviderBalanceSnapshot(
            provider="deepseek",
            status="ok",
            lines=lines,
            available=_optional_bool(payload.get("is_available")) if isinstance(payload, dict) else None,
        )


class YandexBalanceClient:
    def __init__(self, settings: Settings) -> None:
        self._lock = asyncio.Lock()
        self._account_id = settings.yandex_billing_account_id
        self._token = settings.yandex_billing_iam_token or settings.yandex_iam_token
        self._base_url = settings.yandex_billing_base_url
        self._cache_seconds = settings.tech_footer_balance_cache_seconds
        self._cached_until: datetime | None = None
        self._cached_snapshot: ProviderBalanceSnapshot | None = None

    async def snapshot(self) -> ProviderBalanceSnapshot:
        now = datetime.now(UTC)
        if self._cached_snapshot and self._cached_until and now < self._cached_until:
            return self._cached_snapshot

        async with self._lock:
            now = datetime.now(UTC)
            if self._cached_snapshot and self._cached_until and now < self._cached_until:
                return self._cached_snapshot

            snapshot = await self._fetch()
            self._cached_snapshot = snapshot
            self._cached_until = now + timedelta(seconds=self._cache_seconds)
            return snapshot

    async def _fetch(self) -> ProviderBalanceSnapshot:
        if not self._account_id:
            return ProviderBalanceSnapshot(
                provider="yandex",
                status="not_configured",
                lines=["Yandex Cloud: баланс недоступен, YANDEX_BILLING_ACCOUNT_ID не задан."],
                available=None,
            )

        if not self._token:
            return ProviderBalanceSnapshot(
                provider="yandex",
                status="not_configured",
                lines=["Yandex Cloud: баланс недоступен, YANDEX_BILLING_IAM_TOKEN не задан."],
                available=None,
            )

        url = self._base_url.rstrip("/") + f"/billing/v1/billingAccounts/{self._account_id}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {self._token}"})
        except Exception as exc:
            return ProviderBalanceSnapshot(
                provider="yandex",
                status="error",
                lines=[f"Yandex Cloud: баланс недоступен ({type(exc).__name__})."],
                available=None,
                error=str(exc),
            )

        if response.status_code >= 400:
            return ProviderBalanceSnapshot(
                provider="yandex",
                status="error",
                lines=[f"Yandex Cloud: нет доступа (HTTP {response.status_code})."],
                available=None,
            )

        try:
            body = response.json()
        except ValueError:
            return ProviderBalanceSnapshot(
                provider="yandex",
                status="error",
                lines=["Yandex Cloud: не разобрал ответ API."],
                available=None,
            )

        balance = _decimal_from_any(body.get("balance"))
        currency = str(body.get("currency") or "RUB")
        if balance is None:
            return ProviderBalanceSnapshot(
                provider="yandex",
                status="error",
                lines=["Yandex Cloud: баланс не найден в ответе."],
                available=None,
            )

        active = bool(body.get("active", True))
        suffix = "" if active else "; аккаунт неактивен"
        return ProviderBalanceSnapshot(
            provider="yandex",
            status="ok",
            lines=[f"Yandex Cloud: баланс {_format_money(balance, currency)}{suffix}."],
            available=active,
        )


def build_balance_registry(settings: Settings) -> dict[str, BalanceClient]:
    registry: dict[str, BalanceClient] = {}
    if settings.deepseek_api_key:
        registry["deepseek"] = DeepSeekBalanceClient(settings)
    if settings.yandex_billing_account_id and (settings.yandex_billing_iam_token or settings.yandex_iam_token):
        registry["yandex"] = YandexBalanceClient(settings)
    return registry


class TechnicalFooterService:
    def __init__(
        self,
        *,
        balance_registry: dict[str, BalanceClient] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._balance_registry: dict[str, BalanceClient] = (
            balance_registry if balance_registry is not None else build_balance_registry(self._settings)
        )

    async def build_for_agent_result(
        self,
        result: AgentResult,
        *,
        user_id: int | None,
        channel: str,
    ) -> str:
        if not self._is_footer_allowed(user_id=user_id, channel=channel):
            return ""

        lines = [_format_model_usage(result)]
        if self._settings.tech_footer_balance_enabled:
            providers = _balance_providers(result.model_usage, self._balance_registry)
            balance_lines = await self._collect_balance_lines(providers)
            lines.extend(balance_lines)
        return ", ".join(line for line in lines if line)

    async def build_for_pending_action(
        self,
        *,
        user_id: int | None,
        channel: str,
        status: str,
        model_usage: Any | None = None,
    ) -> str:
        if not self._is_footer_allowed(user_id=user_id, channel=channel):
            return ""
        usages = _coerce_model_usage_list(model_usage)
        if not usages:
            return f"LLM: не использовалась; Bitrix action: {status}."

        lines = [_format_model_usage_records(usages), f"Bitrix action: {status}."]
        if self._settings.tech_footer_balance_enabled:
            providers = _balance_providers(usages, self._balance_registry)
            balance_lines = await self._collect_balance_lines(providers)
            lines.extend(balance_lines)
        return ", ".join(line for line in lines if line)

    def _is_footer_allowed(self, *, user_id: int | None, channel: str) -> bool:
        if not self._settings.tech_footer_enabled:
            return False
        if channel != "bitrix24_chat":
            return False
        return bool(user_id is not None and user_id in self._settings.resolved_tech_footer_allowed_user_ids)

    async def _collect_balance_lines(self, providers: set[str]) -> list[str]:
        clients = [
            (provider, self._balance_registry[provider])
            for provider in sorted(providers)
            if provider in self._balance_registry
        ]
        if not clients:
            return []
        snapshots = await asyncio.gather(*[client.snapshot() for _, client in clients], return_exceptions=True)
        lines: list[str] = []
        for snapshot in snapshots:
            if isinstance(snapshot, ProviderBalanceSnapshot):
                lines.extend(_format_balance_snapshot_compact(snapshot))
        return lines


def append_footer(message: str, footer: str) -> str:
    if not message or not footer:
        return message
    return f"{message}\n\n---\nТех: {footer}"


def _format_model_usage(result: AgentResult) -> str:
    return _format_model_usage_records(result.model_usage, result=result)


def _format_model_usage_records(usages: list[ModelUsageRecord], *, result: AgentResult | None = None) -> str:
    if not usages:
        return "LLM не использовалась"

    visible_usages = [usage for usage in usages if usage.status != "skipped"]
    if not visible_usages:
        visible_usages = usages

    input_tokens = sum(usage.input_tokens or 0 for usage in visible_usages)
    output_tokens = sum(usage.output_tokens or 0 for usage in visible_usages)
    total_cost = sum(usage.cost_usd or 0 for usage in visible_usages)

    parts = [_format_model_route(result=result, usages=visible_usages)]
    if any(usage.input_tokens is not None or usage.output_tokens is not None for usage in visible_usages):
        parts.append(f"{input_tokens}/{output_tokens} ток.")
    elif any(usage.status == "skipped" or usage.provider == "local" for usage in visible_usages):
        parts.append("0/0 ток.")
    if total_cost:
        parts.append(f"${total_cost:.4f}")
    return ", ".join(part for part in parts if part)


def _balance_providers(usages: list[ModelUsageRecord], registry: dict[str, BalanceClient]) -> set[str]:
    providers = {usage.provider.casefold() for usage in usages if usage.provider}
    if "deepseek" in registry:
        providers.add("deepseek")
    return providers


def _format_model_route(*, result: AgentResult | None, usages: list[ModelUsageRecord]) -> str:
    if result is not None:
        source = _agent_label(result.agent_id)
        handoffs = [_agent_label(agent_id) for agent_id in result.handoff_to if agent_id]
        handoffs = [label for label in handoffs if label and label != source]
        if handoffs:
            return f"{source} → {', '.join(handoffs)}"
        if result.agent_id:
            return source

    agent_ids: list[str] = []
    for usage in usages:
        if usage.agent_id and usage.agent_id not in agent_ids:
            agent_ids.append(usage.agent_id)
    if not agent_ids:
        return "LLM"
    labels = [_agent_label(agent_id) for agent_id in agent_ids]
    return " → ".join(label for label in labels if label)


def _agent_label(agent_id: str) -> str:
    labels = {
        "internal_orchestrator": "оркестр",
        "bitrix24": "Bitrix",
        "diagnost": "диагност",
        "kartoteka": "картотека",
        "logistics": "логистика",
        "pto": "ПТО",
    }
    return labels.get(agent_id, agent_id)


def _format_balance_snapshot_compact(snapshot: ProviderBalanceSnapshot) -> list[str]:
    if snapshot.provider.casefold() != "deepseek":
        return [line.rstrip(".") for line in snapshot.lines]

    if snapshot.status == "ok":
        status = "OK" if snapshot.available is not False else "недоступен"
        balance = _extract_balance_text(snapshot.lines)
        suffix = f", {balance}" if balance else ""
        return [f"DeepSeek {status}{suffix}"]
    if snapshot.status == "not_configured":
        return ["DeepSeek не настроен"]
    return ["DeepSeek ошибка"]


def _extract_balance_text(lines: list[str]) -> str:
    for line in lines:
        marker = "баланс "
        if marker not in line:
            continue
        value = line.split(marker, 1)[1].strip().rstrip(".")
        if not value or value.startswith("не "):
            return ""
        return value
    return ""


def _coerce_model_usage_list(value: Any) -> list[ModelUsageRecord]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    usages: list[ModelUsageRecord] = []
    for item in raw_items:
        if isinstance(item, ModelUsageRecord):
            usages.append(item)
            continue
        if isinstance(item, dict):
            try:
                usages.append(ModelUsageRecord.model_validate(item))
            except Exception:
                continue
    return usages


def _format_deepseek_balance(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["DeepSeek: баланс недоступен, неожиданный ответ API."]

    availability = "доступен" if payload.get("is_available") is True else "недоступен"
    balances: list[str] = []
    for item in payload.get("balance_infos") or []:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("currency") or "").upper()
        amount = _decimal_from_any(item.get("total_balance"))
        if currency and amount is not None:
            balances.append(_format_money(amount, currency))

    if balances:
        return [f"DeepSeek: {availability}; баланс " + ", ".join(balances) + "."]
    return [f"DeepSeek: {availability}; баланс не вернулся в ответе API."]


def _decimal_from_any(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value).replace(",", "."))
    except (InvalidOperation, ValueError):
        return None


def _format_money(amount: Decimal, currency: str) -> str:
    rounded = amount.quantize(Decimal("0.01"))
    number = f"{rounded:,.2f}".replace(",", " ")
    if currency == "USD":
        return f"${number}"
    if currency == "RUB":
        return f"{number} руб."
    return f"{number} {currency}"


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


# Legacy alias kept for backward compat within this session
_DeepSeekBalanceClient = DeepSeekBalanceClient
