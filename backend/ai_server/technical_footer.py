from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from ai_server.models import AgentResult, ModelUsageRecord
from ai_server.settings import get_settings


@dataclass(frozen=True)
class ProviderBalanceSnapshot:
    provider: str
    status: str
    lines: list[str]
    available: bool | None = None
    error: str = ""


class DeepSeekBalanceClient:
    def __init__(self) -> None:
        self._cached_until: datetime | None = None
        self._cached_snapshot: ProviderBalanceSnapshot | None = None

    async def snapshot(self) -> ProviderBalanceSnapshot:
        now = datetime.now(timezone.utc)
        if self._cached_snapshot and self._cached_until and now < self._cached_until:
            return self._cached_snapshot

        settings = get_settings()
        snapshot = await self._fetch(settings)
        self._cached_snapshot = snapshot
        self._cached_until = now + timedelta(seconds=settings.tech_footer_balance_cache_seconds)
        return snapshot

    async def _fetch(self, settings: Any) -> ProviderBalanceSnapshot:
        if not settings.deepseek_api_key:
            return ProviderBalanceSnapshot(
                provider="deepseek",
                status="not_configured",
                lines=["DeepSeek: баланс недоступен, API-ключ не настроен."],
                available=None,
            )

        url = settings.deepseek_balance_base_url.rstrip("/") + "/user/balance"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.deepseek_balance_timeout_seconds),
                trust_env=False,
            ) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
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


class TechnicalFooterService:
    def __init__(self, *, deepseek_balance: DeepSeekBalanceClient | None = None) -> None:
        self.deepseek_balance = deepseek_balance or DeepSeekBalanceClient()

    async def build_for_agent_result(
        self,
        result: AgentResult,
        *,
        user_id: int | None,
        channel: str,
    ) -> str:
        if not _is_footer_allowed(user_id=user_id, channel=channel):
            return ""

        lines = [_format_model_usage(result.model_usage)]
        providers = {usage.provider.casefold() for usage in result.model_usage if usage.provider}
        settings = get_settings()
        if settings.tech_footer_balance_enabled and "deepseek" in providers:
            balance = await self.deepseek_balance.snapshot()
            lines.extend(balance.lines)
        return "\n".join(line for line in lines if line)

    async def build_for_pending_action(
        self,
        *,
        user_id: int | None,
        channel: str,
        status: str,
    ) -> str:
        if not _is_footer_allowed(user_id=user_id, channel=channel):
            return ""
        return f"LLM: не использовалась; Bitrix action: {status}."


def append_footer(message: str, footer: str) -> str:
    if not message or not footer:
        return message
    return f"{message}\n\n---\nТех: {footer}"


def _is_footer_allowed(*, user_id: int | None, channel: str) -> bool:
    settings = get_settings()
    if not settings.tech_footer_enabled:
        return False
    if channel != "bitrix24_chat":
        return False
    return bool(user_id is not None and user_id in settings.resolved_tech_footer_allowed_user_ids)


def _format_model_usage(usages: list[ModelUsageRecord]) -> str:
    if not usages:
        return "LLM: не использовалась; выполнено системное действие/API."

    parts: list[str] = []
    for usage in usages:
        label = " ".join(part for part in (usage.agent_id, usage.provider, usage.model) if part)
        if usage.status and usage.status != "used":
            label += f" ({usage.status})"
        if usage.input_tokens is not None or usage.output_tokens is not None:
            label += f" tokens {usage.input_tokens or 0}/{usage.output_tokens or 0}"
        if usage.cost_usd is not None:
            label += f" cost ${usage.cost_usd:.4f}"
        parts.append(label)
    return "LLM: " + "; ".join(parts) + "."


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
