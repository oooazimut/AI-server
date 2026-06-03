from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ai_server.settings import get_settings


class BitrixApiError(RuntimeError):
    def __init__(self, method: str, error: str, description: str = "") -> None:
        self.method = method
        self.error = error
        self.description = description
        super().__init__(f"Bitrix REST error in {method}: {error} {description}".strip())


class BitrixConfigError(RuntimeError):
    pass


class BitrixClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        access_token: str | None = None,
        client_endpoint: str | None = None,
    ) -> None:
        settings = get_settings()
        self.access_token = access_token or ""
        resolved_base_url = client_endpoint if self.access_token else base_url
        self.base_url = (resolved_base_url or settings.bitrix_rest_webhook_url).rstrip("/") + "/"
        self.api_base_url = _to_rest_api_base_url(self.base_url)
        self.projects_base_url = (
            self.base_url
            if self.access_token
            else (settings.bitrix_projects_webhook_url or self.base_url)
        ).rstrip("/") + "/"
        self.timeout = httpx.Timeout(30.0)

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip("/"))

    async def call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        resolved_base_url = base_url or self.base_url
        if not resolved_base_url.strip("/"):
            raise BitrixConfigError("Bitrix REST endpoint is not configured")

        url = f"{resolved_base_url}{method}.json"
        request_payload = dict(payload or {})
        if self.access_token:
            request_payload.setdefault("auth", self.access_token)
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            response = await client.post(url, json=request_payload)
            response.raise_for_status()

        data = response.json()
        if "error" in data:
            raise BitrixApiError(
                method=method,
                error=str(data.get("error", "")),
                description=str(data.get("error_description", "")),
            )
        return data

    async def result(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> Any:
        data = await self.call(method, payload, base_url=base_url)
        return data.get("result")

    async def call_v3(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        resolved_base_url = base_url or self.api_base_url
        if not resolved_base_url.strip("/"):
            raise BitrixConfigError("Bitrix REST endpoint is not configured")

        url = f"{resolved_base_url}{method}"
        request_payload = dict(payload or {})
        if self.access_token:
            request_payload.setdefault("auth", self.access_token)
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            response = await client.post(url, json=request_payload)
            response.raise_for_status()

        data = response.json()
        if "error" in data:
            raise BitrixApiError(
                method=method,
                error=str(data.get("error", "")),
                description=str(data.get("error_description", "")),
            )
        return data

    async def result_v3(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> Any:
        data = await self.call_v3(method, payload, base_url=base_url)
        return data.get("result")

    async def send_bot_message(
        self,
        dialog_id: str,
        message: str,
        *,
        bot_id: int | None = None,
        keyboard: object | None = None,
    ) -> Any:
        settings = get_settings()
        resolved_bot_id = bot_id or settings.bitrix_bot_id
        if not resolved_bot_id:
            raise BitrixConfigError("Bot id is required: pass bot_id or set BITRIX_BOT_ID")

        payload: dict[str, Any] = {
            "botId": resolved_bot_id,
            "dialogId": dialog_id,
            "fields": {"message": message},
        }
        if not self.access_token:
            payload["botToken"] = settings.bitrix_bot_token
        if keyboard:
            payload["fields"]["keyboard"] = keyboard
        return await self.result("imbot.v2.Chat.Message.send", payload)

    async def collect_paged(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        list_key: str | None = None,
        limit: int | None = None,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        start: int | None = 0
        while start is not None:
            page_payload = dict(payload or {})
            page_payload["start"] = start
            data = await self.call(method, page_payload, base_url=base_url)
            page_items = _extract_paged_items(data.get("result"), list_key=list_key)
            items.extend(page_items)
            if limit and len(items) >= limit:
                return items[:limit]
            raw_next = data.get("next")
            start = int(raw_next) if raw_next is not None else None
        return items

    async def download_file_from_url(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
    ) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        bytes_read = 0
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                follow_redirects=True,
                trust_env=False,
            ) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code >= 400:
                        raise BitrixApiError("download_file", f"HTTP_{response.status_code}")
                    with destination.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            bytes_read += len(chunk)
                            if bytes_read > max_bytes:
                                raise BitrixApiError(
                                    "download_file",
                                    "FILE_TOO_LARGE",
                                    f"File exceeds {max_bytes} bytes",
                                )
                            handle.write(chunk)
        except Exception:
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise
        return bytes_read


def _to_rest_api_base_url(base_url: str) -> str:
    if "/rest/" not in base_url:
        return base_url
    prefix, _, suffix = base_url.partition("/rest/")
    parts = suffix.strip("/").split("/")
    if len(parts) >= 2 and parts[0].isdigit():
        return f"{prefix}/rest/"
    return base_url


def _extract_paged_items(result: Any, *, list_key: str | None) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        if list_key and isinstance(result.get(list_key), list):
            return [item for item in result[list_key] if isinstance(item, dict)]
        for value in result.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []

