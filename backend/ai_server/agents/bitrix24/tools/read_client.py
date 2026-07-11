from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthError, BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort

MOBILE_AUTHORIZATION_INSTRUCTION = (
    "Для авторизации с телефона откройте мобильное приложение Bitrix24: "
    "Меню справа внизу -> Маркет -> ИИ Агент-помощник. "
    "Если пункт Маркет не виден, прокрутите меню примерно на один экран вниз. "
    "После открытия должна появиться страница 'OAuth подключён'."
)


def oauth_authorization_data(
    bitrix_oauth: BitrixOAuthService | None,
    *,
    user_id: int | None,
) -> dict[str, Any]:
    if bitrix_oauth is None:
        return {}
    hint = bitrix_oauth.authorization_hint(user_id)
    oauth_url = str(hint.get("oauth_start_url") or "").strip()
    app_url = _app_authorization_url(hint)
    primary_url = oauth_url or app_url
    hint = {
        **hint,
        "mobile_instruction": MOBILE_AUTHORIZATION_INSTRUCTION,
        "primary_authorization_url": primary_url,
        "app_authorization_url": app_url,
        "fallback_oauth_url": oauth_url,
        "message": _authorization_message(oauth_url=oauth_url, app_url=app_url),
    }
    return {"authorization": hint, "oauth_required": True}


def oauth_missing_error(_prefix: str, *, user_id: int, authorization: dict[str, Any] | None = None) -> str:
    hint = authorization or {}
    message = hint.get("message") or _authorization_message()
    return f"Требуется авторизация Bitrix24 для пользователя #{user_id}. {message}"


def _app_authorization_url(hint: dict[str, Any]) -> str:
    return (
        str(hint.get("marketplace_app_url") or "").strip()
        or str(hint.get("app_url") or "").strip()
        or str(hint.get("marketplace_app_path") or "").strip()
    )


def _authorization_message(*, oauth_url: str = "", app_url: str = "") -> str:
    parts = [
        "Требуется авторизация в Bitrix24.",
        MOBILE_AUTHORIZATION_INSTRUCTION,
    ]
    if oauth_url:
        parts.append(f"Ссылка для авторизации с компьютера: {oauth_url}")
    if app_url:
        parts.append(f"Запасная ссылка на приложение: {app_url}")
    return " ".join(parts)


async def resolve_current_user_read_client(
    tool_name: str,
    *,
    fallback_client: BitrixToolClientPort | None,
    bitrix_oauth: BitrixOAuthService | None,
    user_id: int | None,
) -> tuple[BitrixToolClientPort, str, ToolResult | None]:
    if bitrix_oauth is None:
        if fallback_client is None:
            return (
                _MissingBitrixClient(),
                "none",
                ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=tool_name, error="BitrixClient is not injected"),
            )
        return fallback_client, "configured_client", None

    if user_id is None:
        return (
            _MissingBitrixClient(),
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error="Bitrix read denied: current Bitrix user_id is missing.",
            ),
        )

    try:
        return await bitrix_oauth.client_for_user(user_id), "oauth_current_user", None
    except BitrixOAuthTokenMissing as exc:
        data = oauth_authorization_data(bitrix_oauth, user_id=exc.user_id)
        return (
            _MissingBitrixClient(),
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error=oauth_missing_error(
                    "Bitrix read denied",
                    user_id=exc.user_id,
                    authorization=data.get("authorization"),
                ),
                data=data,
            ),
        )
    except (BitrixOAuthError, BitrixConfigError) as exc:
        return (
            _MissingBitrixClient(),
            "none",
            ToolResult(status=ToolStatus.ERROR, tool=tool_name, error=f"Bitrix OAuth read client failed: {exc}"),
        )


class _MissingBitrixClient:
    async def result(self, method: str, params: dict[str, Any]) -> Any:
        raise BitrixConfigError("BitrixClient is not injected")
