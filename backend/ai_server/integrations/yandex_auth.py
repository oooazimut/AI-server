from __future__ import annotations

from ai_server.settings import Settings


class YandexAuthError(RuntimeError):
    pass


def yandex_auth_header(settings: Settings) -> dict[str, str]:
    if settings.yandex_api_key:
        return {"Authorization": f"Api-Key {settings.yandex_api_key}"}
    if settings.yandex_iam_token:
        return {"Authorization": f"Bearer {settings.yandex_iam_token}"}
    raise YandexAuthError("YANDEX_API_KEY or YANDEX_IAM_TOKEN is required")
