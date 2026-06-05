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

        data = _response_json(response)
        if "error" in data:
            raise BitrixApiError(
                method=method,
                error=str(data.get("error", "")),
                description=str(data.get("error_description", "")),
            )
        if response.is_error:
            raise BitrixApiError(
                method=method,
                error=f"HTTP_{response.status_code}",
                description=_response_text(response),
            ) from None
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

        data = _response_json(response)
        if "error" in data:
            raise BitrixApiError(
                method=method,
                error=str(data.get("error", "")),
                description=str(data.get("error_description", "")),
            )
        if response.is_error:
            raise BitrixApiError(
                method=method,
                error=f"HTTP_{response.status_code}",
                description=_response_text(response),
            ) from None
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
        if settings.bitrix_bot_uses_oauth and not self.access_token:
            client = await _oauth_bot_client()
            return await client.send_bot_message(
                dialog_id,
                message,
                bot_id=bot_id,
                keyboard=keyboard,
            )

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

    async def create_bot_chat(
        self,
        *,
        title: str,
        user_ids: list[int],
        description: str = "",
        color: str = "mint",
        message: str = "",
        bot_id: int | None = None,
        owner_id: int | None = None,
    ) -> Any:
        settings = get_settings()
        if settings.bitrix_bot_uses_oauth and not self.access_token:
            client = await _oauth_bot_client()
            return await client.create_bot_chat(
                title=title,
                user_ids=user_ids,
                description=description,
                color=color,
                message=message,
                bot_id=bot_id,
                owner_id=owner_id,
            )

        resolved_bot_id = bot_id or settings.bitrix_bot_id
        normalized_user_ids = _unique_positive_ints(user_ids)
        if not resolved_bot_id:
            raise BitrixConfigError("Bot id is required: pass bot_id or set BITRIX_BOT_ID")
        if not title.strip():
            raise BitrixConfigError("Chat title is required")
        if not normalized_user_ids:
            raise BitrixConfigError("At least one user id is required")

        fields: dict[str, Any] = {
            "title": title.strip(),
            "color": color,
            "userIds": normalized_user_ids,
        }
        if description:
            fields["description"] = description
        if message:
            fields["message"] = message
        if owner_id is not None:
            fields["ownerId"] = owner_id

        payload: dict[str, Any] = {
            "botId": resolved_bot_id,
            "fields": fields,
        }
        if not self.access_token:
            if not settings.bitrix_bot_token:
                raise BitrixConfigError("Bot token is required in webhook auth mode: set BITRIX_BOT_TOKEN")
            payload["botToken"] = settings.bitrix_bot_token
        return await self.result("imbot.v2.Chat.add", payload)

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

    async def list_all_tasks(
        self,
        *,
        filter_: dict[str, Any] | None = None,
        select: list[str] | None = None,
        order: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {}
        if filter_:
            payload["filter"] = filter_
        if select:
            payload["select"] = select
        if order:
            payload["order"] = order
        return await self.collect_paged(
            "tasks.task.list",
            payload,
            list_key="tasks",
            limit=limit,
        )

    async def get_task(self, task_id: int, *, select: list[str] | None = None) -> Any:
        payload: dict[str, Any] = {"taskId": task_id}
        if select:
            payload["select"] = select
        return await self.result("tasks.task.get", payload)

    async def list_task_results(self, task_id: int) -> Any:
        return await self.result("tasks.task.result.list", {"taskId": task_id})

    async def add_task_result(self, task_id: int, text: str) -> Any:
        return await self.result(
            "tasks.task.result.add",
            {
                "taskId": task_id,
                "fields": {"text": text},
            },
        )

    async def disapprove_task(self, task_id: int) -> Any:
        return await self.result("tasks.task.disapprove", {"taskId": task_id})

    async def approve_task(self, task_id: int) -> Any:
        return await self.result("tasks.task.approve", {"taskId": task_id})

    async def complete_task(self, task_id: int) -> Any:
        return await self.result("tasks.task.complete", {"taskId": task_id})

    async def renew_task(self, task_id: int) -> Any:
        return await self.result("tasks.task.renew", {"taskId": task_id})

    async def require_task_result(self, task_id: int) -> Any:
        return await self.result_v3(
            "tasks.task.update",
            {
                "id": task_id,
                "fields": {"requireResult": True},
            },
        )

    async def add_task_comment(
        self,
        *,
        task_id: int,
        message: str,
        author_id: int | None = None,
    ) -> Any:
        fields: dict[str, Any] = {"POST_MESSAGE": message}
        if author_id is not None:
            fields["AUTHOR_ID"] = author_id
        return await self.result(
            "task.commentitem.add",
            {
                "TASKID": task_id,
                "FIELDS": fields,
            },
        )

    async def notify_user(
        self,
        *,
        user_id: int,
        message: str,
        tag: str = "ai_server",
        sub_tag: str = "",
    ) -> Any:
        return await self.result(
            "im.notify.system.add",
            {
                "USER_ID": user_id,
                "MESSAGE": message,
                "TAG": f"{tag}:{sub_tag}" if sub_tag else tag,
            },
        )

    async def search_users(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "FILTER": {"ACTIVE": True},
            "SORT": "LAST_NAME",
            "ORDER": "ASC",
            "LIMIT": limit,
        }
        if query:
            payload["FILTER"]["FIND"] = query
        result = await self.result("user.search", payload)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)][:limit]
        if isinstance(result, dict):
            for key in ("users", "items", "result"):
                items = result.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)][:limit]
        return []

    async def get_user(self, user_id: int) -> dict[str, Any] | None:
        for payload in ({"ID": user_id}, {"FILTER": {"ID": user_id}}):
            result = await self.result("user.get", payload)
            user = _extract_user(result, user_id=user_id)
            if user is not None:
                return user
        return None

    async def get_attached_object(self, attached_object_id: int) -> Any:
        return await self.result("disk.attachedObject.get", {"id": attached_object_id})

    async def get_disk_file(self, file_id: int) -> Any:
        return await self.result("disk.file.get", {"id": file_id})

    async def get_disk_file_download_url(self, file_id: int) -> str:
        result = await self.get_disk_file(file_id)
        if isinstance(result, dict):
            download_url = result.get("DOWNLOAD_URL") or result.get("downloadUrl")
            if download_url:
                return str(download_url)
        raise BitrixApiError("disk.file.get", "EMPTY_DOWNLOAD_URL")

    async def list_workgroups(
        self,
        *,
        filter_: dict[str, Any] | None = None,
        order: dict[str, Any] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {}
        if filter_:
            payload["FILTER"] = filter_
        if order:
            payload["ORDER"] = order
        return await self.result("sonet_group.get", payload, base_url=self.projects_base_url)

    async def search_projects(self, query: str = "", *, limit: int = 10) -> list[dict[str, Any]]:
        filter_: dict[str, Any] = {
            "ACTIVE": "Y",
            "PROJECT": "Y",
        }
        if query:
            filter_["%NAME"] = query

        result = await self.list_workgroups(
            filter_=filter_,
            order={"NAME": "ASC"},
        )
        projects = _extract_workgroups(result)

        if query and not projects:
            result = await self.list_workgroups(
                filter_={"ACTIVE": "Y", "PROJECT": "Y"},
                order={"NAME": "ASC"},
            )
            projects = [
                project
                for project in _extract_workgroups(result)
                if query.lower() in str(project.get("NAME") or project.get("name") or "").lower()
            ]

        return projects[:limit]

    async def list_disk_storages(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return await self.collect_paged(
            "disk.storage.getlist",
            {},
            limit=limit,
        )

    async def list_disk_folder_children_all(
        self,
        *,
        folder_id: int,
        filter_: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"id": folder_id}
        if filter_:
            payload["filter"] = filter_
        return await self.collect_paged(
            "disk.folder.getchildren",
            payload,
            limit=limit,
        )

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


def _unique_positive_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        if normalized <= 0 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    if len(text) > 500:
        return text[:500] + "..."
    return text


async def _oauth_bot_client() -> BitrixClient:
    settings = get_settings()
    if not settings.bitrix_bot_oauth_user_id:
        raise BitrixConfigError("BITRIX_BOT_OAUTH_USER_ID is required for OAuth bot mode")
    from ai_server.integrations.bitrix.oauth import BitrixOAuthService

    return await BitrixOAuthService().client_for_user(settings.bitrix_bot_oauth_user_id)


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


def _extract_workgroups(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("groups", "workgroups", "items"):
            items = result.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _extract_user(result: Any, *, user_id: int) -> dict[str, Any] | None:
    if isinstance(result, dict):
        if _optional_int(result.get("ID") or result.get("id")) == user_id:
            return result
        for key in ("users", "items", "result"):
            items = result.get(key)
            if isinstance(items, list):
                user = _extract_user(items, user_id=user_id)
                if user is not None:
                    return user
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and _optional_int(item.get("ID") or item.get("id")) == user_id:
                return item
    return None


def _optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
