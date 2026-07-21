from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ai_server.agents.bitrix24.tools.read_client import resolve_current_user_read_client
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.bitrix.profile import compact_user_profile
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.utils import optional_int

TASK_CLOSE_AUTO_CLOSE_TIME_KEY = "auto_close_time"
TASK_CLOSE_CONTROL_ENABLED_FROM_KEY = "control_enabled_from"
TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME = "20:00"

_CONTROLLED_USER_ACTIONS = {"add_controlled_user", "remove_controlled_user"}
_ADMIN_ONLY_ACTIONS = {"add_operator", "remove_operator", "set_auto_close_time", "set_control_enabled_from"}
_CHANGE_ACTIONS = _CONTROLLED_USER_ACTIONS | _ADMIN_ONLY_ACTIONS
_DRAFT_TYPE = "admin_change"
_DRAFT_TTL_MINUTES = 15
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class TaskCloseControlGetTool:
    name = "task_close_control_get"

    def __init__(self, store: Any | None = None, user_client: Any | None = None) -> None:
        self._store = store
        self._user_client = user_client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read Bitrix task close control settings: operators, controlled users, auto-close time, "
                "control start date, named current members, and active Bitrix users available for assignment. "
                "Admins and task-close operators only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "include_available_users": {
                        "type": "boolean",
                        "description": "Include active Bitrix users that can be assigned. Default: true.",
                    },
                    "available_limit": {
                        "type": "integer",
                        "description": "Maximum Bitrix users to return, default 100, max 200.",
                    },
                    "user_query": {
                        "type": "string",
                        "description": "Optional Bitrix user search text to narrow the candidate list.",
                    },
                },
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="Bitrix store is not configured")
        actor = _actor_context(self._store, user_id=user_id, actor_is_admin=_truthy(args.get("_actor_is_admin")))
        if not actor["is_admin"] and not actor["is_operator"]:
            return _denied(self.name, user_id=user_id)
        data = await _control_snapshot(
            self._store,
            actor=actor,
            user_client=self._user_client,
            args=args,
            include_available_default=True,
        )
        return ToolResult(status=ToolStatus.OK, tool=self.name, data=data)


class TaskCloseControlUpdateTool:
    name = "task_close_control_update"

    def __init__(
        self,
        store: Any | None = None,
        user_client: Any | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._store = store
        self._user_client = user_client
        self._bitrix_oauth = bitrix_oauth

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Prepare, confirm, or discard one task-close control setting change. Every change is admin-only "
                "and first creates a 15-minute review draft; confirm applies exactly that one reviewed change."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "add_operator",
                            "remove_operator",
                            "add_controlled_user",
                            "remove_controlled_user",
                            "set_auto_close_time",
                            "set_control_enabled_from",
                        ],
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["prepare", "confirm", "discard"],
                        "description": "Default prepare. Confirm/discard operate on the active admin-change draft.",
                    },
                    "target_user_id": {"type": "integer"},
                    "target_user_name": {
                        "type": "string",
                        "description": "Exact Bitrix display name when the numeric id is not known.",
                    },
                    "value": {"type": "string"},
                    "auto_close_time": {"type": "string", "description": "HH:MM, for example 20:00."},
                    "control_enabled_from": {"type": "string", "description": "ISO datetime/date for control start."},
                },
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="Bitrix store is not configured")
        operation = str(args.get("operation") or "prepare").strip().casefold()
        if operation == "confirm":
            return await self._confirm(args, user_id=user_id, dialog_key=dialog_key)
        if operation == "discard":
            return await self._discard(user_id=user_id, dialog_key=dialog_key)
        if operation != "prepare":
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_control_update.operation is invalid",
            )
        action = str(args.get("action") or "").strip()
        if action not in _CHANGE_ACTIONS:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_control_update.action is invalid",
            )
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="dialog_key is required for an admin change draft",
            )
        admin_client, admin_error = await self._fresh_admin_client(user_id)
        if admin_error is not None:
            return admin_error
        prepared = await self._prepare_change(args, action=action, user_id=user_id, user_client=admin_client)
        if isinstance(prepared, ToolResult):
            return prepared
        try:
            await self._store.save_task_draft(dialog_key, prepared)
            stored = await self._store.get_task_draft(dialog_key, ttl_minutes=_DRAFT_TTL_MINUTES)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR, tool=self.name, error=f"Could not save admin change draft: {exc}"
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "action": action,
                "operation": "prepare",
                "requires_confirmation": True,
                "draft_ttl_minutes": _DRAFT_TTL_MINUTES,
                "draft": stored or prepared,
            },
        )

    async def _confirm(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None,
        dialog_key: str | None,
    ) -> ToolResult:
        if not dialog_key:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="dialog_key is required")
        admin_client, admin_error = await self._fresh_admin_client(user_id)
        if admin_error is not None:
            return admin_error
        draft = await self._store.get_task_draft(dialog_key, ttl_minutes=_DRAFT_TTL_MINUTES)
        if not isinstance(draft, dict) or draft.get("_draft_type") != _DRAFT_TYPE:
            return ToolResult(
                status=ToolStatus.NOT_FOUND, tool=self.name, error="Active admin change draft was not found"
            )
        draft_user_id = optional_int(draft.get("_draft_user_id") or draft.get("actor_user_id"))
        if draft_user_id is not None and draft_user_id != user_id:
            return _denied(self.name, user_id=user_id, reason="This admin change draft belongs to another user.")
        action = str(draft.get("action") or "")
        if action not in _CHANGE_ACTIONS:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="Admin change draft is invalid"
            )
        draft_id = str(draft.get("_draft_id") or "")
        draft_version = optional_int(draft.get("_draft_version"))
        if not draft_id or draft_version is None:
            return ToolResult(status=ToolStatus.ERROR, tool=self.name, error="Admin change draft has no CAS identity")
        target_error = await self._revalidate_target(draft, user_client=admin_client)
        if target_error is not None:
            return target_error

        atomic_confirm = getattr(self._store, "confirm_admin_change_draft", None)
        if callable(atomic_confirm):
            try:
                outcome = await atomic_confirm(
                    dialog_key=dialog_key,
                    draft_id=draft_id,
                    draft_version=draft_version,
                    actor_user_id=user_id,
                )
            except Exception:
                return ToolResult(
                    status=ToolStatus.ERROR, tool=self.name, error="Could not apply the reviewed admin change"
                )
            outcome_status = str((outcome or {}).get("status") or "") if isinstance(outcome, dict) else ""
            if outcome_status == "conflict":
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="Task-close control setting changed after draft creation; prepare a new draft.",
                    data={
                        "action": action,
                        "expected_old_value": draft.get("old_value"),
                        "current_value": outcome.get("current_value"),
                    },
                )
            if outcome_status != "confirmed":
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="Admin change draft is already being confirmed or was changed; reload and try again.",
                )
            actor = _actor_context(self._store, user_id=user_id, actor_is_admin=True)
            snapshot = await _control_snapshot(
                self._store,
                actor=actor,
                user_client=self._user_client,
                args=args,
                include_available_default=False,
            )
            return ToolResult(
                status=ToolStatus.OK,
                tool=self.name,
                data={**snapshot, "action": action, "operation": "confirm", "confirmed": True},
            )

        claim = getattr(self._store, "claim_task_draft", None)
        release = getattr(self._store, "release_task_draft", None)
        if callable(claim) and draft_id and draft_version is not None:
            claimed = await claim(
                dialog_key,
                expected_draft_id=draft_id,
                expected_version=draft_version,
                expected_type=_DRAFT_TYPE,
            )
            if not isinstance(claimed, dict):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="Admin change draft is already being confirmed or was changed; reload and try again.",
                )
            draft = claimed
        current = _current_change_value(self._store, draft)
        if current != draft.get("old_value"):
            await _release_draft_claim(release, dialog_key=dialog_key, draft=draft)
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="Task-close control setting changed after draft creation; prepare a new draft.",
                data={"action": action, "expected_old_value": draft.get("old_value"), "current_value": current},
            )
        try:
            apply_error = _apply_change(self._store, draft=draft, actor_user_id=user_id)
        except Exception:
            await _release_draft_claim(release, dialog_key=dialog_key, draft=draft)
            return ToolResult(
                status=ToolStatus.ERROR, tool=self.name, error="Could not apply the reviewed admin change"
            )
        if apply_error is not None:
            await _release_draft_claim(release, dialog_key=dialog_key, draft=draft)
            return apply_error
        try:
            await self._store.delete_task_draft(
                dialog_key,
                status="confirmed",
                expected_draft_id=draft_id,
                expected_version=draft_version,
                expected_claim_token=str(draft.get("_draft_claim_token") or ""),
            )
        except Exception:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="The setting changed but the draft could not be finalized; manual audit is required.",
            )
        actor = _actor_context(self._store, user_id=user_id, actor_is_admin=True)
        snapshot = await _control_snapshot(
            self._store,
            actor=actor,
            user_client=self._user_client,
            args=args,
            include_available_default=False,
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={**snapshot, "action": action, "operation": "confirm", "confirmed": True},
        )

    async def _discard(self, *, user_id: int | None, dialog_key: str | None) -> ToolResult:
        if not dialog_key:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="dialog_key is required")
        draft = await self._store.get_task_draft(dialog_key, ttl_minutes=_DRAFT_TTL_MINUTES)
        if not isinstance(draft, dict) or draft.get("_draft_type") != _DRAFT_TYPE:
            return ToolResult(
                status=ToolStatus.NOT_FOUND, tool=self.name, error="Active admin change draft was not found"
            )
        draft_user_id = optional_int(draft.get("_draft_user_id") or draft.get("actor_user_id"))
        if draft_user_id is not None and draft_user_id != user_id:
            return _denied(self.name, user_id=user_id, reason="This admin change draft belongs to another user.")
        claim = getattr(self._store, "claim_task_draft", None)
        draft_id = str(draft.get("_draft_id") or "")
        draft_version = optional_int(draft.get("_draft_version"))
        if not callable(claim) or not draft_id or draft_version is None:
            return ToolResult(status=ToolStatus.ERROR, tool=self.name, error="Admin change draft has no CAS identity")
        claimed = await claim(
            dialog_key,
            expected_draft_id=draft_id,
            expected_version=draft_version,
            expected_type=_DRAFT_TYPE,
        )
        if not isinstance(claimed, dict):
            return ToolResult(status=ToolStatus.ERROR, tool=self.name, error="Admin change draft changed; reload it")
        await self._store.delete_task_draft(
            dialog_key,
            status="cancelled",
            expected_draft_id=draft_id,
            expected_version=draft_version,
            expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"action": str(draft.get("action") or ""), "operation": "discard", "discarded": True},
        )

    async def _fresh_admin_client(self, user_id: int | None) -> tuple[Any | None, ToolResult | None]:
        if self._bitrix_oauth is None:
            return None, _denied(
                self.name,
                user_id=user_id,
                reason="Fresh current-user Bitrix OAuth admin verification is required.",
            )
        read_client, access_actor, access_error = await resolve_current_user_read_client(
            self.name,
            fallback_client=self._user_client,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return None, access_error
        if user_id is None:
            return None, _denied(self.name, user_id=user_id, reason="Fresh Bitrix admin verification is unavailable.")
        try:
            raw_user = await read_client.get_user(user_id)
        except Exception:
            return None, _denied(self.name, user_id=user_id, reason="Fresh Bitrix admin verification failed.")
        profile = compact_user_profile(raw_user) if isinstance(raw_user, dict) else {}
        if optional_int(profile.get("id")) != user_id or not profile.get("active") or not profile.get("is_admin"):
            return None, _denied(
                self.name,
                user_id=user_id,
                reason=f"Only a currently verified Bitrix admin can change settings (actor={access_actor}).",
            )
        return read_client, None

    async def _prepare_change(
        self,
        args: dict[str, Any],
        *,
        action: str,
        user_id: int | None,
        user_client: Any,
    ) -> dict[str, Any] | ToolResult:
        draft: dict[str, Any] = {
            "_draft_type": _DRAFT_TYPE,
            "_original_request": str(args.get("_original_request") or ""),
            "_draft_user_id": user_id,
            "_draft_specialist": str(args.get("_draft_specialist") or "bitrix24"),
            "actor_user_id": user_id,
            "action": action,
        }
        if action in _CONTROLLED_USER_ACTIONS | {"add_operator", "remove_operator"}:
            target = await _resolve_target_user(user_client, args)
            if isinstance(target, ToolResult):
                return target
            draft.update(target)
        value_error = _populate_change_values(self._store, draft=draft, args=args)
        if value_error is not None:
            return value_error
        return draft

    async def _revalidate_target(self, draft: dict[str, Any], *, user_client: Any) -> ToolResult | None:
        target_user_id = optional_int(draft.get("target_user_id"))
        if target_user_id is None:
            return None
        try:
            raw_user = await user_client.get_user(target_user_id)
        except Exception:
            return ToolResult(
                status=ToolStatus.ERROR, tool=self.name, error="Could not revalidate the target Bitrix user"
            )
        profile = _compact_user(raw_user) if isinstance(raw_user, dict) else None
        if profile is None or not profile.get("active"):
            return ToolResult(
                status=ToolStatus.NOT_FOUND, tool=self.name, error="Target Bitrix user is no longer active"
            )
        return None


async def _resolve_target_user(user_client: Any | None, args: dict[str, Any]) -> dict[str, Any] | ToolResult:
    if user_client is None:
        return ToolResult(
            status=ToolStatus.NOT_AVAILABLE,
            tool=TaskCloseControlUpdateTool.name,
            error="Bitrix user lookup is unavailable",
        )
    target_user_id = _target_user_id(args)
    if target_user_id is not None:
        try:
            raw_user = await user_client.get_user(target_user_id)
        except Exception:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=TaskCloseControlUpdateTool.name,
                error="Could not load the target Bitrix user",
            )
        profile = _compact_user(raw_user) if isinstance(raw_user, dict) else None
        if profile is None or not profile.get("active"):
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=TaskCloseControlUpdateTool.name,
                error="Target Bitrix user was not found or is inactive",
            )
        return {"target_user_id": target_user_id, "target_user_name": profile["name"]}

    target_name = str(args.get("target_user_name") or args.get("user_name") or "").strip()
    if not target_name:
        return _missing_target_user()
    search = getattr(user_client, "search_users", None)
    if not callable(search):
        return ToolResult(
            status=ToolStatus.NOT_AVAILABLE,
            tool=TaskCloseControlUpdateTool.name,
            error="Bitrix exact-name search is unavailable",
        )
    try:
        raw_users = await search(target_name, limit=20)
    except Exception:
        return ToolResult(
            status=ToolStatus.ERROR,
            tool=TaskCloseControlUpdateTool.name,
            error="Could not search Bitrix users",
        )
    normalized_name = _normalized_name(target_name)
    matches: list[dict[str, Any]] = []
    for raw_user in raw_users if isinstance(raw_users, list) else []:
        profile = _compact_user(raw_user) if isinstance(raw_user, dict) else None
        if profile and profile.get("active") and _normalized_name(profile.get("name")) == normalized_name:
            matches.append(profile)
    matches = sorted({int(item["user_id"]): item for item in matches}.values(), key=lambda item: int(item["user_id"]))
    if not matches:
        return ToolResult(
            status=ToolStatus.NOT_FOUND,
            tool=TaskCloseControlUpdateTool.name,
            error="No active Bitrix user has that exact name",
            data={"target_user_name": target_name},
        )
    if len(matches) > 1:
        return ToolResult(
            status=ToolStatus.AMBIGUOUS,
            tool=TaskCloseControlUpdateTool.name,
            error="Several active Bitrix users have that exact name; specify the numeric user id",
            data={"target_user_name": target_name, "matches": matches},
        )
    return {"target_user_id": int(matches[0]["user_id"]), "target_user_name": str(matches[0]["name"])}


def _populate_change_values(store: Any, *, draft: dict[str, Any], args: dict[str, Any]) -> ToolResult | None:
    action = str(draft.get("action") or "")
    target_user_id = optional_int(draft.get("target_user_id"))
    if action in {"add_operator", "remove_operator"}:
        if target_user_id is None:
            return _missing_target_user()
        draft["field"] = "operator"
        draft["old_value"] = target_user_id in set(_ids_from_store(store, "task_close_operator_ids"))
        draft["new_value"] = action == "add_operator"
        return None
    if action in _CONTROLLED_USER_ACTIONS:
        if target_user_id is None:
            return _missing_target_user()
        draft["field"] = "controlled_user"
        draft["old_value"] = target_user_id in set(_ids_from_store(store, "task_close_controlled_user_ids"))
        draft["new_value"] = action == "add_controlled_user"
        return None
    if action == "set_auto_close_time":
        value = str(args.get("auto_close_time") or args.get("value") or "").strip()
        if not _TIME_RE.match(value):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=TaskCloseControlUpdateTool.name,
                error="auto_close_time must be in HH:MM format",
            )
        draft["field"] = TASK_CLOSE_AUTO_CLOSE_TIME_KEY
        draft["old_value"] = _setting_value(store, TASK_CLOSE_AUTO_CLOSE_TIME_KEY, TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME)
        draft["new_value"] = value
        return None
    value = str(args.get("control_enabled_from") or args.get("value") or "").strip()
    if not value:
        return ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=TaskCloseControlUpdateTool.name,
            error="control_enabled_from is required",
        )
    if not _valid_iso_datetime_or_date(value):
        return ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=TaskCloseControlUpdateTool.name,
            error="control_enabled_from must be an ISO date or datetime",
        )
    draft["field"] = TASK_CLOSE_CONTROL_ENABLED_FROM_KEY
    draft["old_value"] = _setting_value(store, TASK_CLOSE_CONTROL_ENABLED_FROM_KEY, "")
    draft["new_value"] = value
    return None


def _current_change_value(store: Any, draft: dict[str, Any]) -> object:
    field = str(draft.get("field") or "")
    target_user_id = optional_int(draft.get("target_user_id"))
    if field == "operator":
        return bool(
            target_user_id is not None and target_user_id in set(_ids_from_store(store, "task_close_operator_ids"))
        )
    if field == "controlled_user":
        return bool(
            target_user_id is not None
            and target_user_id in set(_ids_from_store(store, "task_close_controlled_user_ids"))
        )
    if field == TASK_CLOSE_AUTO_CLOSE_TIME_KEY:
        return _setting_value(store, field, TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME)
    if field == TASK_CLOSE_CONTROL_ENABLED_FROM_KEY:
        return _setting_value(store, field, "")
    return None


def _apply_change(store: Any, *, draft: dict[str, Any], actor_user_id: int | None) -> ToolResult | None:
    action = str(draft.get("action") or "")
    target_user_id = optional_int(draft.get("target_user_id"))
    if action in {"add_operator", "remove_operator"}:
        if target_user_id is None:
            return _missing_target_user()
        setter = getattr(store, "upsert_task_close_operator", None)
        if not callable(setter):
            return _store_not_supported()
        setter(user_id=target_user_id, active=action == "add_operator", updated_by=actor_user_id)
        return None
    if action in _CONTROLLED_USER_ACTIONS:
        if target_user_id is None:
            return _missing_target_user()
        setter = getattr(store, "upsert_task_close_controlled_user", None)
        if not callable(setter):
            return _store_not_supported()
        setter(user_id=target_user_id, active=bool(draft.get("new_value")), updated_by=actor_user_id)
        return None
    if action in {"set_auto_close_time", "set_control_enabled_from"}:
        return _set_setting(
            store,
            key=str(draft.get("field") or ""),
            value=str(draft.get("new_value") or ""),
            actor_user_id=actor_user_id,
        )
    return ToolResult(
        status=ToolStatus.INVALID_TOOL_CALL, tool=TaskCloseControlUpdateTool.name, error="Invalid draft action"
    )


def _normalized_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


async def _release_draft_claim(release: Any, *, dialog_key: str, draft: dict[str, Any]) -> None:
    draft_id = str(draft.get("_draft_id") or "")
    if callable(release) and draft_id:
        await release(
            dialog_key,
            draft_id=draft_id,
            claim_token=str(draft.get("_draft_claim_token") or ""),
        )


def _set_setting(store: Any, *, key: str, value: str, actor_user_id: int | None) -> ToolResult | None:
    setter = getattr(store, "set_task_close_control_setting", None)
    if not callable(setter):
        return _store_not_supported()
    setter(key=key, value=value, updated_by=actor_user_id)
    return None


async def _control_snapshot(
    store: Any,
    *,
    actor: dict[str, Any],
    user_client: Any | None = None,
    args: dict[str, Any] | None = None,
    include_available_default: bool,
) -> dict[str, Any]:
    operator_ids = _ids_from_store(store, "task_close_operator_ids")
    controlled_user_ids = _ids_from_store(store, "task_close_controlled_user_ids")
    auto_close_time = _setting_value(store, TASK_CLOSE_AUTO_CLOSE_TIME_KEY, TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME)
    control_enabled_from = _setting_value(store, TASK_CLOSE_CONTROL_ENABLED_FROM_KEY, "")
    member_ids = sorted(set(operator_ids) | set(controlled_user_ids))
    directory = await _load_user_directory(
        user_client,
        member_ids=member_ids,
        operator_ids=set(operator_ids),
        controlled_user_ids=set(controlled_user_ids),
        args=args or {},
        include_available_default=include_available_default,
    )
    return {
        "operator_user_ids": operator_ids,
        "controlled_user_ids": controlled_user_ids,
        "members": [
            _user_entry(
                user_id=item,
                profile=directory["profiles"].get(item),
                operator_ids=set(operator_ids),
                controlled_user_ids=set(controlled_user_ids),
            )
            for item in member_ids
        ],
        "available_users": [
            _user_entry(
                user_id=int(profile["user_id"]),
                profile=profile,
                operator_ids=set(operator_ids),
                controlled_user_ids=set(controlled_user_ids),
            )
            for profile in directory["available_users"]
        ],
        "available_users_truncated": directory["available_users_truncated"],
        "available_users_limit": directory["available_users_limit"],
        "available_users_query": directory["available_users_query"],
        "user_lookup_status": directory["status"],
        "user_lookup_error": directory["error"],
        "auto_close_time": auto_close_time,
        "control_enabled_from": control_enabled_from,
        "actor_role": actor["role"],
    }


def _actor_context(store: Any, *, user_id: int | None, actor_is_admin: bool) -> dict[str, Any]:
    is_operator = bool(user_id is not None and user_id in set(_ids_from_store(store, "task_close_operator_ids")))
    role = "admin" if actor_is_admin else "operator" if is_operator else "user"
    return {"is_admin": actor_is_admin, "is_operator": is_operator, "role": role}


def _ids_from_store(store: Any, method_name: str) -> list[int]:
    getter = getattr(store, method_name, None)
    if not callable(getter):
        return []
    return sorted({int(item) for item in getter() if optional_int(item) is not None and int(item) > 0})


def _setting_value(store: Any, key: str, default: str) -> str:
    getter = getattr(store, "get_task_close_control_setting", None)
    if not callable(getter):
        return default
    setting = getter(key)
    if not isinstance(setting, dict):
        return default
    return str(setting.get("value") or default)


async def _load_user_directory(
    user_client: Any | None,
    *,
    member_ids: list[int],
    operator_ids: set[int],
    controlled_user_ids: set[int],
    args: dict[str, Any],
    include_available_default: bool,
) -> dict[str, Any]:
    limit = _available_limit(args)
    include_available = _include_available_users(args, default=include_available_default) and limit > 0
    query = str(args.get("user_query") or args.get("query") or "").strip()
    profiles: dict[int, dict[str, Any]] = {}
    available_users: list[dict[str, Any]] = []
    truncated = False
    status = "not_configured" if user_client is None else "ok"
    error = ""

    if user_client is not None and include_available:
        try:
            raw_users = await _load_available_users(user_client, query=query, limit=limit + 1)
            truncated = len(raw_users) > limit
            for raw in raw_users[:limit]:
                profile = _compact_user(raw)
                if profile is None:
                    continue
                user_id = int(profile["user_id"])
                profiles[user_id] = profile
                available_users.append(profile)
        except Exception as exc:  # pragma: no cover - exact Bitrix errors depend on portal/runtime
            status = "failed"
            error = str(exc)

    if user_client is not None:
        for user_id in member_ids:
            if user_id in profiles:
                continue
            try:
                user = await user_client.get_user(user_id)
            except Exception as exc:  # pragma: no cover - exact Bitrix errors depend on portal/runtime
                if status == "ok":
                    status = "partial"
                    error = str(exc)
                continue
            profile = _compact_user(user) if isinstance(user, dict) else None
            if profile is not None:
                profiles[int(profile["user_id"])] = profile

    available_users = sorted(
        available_users,
        key=lambda item: (
            int(item["user_id"]) not in operator_ids and int(item["user_id"]) not in controlled_user_ids,
            _user_name(item).casefold(),
            int(item["user_id"]),
        ),
    )
    return {
        "profiles": profiles,
        "available_users": available_users,
        "available_users_truncated": truncated,
        "available_users_limit": limit,
        "available_users_query": query,
        "status": status,
        "error": error,
    }


async def _load_available_users(user_client: Any, *, query: str, limit: int) -> list[dict[str, Any]]:
    if query:
        search = getattr(user_client, "search_users", None)
        if callable(search):
            users = await search(query, limit=limit)
            return [item for item in users if isinstance(item, dict)]
    list_all = getattr(user_client, "list_all_users", None)
    if callable(list_all):
        users = await list_all(
            filter_={"ACTIVE": True},
            select=["ID", "NAME", "LAST_NAME", "SECOND_NAME", "EMAIL", "WORK_POSITION", "ACTIVE"],
            limit=limit,
        )
        return [item for item in users if isinstance(item, dict)]
    search = getattr(user_client, "search_users", None)
    if callable(search):
        users = await search(query, limit=limit)
        return [item for item in users if isinstance(item, dict)]
    return []


def _compact_user(user: dict[str, Any]) -> dict[str, Any] | None:
    profile = compact_user_profile(user)
    user_id = optional_int(profile.get("id"))
    if user_id is None or user_id <= 0:
        return None
    return {
        "user_id": user_id,
        "name": str(profile.get("label") or f"Bitrix user #{user_id}"),
        "active": bool(profile.get("active", True)),
        "work_position": str(profile.get("work_position") or ""),
    }


def _user_entry(
    *,
    user_id: int,
    profile: dict[str, Any] | None,
    operator_ids: set[int],
    controlled_user_ids: set[int],
) -> dict[str, Any]:
    is_operator = user_id in operator_ids
    is_controlled = user_id in controlled_user_ids
    roles: list[str] = []
    if is_operator:
        roles.append("operator")
    if is_controlled:
        roles.append("controlled_user")
    return {
        "user_id": user_id,
        "name": _user_name(profile) if profile else f"Bitrix user #{user_id}",
        "roles": roles,
        "is_operator": is_operator,
        "is_controlled": is_controlled,
        "can_add_operator": not is_operator,
        "can_add_controlled": not is_controlled,
        "active": bool((profile or {}).get("active", True)),
        "work_position": str((profile or {}).get("work_position") or ""),
    }


def _user_name(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    return str(profile.get("name") or profile.get("label") or "").strip()


def _include_available_users(args: dict[str, Any], *, default: bool) -> bool:
    if "include_available_users" not in args:
        return default
    return _truthy(args.get("include_available_users"))


def _available_limit(args: dict[str, Any]) -> int:
    limit = optional_int(args.get("available_limit") or args.get("limit"))
    if limit is None:
        return 100
    return max(0, min(limit, 200))


def _target_user_id(args: dict[str, Any]) -> int | None:
    return optional_int(
        args.get("target_user_id")
        or args.get("user_id")
        or args.get("operator_user_id")
        or args.get("controlled_user_id")
    )


def _valid_iso_datetime_or_date(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _missing_target_user() -> ToolResult:
    return ToolResult(
        status=ToolStatus.INVALID_TOOL_CALL,
        tool=TaskCloseControlUpdateTool.name,
        error="target_user_id is required for this action",
    )


def _store_not_supported() -> ToolResult:
    return ToolResult(
        status=ToolStatus.NOT_CONFIGURED,
        tool=TaskCloseControlUpdateTool.name,
        error="Bitrix store does not support task close control settings",
    )


def _denied(tool: str, *, user_id: int | None, reason: str = "") -> ToolResult:
    return ToolResult(
        status=ToolStatus.DENIED,
        tool=tool,
        error=reason or "Only Bitrix admins and task close operators can manage this control block.",
        data={"user_id": user_id},
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


__all__ = [
    "TASK_CLOSE_AUTO_CLOSE_TIME_KEY",
    "TASK_CLOSE_CONTROL_ENABLED_FROM_KEY",
    "TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME",
    "TaskCloseControlGetTool",
    "TaskCloseControlUpdateTool",
]
