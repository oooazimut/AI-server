from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.utils import optional_int

TASK_CLOSE_AUTO_CLOSE_TIME_KEY = "auto_close_time"
TASK_CLOSE_CONTROL_ENABLED_FROM_KEY = "control_enabled_from"
TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME = "20:00"

_CONTROLLED_USER_ACTIONS = {"add_controlled_user", "remove_controlled_user"}
_ADMIN_ONLY_ACTIONS = {"add_operator", "remove_operator", "set_auto_close_time", "set_control_enabled_from"}
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class TaskCloseControlGetTool:
    name = "task_close_control_get"

    def __init__(self, store: Any | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read Bitrix task close control settings: operators, controlled users, auto-close time, "
                "and control start date. Admins and task-close operators only."
            ),
            parameters={"type": "object", "properties": {}},
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
        return ToolResult(status=ToolStatus.OK, tool=self.name, data=_control_snapshot(self._store, actor=actor))


class TaskCloseControlUpdateTool:
    name = "task_close_control_update"

    def __init__(self, store: Any | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Update Bitrix task close control settings. Admins may manage operators, controlled users, "
                "auto-close time, and control start date. Operators may only add/remove controlled users."
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
                    "target_user_id": {"type": "integer"},
                    "value": {"type": "string"},
                    "auto_close_time": {"type": "string", "description": "HH:MM, for example 20:00."},
                    "control_enabled_from": {"type": "string", "description": "ISO datetime/date for control start."},
                },
                "required": ["action"],
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
        action = str(args.get("action") or "").strip()
        if action not in _CONTROLLED_USER_ACTIONS | _ADMIN_ONLY_ACTIONS:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_control_update.action is invalid",
            )
        actor = _actor_context(self._store, user_id=user_id, actor_is_admin=_truthy(args.get("_actor_is_admin")))
        if action in _ADMIN_ONLY_ACTIONS and not actor["is_admin"]:
            return _denied(self.name, user_id=user_id, reason="Only Bitrix admins can change this setting.")
        if action in _CONTROLLED_USER_ACTIONS and not actor["is_admin"] and not actor["is_operator"]:
            return _denied(self.name, user_id=user_id)

        if action in {"add_operator", "remove_operator"}:
            result = _update_operator(self._store, args=args, action=action, actor_user_id=user_id)
        elif action in _CONTROLLED_USER_ACTIONS:
            result = _update_controlled_user(self._store, args=args, action=action, actor_user_id=user_id)
        elif action == "set_auto_close_time":
            result = _set_auto_close_time(self._store, args=args, actor_user_id=user_id)
        else:
            result = _set_control_enabled_from(self._store, args=args, actor_user_id=user_id)

        if result is not None:
            return result
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={**_control_snapshot(self._store, actor=actor), "action": action},
        )


def _update_operator(store: Any, *, args: dict[str, Any], action: str, actor_user_id: int | None) -> ToolResult | None:
    target_user_id = _target_user_id(args)
    if target_user_id is None:
        return _missing_target_user()
    getter = getattr(store, "task_close_operator_ids", None)
    setter = getattr(store, "set_task_close_operators", None)
    if not callable(getter) or not callable(setter):
        return _store_not_supported()
    operator_ids = set(getter())
    if action == "add_operator":
        operator_ids.add(target_user_id)
    else:
        operator_ids.discard(target_user_id)
    setter(operator_user_ids=sorted(operator_ids), actor_user_id=actor_user_id)
    return None


def _update_controlled_user(
    store: Any, *, args: dict[str, Any], action: str, actor_user_id: int | None
) -> ToolResult | None:
    target_user_id = _target_user_id(args)
    if target_user_id is None:
        return _missing_target_user()
    setter = getattr(store, "upsert_task_close_controlled_user", None)
    if not callable(setter):
        return _store_not_supported()
    setter(user_id=target_user_id, active=action == "add_controlled_user", updated_by=actor_user_id)
    return None


def _set_auto_close_time(store: Any, *, args: dict[str, Any], actor_user_id: int | None) -> ToolResult | None:
    value = str(args.get("auto_close_time") or args.get("value") or "").strip()
    if not _TIME_RE.match(value):
        return ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=TaskCloseControlUpdateTool.name,
            error="auto_close_time must be in HH:MM format",
        )
    return _set_setting(store, key=TASK_CLOSE_AUTO_CLOSE_TIME_KEY, value=value, actor_user_id=actor_user_id)


def _set_control_enabled_from(store: Any, *, args: dict[str, Any], actor_user_id: int | None) -> ToolResult | None:
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
    return _set_setting(store, key=TASK_CLOSE_CONTROL_ENABLED_FROM_KEY, value=value, actor_user_id=actor_user_id)


def _set_setting(store: Any, *, key: str, value: str, actor_user_id: int | None) -> ToolResult | None:
    setter = getattr(store, "set_task_close_control_setting", None)
    if not callable(setter):
        return _store_not_supported()
    setter(key=key, value=value, updated_by=actor_user_id)
    return None


def _control_snapshot(store: Any, *, actor: dict[str, Any]) -> dict[str, Any]:
    operator_ids = _ids_from_store(store, "task_close_operator_ids")
    controlled_user_ids = _ids_from_store(store, "task_close_controlled_user_ids")
    auto_close_time = _setting_value(store, TASK_CLOSE_AUTO_CLOSE_TIME_KEY, TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME)
    control_enabled_from = _setting_value(store, TASK_CLOSE_CONTROL_ENABLED_FROM_KEY, "")
    return {
        "operator_user_ids": operator_ids,
        "controlled_user_ids": controlled_user_ids,
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
