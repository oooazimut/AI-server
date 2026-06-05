from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.runtime import runtime_paths
from ai_server.settings import get_settings
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy


CONFIRM_WORDS = {
    "да",
    "ок",
    "окей",
    "подтверждаю",
    "подтвердить",
    "создай",
    "выполняй",
    "выполнить",
    "делай",
    "запускай",
    "согласен",
}
CANCEL_WORDS = {"отмена", "отмени", "не надо", "сбрось", "не создавай", "не выполняй", "cancel"}
NO_DEADLINE_FIELD_NAMES = {
    "NO_DEADLINE",
    "WITHOUT_DEADLINE",
    "noDeadline",
    "withoutDeadline",
}


@dataclass
class PendingBitrixAction:
    method: str
    params: dict[str, Any]
    summary: str
    created_by: int | None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BitrixDialogState:
    key: str
    summary: str = ""
    turns: list[dict[str, str]] = field(default_factory=list)
    pending_action: PendingBitrixAction | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "turns": self.turns,
            "pending_action": asdict(self.pending_action) if self.pending_action else None,
        }


@dataclass(frozen=True)
class PendingActionResult:
    status: str
    message: str
    action: PendingBitrixAction | None = None
    data: dict[str, Any] = field(default_factory=dict)


class DialogStateStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else runtime_paths().dialog_state_db

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dialog_states (
                    dialog_key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_dialog_states_updated ON dialog_states(updated_at)"
            )

    def load_raw(self, key: str) -> dict[str, Any] | None:
        self.ensure_schema()
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT state_json FROM dialog_states WHERE dialog_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def save_raw(self, key: str, state: dict[str, Any]) -> None:
        self.ensure_schema()
        updated_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO dialog_states (dialog_key, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(dialog_key) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(state, ensure_ascii=False), updated_at),
            )

    def load(self, key: str) -> BitrixDialogState:
        return state_from_dict(key, self.load_raw(key))

    def save(self, state: BitrixDialogState) -> None:
        self.save_raw(state.key, state.as_dict())

    def set_pending(self, key: str, action: PendingBitrixAction) -> BitrixDialogState:
        state = self.load(key)
        state.pending_action = action
        self.save(state)
        return state

    def clear_pending(self, key: str) -> BitrixDialogState:
        state = self.load(key)
        state.pending_action = None
        self.save(state)
        return state


class BitrixPendingActionService:
    def __init__(
        self,
        *,
        store: DialogStateStore | None = None,
        bitrix: BitrixClient | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        audit_log_path: Path | str | None = None,
    ) -> None:
        self.store = store or DialogStateStore()
        self.bitrix = bitrix or BitrixClient()
        self.bitrix_oauth = bitrix_oauth
        self.audit_log_path = (
            Path(audit_log_path)
            if audit_log_path is not None
            else runtime_paths().bitrix_write_audit_log
        )

    def pending_for(self, key: str) -> PendingBitrixAction | None:
        return self.store.load(key).pending_action

    def save_pending(self, key: str, action: PendingBitrixAction) -> None:
        self.store.set_pending(key, action)

    def cancel(self, key: str) -> PendingActionResult:
        action = self.pending_for(key)
        if not action:
            return PendingActionResult(status="nothing_to_cancel", message="Нет ожидающего действия для отмены.")
        self.store.clear_pending(key)
        self._audit("cancelled", key=key, action=action)
        return PendingActionResult(
            status="cancelled",
            message=f"Ок, отменил ожидающее действие: {action.summary}.",
            action=action,
        )

    async def confirm(self, key: str, *, user_id: int | None = None) -> PendingActionResult:
        action = self.pending_for(key)
        if not action:
            return PendingActionResult(status="nothing_to_confirm", message="Нет ожидающего действия для подтверждения.")
        if action.created_by and user_id and action.created_by != user_id:
            return PendingActionResult(
                status="denied",
                message="Подтвердить действие может только пользователь, который его запросил.",
                action=action,
            )

        if action.method == "ai_server.task_closure":
            return await self._confirm_task_closure(key, action=action, user_id=user_id)

        decision = decide_bitrix_method_policy(action.method)
        if decision.decision != "confirm":
            self.store.clear_pending(key)
            self._audit(
                "denied",
                key=key,
                action=action,
                data={"reason": f"unexpected method policy: {decision.decision}", "policy_reason": decision.reason},
            )
            return PendingActionResult(
                status="denied",
                message="Не выполнил действие: метод больше не проходит политику подтверждаемых Bitrix-записей.",
                action=action,
                data={"policy": decision.model_dump()},
            )

        if not can_prepare_write_action(action.method, action.params, user_id):
            self._audit(
                "denied",
                key=key,
                action=action,
                data={"reason": "user is not allowed to write Bitrix through agent"},
            )
            return PendingActionResult(
                status="denied",
                message=(
                    "Не выполнил действие: у пользователя нет права на Bitrix-запись через агента."
                ),
                action=action,
            )

        params = apply_write_policy(action.method, deepcopy(action.params), user_id)
        write_client = await self._write_client(user_id, action=action, key=key)
        if isinstance(write_client, PendingActionResult):
            return write_client

        try:
            data = await write_client.call(action.method, params)
        except Exception as exc:
            self._audit("error", key=key, action=action, data={"error": f"{type(exc).__name__}: {exc}"})
            return PendingActionResult(
                status="error",
                message=f"Не смог выполнить действие в Bitrix: {type(exc).__name__}: {exc}",
                action=action,
            )

        self.store.clear_pending(key)
        result = data.get("result") if isinstance(data, dict) else data
        self._audit("executed", key=key, action=action, data={"result": result})
        return PendingActionResult(
            status="executed",
            message=f"Готово, выполнил в Bitrix: {action.summary}.",
            action=action,
            data={"result": result},
        )

    async def _write_client(
        self,
        user_id: int | None,
        *,
        action: PendingBitrixAction,
        key: str,
    ) -> BitrixClient | PendingActionResult:
        settings = get_settings()
        if not settings.bitrix_oauth_enabled or self.bitrix_oauth is None or not user_id:
            if settings.bitrix_oauth_required_for_writes:
                self._audit("oauth_required", key=key, action=action)
                return PendingActionResult(
                    status="oauth_required",
                    message="Для записи в Bitrix нужна OAuth-авторизация пользователя.",
                    action=action,
                )
            return self.bitrix

        try:
            return await self.bitrix_oauth.client_for_user(user_id)
        except (BitrixOAuthTokenMissing, BitrixApiError, BitrixConfigError) as exc:
            if settings.bitrix_oauth_required_for_writes:
                self._audit("oauth_required", key=key, action=action, data={"error": str(exc)})
                return PendingActionResult(
                    status="oauth_required",
                    message="Для записи в Bitrix нужна OAuth-авторизация пользователя.",
                    action=action,
                )
            return self.bitrix

    async def _confirm_task_closure(
        self,
        key: str,
        *,
        action: PendingBitrixAction,
        user_id: int | None,
    ) -> PendingActionResult:
        if user_id is None:
            self._audit("denied", key=key, action=action, data={"reason": "missing Bitrix user id"})
            self.store.clear_pending(key)
            return PendingActionResult(
                status="denied",
                message="Не выполнил действие: не вижу ID текущего пользователя Bitrix.",
                action=action,
            )

        write_client = await self._write_client(user_id, action=action, key=key)
        if isinstance(write_client, PendingActionResult):
            return write_client
        actor_client = await self._task_closure_actor_client(user_id, fallback=write_client)

        from ai_server.agents.bitrix_task_closure import TaskClosureService

        service = TaskClosureService(
            write_client,
            PortalSearchIndex(),
            actor_bitrix=actor_client,
        )
        try:
            data = await service.execute(action.params, current_user_id=user_id)
        except Exception as exc:
            self._audit("error", key=key, action=action, data={"error": f"{type(exc).__name__}: {exc}"})
            return PendingActionResult(
                status="error",
                message=f"Не смог закрыть задачу в Bitrix: {type(exc).__name__}: {exc}",
                action=action,
            )

        status = str(data.get("status") or "executed")
        if status != "error":
            self.store.clear_pending(key)
        audit_status = "executed" if status == "executed" else status
        self._audit(audit_status, key=key, action=action, data=data)
        return PendingActionResult(
            status=status,
            message=_task_closure_result_message(data),
            action=action,
            data=data,
        )

    async def _task_closure_actor_client(
        self,
        user_id: int,
        *,
        fallback: BitrixClient,
    ) -> BitrixClient:
        settings = get_settings()
        actor_user_id = settings.quality_control_actor_user_id
        if not actor_user_id or not settings.bitrix_oauth_enabled or self.bitrix_oauth is None:
            return fallback
        if actor_user_id == user_id:
            return fallback
        try:
            return await self.bitrix_oauth.client_for_user(actor_user_id)
        except (BitrixOAuthTokenMissing, BitrixApiError, BitrixConfigError):
            return fallback

    def _audit(
        self,
        status: str,
        *,
        key: str,
        action: PendingBitrixAction,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dialog_key": key,
            "status": status,
            "method": action.method,
            "params": action.params,
            "summary": action.summary,
            "created_by": action.created_by,
            "created_at": action.created_at,
            "data": data or {},
        }
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_dialog_key(*, chat_id: int | None = None, dialog_id: str = "", user_id: int | None = None) -> str:
    resolved_user_id = user_id or 0
    if chat_id:
        return f"chat:{chat_id}:user:{resolved_user_id}"
    if dialog_id:
        return f"dialog:{dialog_id}:user:{resolved_user_id}"
    return f"user:{resolved_user_id}"


def normalize_control_text(text: str) -> str:
    value = text.strip().lower().replace("ё", "е")
    value = re.sub(r"[.!?,:;]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def is_confirm_text(text: str) -> bool:
    return normalize_control_text(text) in CONFIRM_WORDS


def is_cancel_text(text: str) -> bool:
    return normalize_control_text(text) in CANCEL_WORDS


def state_from_dict(key: str, data: dict[str, Any] | None) -> BitrixDialogState:
    data = data or {}
    turns = data.get("turns") if isinstance(data.get("turns"), list) else []
    return BitrixDialogState(
        key=key,
        summary=str(data.get("summary") or ""),
        turns=[turn for turn in turns if isinstance(turn, dict)],
        pending_action=pending_bitrix_from_dict(data.get("pending_action")),
    )


def pending_bitrix_from_dict(data: object) -> PendingBitrixAction | None:
    if not isinstance(data, dict):
        return None
    params = data.get("params")
    method = str(data.get("method") or "").strip()
    if not method:
        return None
    return PendingBitrixAction(
        method=method,
        params=params if isinstance(params, dict) else {},
        summary=str(data.get("summary") or method),
        created_by=optional_int(data.get("created_by")),
        created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
    )


def optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def can_prepare_write_action(method: str, params: dict[str, Any], user_id: int | None) -> bool:
    settings = get_settings()
    if user_id in settings.resolved_agent_write_allowed_user_ids:
        return True
    normalized = method.strip().lower()
    if normalized != "tasks.task.add":
        return False
    allowed_project_id = settings.agent_limited_task_create_project_id
    if allowed_project_id is None or user_id not in settings.resolved_agent_limited_task_create_user_ids:
        return False
    requested_group_id = task_add_group_id(params)
    return requested_group_id is None or requested_group_id == allowed_project_id


def apply_write_policy(method: str, params: dict[str, Any], user_id: int | None) -> dict[str, Any]:
    settings = get_settings()
    normalized = method.strip().lower()
    if normalized != "tasks.task.add":
        return params
    fields = params.get("fields")
    if not isinstance(fields, dict):
        return params
    if no_deadline_requested(fields):
        prepare_no_deadline_fields(fields)
    allowed_project_id = settings.agent_limited_task_create_project_id
    if allowed_project_id is None or user_id not in settings.resolved_agent_limited_task_create_user_ids:
        return params
    if task_add_group_id(params) is None:
        fields["GROUP_ID"] = allowed_project_id
    return params


def task_add_group_id(params: dict[str, Any]) -> int | None:
    fields = params.get("fields")
    if not isinstance(fields, dict):
        return None
    for key in ("GROUP_ID", "groupId", "group_id"):
        value = fields.get(key)
        try:
            return int(str(value).strip()) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None
    return None


def no_deadline_requested(fields: dict[str, Any]) -> bool:
    for key in NO_DEADLINE_FIELD_NAMES:
        if _truthy(fields.get(key)):
            return True
    for key, value in fields.items():
        if key.lower() != "deadline":
            continue
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "none",
            "null",
            "no",
            "n",
            "без срока",
            "бессрочно",
        }:
            return True
    return False


def prepare_no_deadline_fields(fields: dict[str, Any]) -> None:
    for key in list(fields):
        if key in NO_DEADLINE_FIELD_NAMES or key.lower() == "deadline":
            fields.pop(key, None)
    fields["DEADLINE"] = ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "да", "без срока", "бессрочно"}
    return bool(value)


def _task_closure_result_message(data: dict[str, Any]) -> str:
    status = str(data.get("status") or "")
    outcome = str(data.get("outcome") or "")
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    task_id = task.get("id") if isinstance(task, dict) else None
    if status == "executed" and outcome == "closed":
        return f"Готово, закрыл задачу #{task_id}." if task_id else "Готово, закрыл задачу."
    if status == "executed" and outcome == "already_closed":
        return f"Задача #{task_id} уже была закрыта." if task_id else "Задача уже была закрыта."
    if status == "executed" and outcome == "needs_revision":
        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        issue_text = "; ".join(str(issue) for issue in issues[:3] if str(issue).strip())
        base = (
            f"Задачу #{task_id} не закрыл: результат нужно доработать."
            if task_id
            else "Задачу не закрыл: результат нужно доработать."
        )
        return f"{base} {issue_text}".strip()
    if status == "ambiguous":
        return str(data.get("message") or "Нашёл несколько похожих задач. Укажите номер нужной задачи.")
    if status == "not_found":
        return str(data.get("message") or "Не нашёл подходящую задачу.")
    if status == "denied":
        return str(data.get("reason") or "Не выполнил действие: недостаточно прав.")
    if status == "contract_violation":
        return str(data.get("error") or "Не выполнил действие: не хватает данных для закрытия задачи.")
    return str(data.get("message") or "Действие обработано.")
