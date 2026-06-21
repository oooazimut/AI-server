from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ai_server.agent_store import SqliteStore
from ai_server.integrations.bitrix.bitrix_policy import decide_bitrix_method_policy
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.ports import BitrixUserPort, BitrixWritePort
from ai_server.integrations.bitrix.profile import compact_user_profile
from ai_server.runtime import runtime_paths
from ai_server.settings import Settings, get_settings
from ai_server.utils import optional_int, truthy

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
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    specialist_id: str = "bitrix24"


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


class DialogStateStore(SqliteStore):
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else runtime_paths().dialog_state_db

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dialog_states (
                    dialog_key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_dialog_states_updated ON dialog_states(updated_at)")

    def load_raw(self, key: str) -> dict[str, Any] | None:
        self.ensure_schema()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM dialog_states WHERE dialog_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(str(row["state_json"]))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def save_raw(self, key: str, state: dict[str, Any]) -> None:
        self.ensure_schema()
        updated_at = datetime.now(UTC).isoformat()
        with self._connection() as connection:
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

    def append_turn(self, key: str, user_text: str, agent_response: str) -> None:
        state = self.load(key)
        state.turns.append({"role": "user", "content": user_text})
        state.turns.append({"role": "assistant", "content": agent_response})
        state.turns = state.turns[-20:]
        self.save(state)


class BitrixPendingActionService:
    def __init__(
        self,
        *,
        store: DialogStateStore,
        bitrix: BitrixWritePort,
        bitrix_oauth: BitrixOAuthService | None = None,
        audit_log_path: Path | str | None = None,
        dry_run: bool = False,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.store = store
        self.bitrix = bitrix
        self.bitrix_oauth = bitrix_oauth
        self.dry_run = dry_run
        self.audit_log_path = (
            Path(audit_log_path) if audit_log_path is not None else runtime_paths().bitrix_write_audit_log
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

    def _validate_ownership(self, action: PendingBitrixAction, user_id: int | None) -> PendingActionResult | None:
        if action.created_by and user_id and action.created_by != user_id:
            return PendingActionResult(
                status="denied",
                message="Подтвердить действие может только пользователь, который его запросил.",
                action=action,
            )
        return None

    async def confirm(self, key: str, *, user_id: int | None = None) -> PendingActionResult:
        action = self.pending_for(key)
        if not action:
            return PendingActionResult(
                status="nothing_to_confirm", message="Нет ожидающего действия для подтверждения."
            )
        ownership_error = self._validate_ownership(action, user_id)
        if ownership_error:
            return ownership_error

        if self.dry_run:
            return PendingActionResult(
                status="dry_run",
                message="AGENT_DRY_RUN включён: действие не выполнено. Ожидающее действие оставлено без изменений.",
                action=action,
            )

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

        can_write = await _user_can_write(user_id, self.bitrix)
        if not can_write:
            self._audit(
                "denied",
                key=key,
                action=action,
                data={"reason": "user is not allowed to write Bitrix through agent"},
            )
            return PendingActionResult(
                status="denied",
                message=("Не выполнил действие: у пользователя нет права на Bitrix-запись через агента."),
                action=action,
            )

        params = apply_write_policy(action.method, deepcopy(action.params))
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
    ) -> BitrixWritePort | PendingActionResult:
        settings = self._settings
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
            "timestamp": datetime.now(UTC).isoformat(),
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
        created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
        specialist_id=str(data.get("specialist_id") or "bitrix24"),
    )


async def _user_can_write(user_id: int | None, bitrix: BitrixUserPort) -> bool:
    if user_id is None:
        return False
    try:
        user = await bitrix.get_user(user_id)
    except Exception:
        return False
    if user is None:
        return False
    profile = compact_user_profile(user)
    if not profile.get("active", True):
        return False
    user_type = str(profile.get("user_type") or "").lower()
    return user_type in ("", "employee")


class _WritePolicy(Protocol):
    def apply(self, params: dict[str, Any]) -> dict[str, Any]: ...


class _NoDeadlinePolicy:
    def apply(self, params: dict[str, Any]) -> dict[str, Any]:
        fields = params.get("fields")
        if isinstance(fields, dict) and no_deadline_requested(fields):
            prepare_no_deadline_fields(fields)
        return params


_WRITE_POLICY_REGISTRY: dict[str, _WritePolicy] = {
    "tasks.task.add": _NoDeadlinePolicy(),
}


def apply_write_policy(method: str, params: dict[str, Any]) -> dict[str, Any]:
    policy = _WRITE_POLICY_REGISTRY.get(method.strip().lower())
    return policy.apply(params) if policy else params


def no_deadline_requested(fields: dict[str, Any]) -> bool:
    for key in NO_DEADLINE_FIELD_NAMES:
        if truthy(fields.get(key)):
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
