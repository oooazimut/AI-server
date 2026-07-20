from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.agents.bitrix24.tools.draft_lifecycle import (
    attach_draft_metadata,
    bitrix_mutation_outcome,
    claim_exact_draft,
    discard_exact_draft,
    release_exact_draft,
    renew_exact_draft_claim,
    resolve_unknown_draft_claim,
)
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.tools.bitrix_ports import BitrixWritePort
from ai_server.utils import MOSCOW_TZ, compact_text, optional_int

CALENDAR_EVENT_DRAFT_TYPE = "calendar_event"
DEFAULT_CALENDAR_EVENT_HOUR = 12
DEFAULT_CALENDAR_EVENT_DURATION_MINUTES = 30
DEFAULT_CALENDAR_TIMEZONE = "Europe/Moscow"


@dataclass(frozen=True)
class BitrixCalendarEventDraft:
    payload: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, Any] = field(default_factory=dict)
    contract_errors: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        params = self.payload.get("params")
        return not self.contract_errors and isinstance(params, dict) and bool(params.get("name"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "preview": self.preview,
            "contract_errors": self.contract_errors,
        }


def build_calendar_event_draft_from_args(
    args: dict[str, Any],
    *,
    user_id: int | None = None,
) -> BitrixCalendarEventDraft:
    args = args or {}
    title = compact_text(str(args.get("title") or args.get("name") or "")).strip(" .,:;-")
    description = compact_text(str(args.get("description") or args.get("details") or ""))
    owner_id = user_id
    start_at, start_error = _start_from_args(args)
    end_at, end_error = _end_from_args(args, start_at)
    attendee_ids = _int_list(args.get("attendee_ids") or args.get("attendees"))
    owner_name = compact_text(str(args.get("owner_name") or args.get("owner_label") or ""))
    reminder_minutes = optional_int(args.get("reminder_minutes"))

    contract_errors: list[str] = []
    if not title:
        contract_errors.append("calendar_event_draft.title is required")
    if owner_id is None:
        contract_errors.append("calendar_event_draft requires current Bitrix user_id")
    if start_error:
        contract_errors.append(start_error)
    if end_error:
        contract_errors.append(end_error)
    if start_at is None:
        contract_errors.append("calendar_event_draft.start_iso or date_iso is required")

    params: dict[str, Any] = {}
    if title and owner_id is not None and start_at is not None and end_at is not None:
        params = {
            "type": "user",
            "ownerId": owner_id,
            "name": title,
            "from": start_at.isoformat(),
            "to": end_at.isoformat(),
            "timezone_from": DEFAULT_CALENDAR_TIMEZONE,
            "timezone_to": DEFAULT_CALENDAR_TIMEZONE,
        }
        if description:
            params["description"] = description
        if attendee_ids:
            params["attendees"] = attendee_ids
        if reminder_minutes is not None:
            params["remind"] = [{"type": "min", "count": max(0, reminder_minutes)}]

    payload = {
        "_draft_type": CALENDAR_EVENT_DRAFT_TYPE,
        "method": "calendar.event.add",
        "params": params,
        "title": title,
        "description": description,
        "owner_id": owner_id,
        "start_iso": start_at.isoformat() if start_at else "",
        "end_iso": end_at.isoformat() if end_at else "",
        "timezone": DEFAULT_CALENDAR_TIMEZONE,
        "attendee_ids": attendee_ids,
        "reminder_minutes": reminder_minutes,
    }
    preview = {
        "title": title,
        "description": description,
        "start": _format_datetime_for_preview(start_at),
        "end": _format_datetime_for_preview(end_at),
        "participants": owner_name
        if not attendee_ids and owner_name
        else "только текущий пользователь"
        if not attendee_ids
        else f"{len(attendee_ids)} участник(а)",
        "reminder": _format_reminder_for_preview(reminder_minutes),
    }
    return BitrixCalendarEventDraft(
        payload=payload,
        preview=preview,
        contract_errors=_unique(contract_errors),
    )


class CalendarEventDraftTool:
    name = "calendar_event_draft"

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Prepare a Bitrix calendar event/reminder draft. Use for requests like 'remind me tomorrow'. "
                "The event is saved as a draft and must be confirmed by the user before calendar.event.add."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "start_iso": {"type": "string"},
                    "date_iso": {"type": "string"},
                    "end_iso": {"type": "string"},
                    "attendee_ids": {"type": "array", "items": {"type": "integer"}},
                    "reminder_minutes": {"type": "integer"},
                },
                "required": ["title"],
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
        if context_error := _write_context_error(user_id=user_id, dialog_id=dialog_id):
            return ToolResult(status=ToolStatus.DENIED, tool=self.name, error=context_error)
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="calendar_event_draft requires dialog_key",
            )
        draft = build_calendar_event_draft_from_args(args, user_id=user_id)
        if not draft.is_ready:
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error=f"calendar_event_draft called outside contract: {'; '.join(draft.contract_errors)}",
                data=draft.as_action_details(),
            )
        payload = attach_draft_metadata(draft.payload, source_args=args, user_id=user_id)
        await self._store.save_task_draft(dialog_key, payload)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"draft": payload, "preview": draft.preview})


class CalendarEventConfirmTool:
    name = "calendar_event_confirm"

    def __init__(
        self,
        store: TaskDraftStorePort,
        write_client: BitrixWritePort | None = None,
        *,
        bitrix_oauth: BitrixOAuthService | None = None,
        dry_run: bool = False,
        oauth_required_for_writes: bool = True,
        draft_ttl_minutes: int | None = None,
    ) -> None:
        self._store = store
        self._write_client = write_client
        self._bitrix_oauth = bitrix_oauth
        self._dry_run = dry_run
        self._oauth_required_for_writes = oauth_required_for_writes
        self._draft_ttl_minutes = draft_ttl_minutes

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Confirm and create the pending Bitrix calendar event draft as the current OAuth user.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="calendar_event_confirm requires dialog_key",
            )
        draft = await self._store.get_task_draft(dialog_key, ttl_minutes=self._draft_ttl_minutes)
        if not draft:
            unresolved = await resolve_unknown_draft_claim(
                self._store, dialog_key=dialog_key, expected_type=CALENDAR_EVENT_DRAFT_TYPE
            )
            if unresolved is not None:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="calendar creation outcome requires durable audit before retry",
                    data={
                        "mutation_outcome": "unknown",
                        "resolution_status": unresolved.get("_draft_resolution_status"),
                        "draft_id": unresolved.get("_draft_id"),
                    },
                )
        if not draft or draft.get("_draft_type") != CALENDAR_EVENT_DRAFT_TYPE:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error="no pending calendar event draft found for this dialog",
            )
        params = draft.get("params") if isinstance(draft.get("params"), dict) else {}
        if not params:
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error="calendar event draft has no params",
                data={"draft": draft},
            )
        sanitized = apply_write_policy("calendar.event.add", params)
        if self._dry_run:
            return ToolResult(status=ToolStatus.DRY_RUN, tool=self.name, data={"params": sanitized, "draft": draft})

        if self._oauth_required_for_writes:
            if context_error := _write_context_error(user_id=user_id, dialog_id=dialog_id):
                return ToolResult(status=ToolStatus.DENIED, tool=self.name, error=context_error, data={"draft": draft})
            if self._bitrix_oauth is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error="Bitrix OAuth is required for calendar event creation.",
                    data={"draft": draft},
                )
            claimed = None
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                claimed = await claim_exact_draft(
                    self._store, dialog_key=dialog_key, draft=draft, expected_type=CALENDAR_EVENT_DRAFT_TYPE
                )
                if claimed is None:
                    return ToolResult(
                        status=ToolStatus.DENIED,
                        tool=self.name,
                        error="calendar draft changed, expired, or is already being confirmed",
                    )
                if not await renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed):
                    return ToolResult(status=ToolStatus.DENIED, tool=self.name, error="calendar draft claim was lost")
                result = await _execute_calendar_event_create(oauth_client.call, sanitized, draft)
            except BitrixOAuthTokenMissing as exc:
                if claimed is not None:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=str(exc), data={"draft": draft}
                )
            except BitrixConfigError as exc:
                if claimed is not None:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error=str(exc),
                    data={"draft": draft, "mutation_outcome": "not_started"},
                )
            except BitrixApiError as exc:
                outcome = bitrix_mutation_outcome(exc)
                if claimed is not None and outcome == "rejected":
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=str(exc),
                    data={"draft": draft, "mutation_outcome": outcome},
                )
            except Exception as exc:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=f"{type(exc).__name__}: {exc}",
                    data={"draft": draft, "mutation_outcome": "unknown"},
                )
            await self._store.delete_task_draft(
                dialog_key,
                status="confirmed",
                expected_draft_id=str(claimed.get("_draft_id") or ""),
                expected_version=optional_int(claimed.get("_draft_version")),
                expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
            )
            return ToolResult(status=ToolStatus.OK, tool=self.name, data={**result, "draft": draft})

        if self._write_client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="write_client is not configured",
                data={"draft": draft},
            )
        claimed = await claim_exact_draft(
            self._store, dialog_key=dialog_key, draft=draft, expected_type=CALENDAR_EVENT_DRAFT_TYPE
        )
        if claimed is None:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool=self.name,
                error="calendar draft changed, expired, or is already being confirmed",
            )
        try:
            if not await renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed):
                return ToolResult(status=ToolStatus.DENIED, tool=self.name, error="calendar draft claim was lost")
            result = await _execute_calendar_event_create(self._write_client.call, sanitized, draft)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"{type(exc).__name__}: {exc}",
                data={"draft": draft, "mutation_outcome": "unknown"},
            )
        await self._store.delete_task_draft(
            dialog_key,
            status="confirmed",
            expected_draft_id=str(claimed.get("_draft_id") or ""),
            expected_version=optional_int(claimed.get("_draft_version")),
            expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
        )
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={**result, "draft": draft})


class CalendarEventDiscardTool:
    name = "calendar_event_discard"

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Discard the pending Bitrix calendar event draft.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if dialog_key:
            draft = await self._store.get_task_draft(dialog_key)
            if isinstance(draft, dict) and draft.get("_draft_type") == CALENDAR_EVENT_DRAFT_TYPE:
                await discard_exact_draft(
                    self._store,
                    dialog_key=dialog_key,
                    draft=draft,
                    expected_type=CALENDAR_EVENT_DRAFT_TYPE,
                )
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"discarded": bool(dialog_key)})


async def _execute_calendar_event_create(
    call: Callable[[str, dict[str, Any]], Awaitable[Any]],
    params: dict[str, Any],
    draft: dict[str, Any],
) -> dict[str, Any]:
    raw = await call("calendar.event.add", params)
    return {
        "result": raw,
        "event_id": _created_event_id(raw),
        "title": str(draft.get("title") or params.get("name") or ""),
        "start_label": _format_datetime_for_preview(_parse_datetime_or_date(str(draft.get("start_iso") or ""))[0]),
        "params": params,
    }


def _start_from_args(args: dict[str, Any]) -> tuple[datetime | None, str]:
    raw = str(args.get("start_iso") or args.get("from") or args.get("date_iso") or args.get("date") or "").strip()
    if not raw:
        return None, ""
    parsed, error = _parse_datetime_or_date(raw)
    return parsed, error


def _end_from_args(args: dict[str, Any], start_at: datetime | None) -> tuple[datetime | None, str]:
    raw = str(args.get("end_iso") or args.get("to") or "").strip()
    if raw:
        parsed, error = _parse_datetime_or_date(raw)
        if error or parsed is None:
            return None, error or "calendar_event_draft.end_iso must be ISO 8601"
        if start_at is not None and parsed <= start_at:
            return None, "calendar_event_draft.end_iso must be after start_iso"
        return parsed, ""
    if start_at is None:
        return None, ""
    return start_at + timedelta(minutes=DEFAULT_CALENDAR_EVENT_DURATION_MINUTES), ""


def _parse_datetime_or_date(value: str) -> tuple[datetime | None, str]:
    raw = value.strip()
    if not raw:
        return None, ""
    try:
        if len(raw) == 10:
            parsed_date = date.fromisoformat(raw)
            return datetime.combine(parsed_date, time(DEFAULT_CALENDAR_EVENT_HOUR, 0), tzinfo=MOSCOW_TZ), ""
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None, "calendar_event_draft.start_iso/end_iso must be ISO 8601"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MOSCOW_TZ)
    return parsed.astimezone(MOSCOW_TZ), ""


def _format_datetime_for_preview(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(MOSCOW_TZ).strftime("%d.%m.%Y %H:%M МСК")


def _format_reminder_for_preview(value: int | None) -> str:
    if value is None:
        return "по настройкам календаря Bitrix"
    if value <= 0:
        return "в момент события"
    return f"за {value} мин."


def _int_list(value: object) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[int] = []
    for item in values:
        parsed = optional_int(item)
        if parsed is not None:
            result.append(parsed)
    return _unique(result)


def _created_event_id(value: object) -> str:
    if not isinstance(value, dict):
        return str(value) if value is not None else ""
    candidates: list[object] = [value.get("id"), value.get("ID"), value.get("eventId"), value.get("EVENT_ID")]
    nested = value.get("result")
    if isinstance(nested, dict):
        candidates.extend([nested.get("id"), nested.get("ID"), nested.get("eventId"), nested.get("EVENT_ID")])
    else:
        candidates.append(nested)
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _unique(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _write_context_error(*, user_id: int | None, dialog_id: str | None) -> str:
    if user_id is None:
        return "Bitrix calendar event denied: current Bitrix user_id is missing."
    if not str(dialog_id or "").strip():
        return "Bitrix calendar event denied: current Bitrix dialog_id is missing."
    return ""


__all__ = [
    "BitrixCalendarEventDraft",
    "CALENDAR_EVENT_DRAFT_TYPE",
    "CalendarEventConfirmTool",
    "CalendarEventDiscardTool",
    "CalendarEventDraftTool",
    "build_calendar_event_draft_from_args",
]
