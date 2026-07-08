"""Tests for Bitrix calendar event/reminder draft tools."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio

from ai_server.agents.bitrix24.tools.calendar import (
    CALENDAR_EVENT_DRAFT_TYPE,
    CalendarEventConfirmTool,
    CalendarEventDraftTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


def _exec(tool, args, *, user_id=None, dialog_key=None, dialog_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)

    return anyio.run(_run)


def test_calendar_draft_uses_date_only_default_noon():
    store = FakeTaskDraftStore()
    tool = CalendarEventDraftTool(store=store)

    result = _exec(
        tool,
        {"title": "позвонить Борисову", "date_iso": "2026-07-09", "description": "Позвонить Борисову"},
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["_draft_type"] == CALENDAR_EVENT_DRAFT_TYPE
    assert draft["method"] == "calendar.event.add"
    assert draft["params"]["ownerId"] == 13
    assert draft["params"]["from"] == "2026-07-09T12:00:00+03:00"
    assert draft["params"]["to"] == "2026-07-09T12:30:00+03:00"
    assert draft["params"].get("attendees") is None
    assert result.data["preview"]["participants"] == "только текущий пользователь"


def test_calendar_draft_denies_missing_dialog_id():
    store = FakeTaskDraftStore()
    tool = CalendarEventDraftTool(store=store)

    result = _exec(tool, {"title": "позвонить", "date_iso": "2026-07-09"}, user_id=13, dialog_key="d:13")

    assert result.status == ToolStatus.DENIED
    assert not store._drafts


def test_calendar_confirm_uses_oauth_calendar_event_add():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": CALENDAR_EVENT_DRAFT_TYPE,
                "method": "calendar.event.add",
                "title": "позвонить Борисову",
                "start_iso": "2026-07-09T12:00:00+03:00",
                "params": {
                    "type": "user",
                    "ownerId": 13,
                    "name": "позвонить Борисову",
                    "from": "2026-07-09T12:00:00+03:00",
                    "to": "2026-07-09T12:30:00+03:00",
                    "timezone_from": "Europe/Moscow",
                    "timezone_to": "Europe/Moscow",
                },
            },
        )
    )
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"result": {"id": 444}})
    oauth = FakeBitrixOAuth(oauth_client)

    tool = CalendarEventConfirmTool(store=store, bitrix_oauth=oauth, oauth_required_for_writes=True)
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert result.data["event_id"] == "444"
    assert oauth.user_ids == [13]
    oauth_client.call.assert_awaited_once_with(
        "calendar.event.add",
        {
            "type": "user",
            "ownerId": 13,
            "name": "позвонить Борисову",
            "from": "2026-07-09T12:00:00+03:00",
            "to": "2026-07-09T12:30:00+03:00",
            "timezone_from": "Europe/Moscow",
            "timezone_to": "Europe/Moscow",
        },
    )
    assert "d:13" not in store._drafts


def test_calendar_confirm_ignores_task_create_draft():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:13", {"fields": {"TITLE": "задача", "RESPONSIBLE_ID": 13}}))
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"result": {"id": 444}})

    tool = CalendarEventConfirmTool(store=store, bitrix_oauth=FakeBitrixOAuth(oauth_client))
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.NOT_FOUND
    oauth_client.call.assert_not_called()
    assert "d:13" in store._drafts


class FakeBitrixOAuth:
    def __init__(self, client) -> None:
        self.client = client
        self.user_ids = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client
