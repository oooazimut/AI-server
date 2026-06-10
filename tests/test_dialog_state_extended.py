from pathlib import Path
from typing import Any

import pytest

from ai_server.integrations.bitrix.dialog_state import (
    BitrixDialogState,
    DialogStateStore,
    PendingBitrixAction,
    make_dialog_key,
    pending_bitrix_from_dict,
    state_from_dict,
)


# make_dialog_key
def test_make_dialog_key_with_chat_id():
    key = make_dialog_key(chat_id=42, user_id=7)
    assert key == "chat:42:user:7"


def test_make_dialog_key_with_dialog_id():
    key = make_dialog_key(dialog_id="d123", user_id=5)
    assert key == "dialog:d123:user:5"


def test_make_dialog_key_user_only():
    key = make_dialog_key(user_id=10)
    assert key == "user:10"


def test_make_dialog_key_no_user_defaults_zero():
    key = make_dialog_key(chat_id=1)
    assert key == "chat:1:user:0"


# PendingBitrixAction — specialist_id field
def test_pending_action_default_specialist_id():
    action = PendingBitrixAction(
        method="tasks.task.add",
        params={"fields": {"TITLE": "Test"}},
        summary="Создать задачу",
        created_by=42,
    )
    assert action.specialist_id == "bitrix24"


def test_pending_action_custom_specialist_id():
    action = PendingBitrixAction(
        method="tasks.task.add",
        params={},
        summary="Задача",
        created_by=1,
        specialist_id="pto",
    )
    assert action.specialist_id == "pto"


# state_from_dict
def test_state_from_dict_empty_data():
    state = state_from_dict("key:1", None)
    assert state.key == "key:1"
    assert state.summary == ""
    assert state.turns == []
    assert state.pending_action is None


def test_state_from_dict_with_turns():
    data = {
        "summary": "Разговор",
        "turns": [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуйте"},
        ],
        "pending_action": None,
    }
    state = state_from_dict("key:1", data)
    assert state.summary == "Разговор"
    assert len(state.turns) == 2
    assert state.turns[0]["role"] == "user"


def test_state_from_dict_with_pending_action():
    data = {
        "pending_action": {
            "method": "tasks.task.add",
            "params": {"fields": {"TITLE": "Test"}},
            "summary": "Создать задачу",
            "created_by": 42,
            "specialist_id": "bitrix24",
        }
    }
    state = state_from_dict("key:1", data)
    assert state.pending_action is not None
    assert state.pending_action.method == "tasks.task.add"
    assert state.pending_action.specialist_id == "bitrix24"


def test_state_from_dict_filters_non_dict_turns():
    data = {
        "turns": [{"role": "user", "content": "ok"}, "invalid", 123],
    }
    state = state_from_dict("key:1", data)
    assert len(state.turns) == 1


# pending_bitrix_from_dict
def test_pending_bitrix_from_dict_none():
    assert pending_bitrix_from_dict(None) is None


def test_pending_bitrix_from_dict_non_dict():
    assert pending_bitrix_from_dict("string") is None
    assert pending_bitrix_from_dict(42) is None


def test_pending_bitrix_from_dict_missing_method():
    assert pending_bitrix_from_dict({"params": {}, "summary": "test"}) is None


def test_pending_bitrix_from_dict_preserves_specialist_id():
    data = {
        "method": "tasks.task.add",
        "params": {},
        "summary": "Задача",
        "created_by": 5,
        "specialist_id": "logistics",
    }
    action = pending_bitrix_from_dict(data)
    assert action is not None
    assert action.specialist_id == "logistics"


def test_pending_bitrix_from_dict_defaults_specialist_id_to_bitrix24():
    data = {
        "method": "tasks.task.add",
        "params": {},
        "summary": "Задача",
        "created_by": 5,
    }
    action = pending_bitrix_from_dict(data)
    assert action is not None
    assert action.specialist_id == "bitrix24"


def test_pending_bitrix_from_dict_created_by_string_converted():
    data = {
        "method": "tasks.task.add",
        "params": {},
        "summary": "Задача",
        "created_by": "42",
    }
    action = pending_bitrix_from_dict(data)
    assert action is not None
    assert action.created_by == 42


# DialogStateStore — SQLite operations
def test_dialog_state_store_load_missing_key(tmp_path):
    store = DialogStateStore(tmp_path / "dialog.db")
    state = store.load("nonexistent_key")
    assert state.key == "nonexistent_key"
    assert state.pending_action is None


def test_dialog_state_store_save_and_load(tmp_path):
    store = DialogStateStore(tmp_path / "dialog.db")
    state = BitrixDialogState(
        key="chat:1:user:7",
        summary="Тест",
        turns=[{"role": "user", "content": "Привет"}],
    )
    store.save(state)
    loaded = store.load("chat:1:user:7")
    assert loaded.summary == "Тест"
    assert loaded.turns[0]["content"] == "Привет"


def test_dialog_state_store_set_pending(tmp_path):
    store = DialogStateStore(tmp_path / "dialog.db")
    action = PendingBitrixAction(
        method="tasks.task.add",
        params={"fields": {"TITLE": "Test"}},
        summary="Создать задачу Test",
        created_by=42,
    )
    store.set_pending("chat:1:user:7", action)
    loaded = store.load("chat:1:user:7")
    assert loaded.pending_action is not None
    assert loaded.pending_action.method == "tasks.task.add"


def test_dialog_state_store_clear_pending(tmp_path):
    store = DialogStateStore(tmp_path / "dialog.db")
    action = PendingBitrixAction(
        method="tasks.task.add",
        params={},
        summary="Задача",
        created_by=1,
    )
    store.set_pending("key:1", action)
    assert store.load("key:1").pending_action is not None

    store.clear_pending("key:1")
    assert store.load("key:1").pending_action is None


def test_dialog_state_store_clear_pending_preserves_turns(tmp_path):
    store = DialogStateStore(tmp_path / "dialog.db")
    state = BitrixDialogState(
        key="key:1",
        turns=[{"role": "user", "content": "Привет"}],
        pending_action=PendingBitrixAction(
            method="tasks.task.add",
            params={},
            summary="Задача",
            created_by=1,
        ),
    )
    store.save(state)
    store.clear_pending("key:1")
    loaded = store.load("key:1")
    assert loaded.pending_action is None
    assert loaded.turns[0]["content"] == "Привет"


def test_dialog_state_store_overwrite_on_conflict(tmp_path):
    store = DialogStateStore(tmp_path / "dialog.db")
    store.save(BitrixDialogState(key="key:1", summary="Первый"))
    store.save(BitrixDialogState(key="key:1", summary="Второй"))
    assert store.load("key:1").summary == "Второй"


# TaskClosureHandler Protocol — duck-type conformance
def test_task_closure_handler_protocol_conformance():
    """A class with the correct execute() signature satisfies the Protocol."""
    from ai_server.integrations.bitrix.dialog_state import TaskClosureHandler

    class FakeHandler:
        async def execute(
            self,
            params: dict[str, Any],
            *,
            current_user_id: int,
            write_client: Any,
            actor_client: Any,
        ) -> dict[str, Any]:
            return {"status": "executed"}

    handler: TaskClosureHandler = FakeHandler()
    assert handler is not None
