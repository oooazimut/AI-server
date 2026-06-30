"""Tests for fetch_staff_roster (Bitrix domain) and StaffMember."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_server.tools.vehicle_usage import StaffMember
from ai_server.workers.bitrix.staff_roster_publisher import fetch_staff_roster


def anyio_run(coro):
    import anyio

    async def _runner():
        return await coro

    return anyio.run(_runner)


def _make_bitrix(users: list[dict]) -> AsyncMock:
    client = AsyncMock()
    client.list_all_users = AsyncMock(return_value=users)
    return client


# ---------------------------------------------------------------------------
# fetch_staff_roster
# ---------------------------------------------------------------------------


def test_fetch_staff_roster_basic():
    users = [
        {"ID": "9", "NAME": "Иван", "LAST_NAME": "Петров"},
        {"ID": "15", "NAME": "Олег", "LAST_NAME": "Сидоров"},
    ]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert len(roster) == 2
    names = [m.name for m in roster]
    assert "Петров Иван" in names
    assert "Сидоров Олег" in names


def test_fetch_staff_roster_sorted_alphabetically():
    users = [
        {"ID": "9", "NAME": "Яков", "LAST_NAME": "Яковлев"},
        {"ID": "10", "NAME": "Антон", "LAST_NAME": "Антонов"},
    ]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert roster[0].name == "Антонов Антон"
    assert roster[1].name == "Яковлев Яков"


def test_fetch_staff_roster_order_assigned():
    users = [
        {"ID": "10", "NAME": "Борис", "LAST_NAME": "Борисов"},
        {"ID": "9", "NAME": "Антон", "LAST_NAME": "Антонов"},
    ]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert roster[0].order == 1
    assert roster[1].order == 2


def test_fetch_staff_roster_excludes_user_ids():
    users = [
        {"ID": "9", "NAME": "Иван", "LAST_NAME": "Петров"},
        {"ID": "1", "NAME": "Марат", "LAST_NAME": "Директор"},
    ]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users), exclude_user_ids={1}))

    assert len(roster) == 1
    assert roster[0].user_id == 9


def test_fetch_staff_roster_skips_missing_id():
    users = [
        {"NAME": "Без ID", "LAST_NAME": "Пример"},
        {"ID": "9", "NAME": "С ID", "LAST_NAME": "Нормальный"},
    ]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert len(roster) == 1
    assert roster[0].user_id == 9


def test_fetch_staff_roster_skips_empty_name():
    users = [
        {"ID": "9", "NAME": "", "LAST_NAME": ""},
        {"ID": "10", "NAME": "Иван", "LAST_NAME": "Петров"},
    ]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert len(roster) == 1
    assert roster[0].user_id == 10


def test_fetch_staff_roster_empty():
    roster = anyio_run(fetch_staff_roster(_make_bitrix([])))
    assert roster == []


def test_fetch_staff_roster_assigns_bitrix_user_id():
    users = [{"ID": "42", "NAME": "Тест", "LAST_NAME": "Тестов"}]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert roster[0].user_id == 42


def test_fetch_staff_roster_name_only_first():
    users = [{"ID": "9", "NAME": "Иван", "LAST_NAME": ""}]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert roster[0].name == "Иван"


def test_fetch_staff_roster_returns_staff_members():
    users = [{"ID": "9", "NAME": "Иван", "LAST_NAME": "Петров"}]
    roster = anyio_run(fetch_staff_roster(_make_bitrix(users)))

    assert all(isinstance(m, StaffMember) for m in roster)
