"""Tests for Bitrix24 agent tools: notify_users, resolve_user, resolve_project."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_server.agents.bitrix24.tools.notify_users import NotifyUsersTool
from ai_server.agents.bitrix24.tools.resolve_project import ResolveProjectTool
from ai_server.agents.bitrix24.tools.resolve_user import ResolveUserTool
from ai_server.models import ToolStatus


def anyio_run(coro):
    import anyio

    async def _runner():
        return await coro

    return anyio.run(_runner)


# ---------------------------------------------------------------------------
# NotifyUsersTool
# ---------------------------------------------------------------------------


def test_notify_users_empty_user_ids():
    tool = NotifyUsersTool()
    result = anyio_run(tool.execute({"user_ids": [], "message": "тест"}))
    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_notify_users_missing_message():
    tool = NotifyUsersTool()
    result = anyio_run(tool.execute({"user_ids": [9], "message": ""}))
    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_notify_users_not_configured():
    tool = NotifyUsersTool(client=None)
    result = anyio_run(tool.execute({"user_ids": [9], "message": "уведомление"}))
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_notify_users_ok_single():
    client = AsyncMock()
    client.notify_user = AsyncMock(return_value=1)
    tool = NotifyUsersTool(client=client)

    result = anyio_run(tool.execute({"user_ids": [9], "message": "Срок задачи истёк"}))

    assert result.status == ToolStatus.OK
    assert result.data["user_ids"] == [9]
    client.notify_user.assert_called_once_with(user_id=9, message="Срок задачи истёк", tag="ai_server")


def test_notify_users_ok_multiple():
    client = AsyncMock()
    client.notify_user = AsyncMock(return_value=1)
    tool = NotifyUsersTool(client=client)

    result = anyio_run(tool.execute({"user_ids": [9, 15, 16], "message": "тест"}))

    assert result.status == ToolStatus.OK
    assert client.notify_user.call_count == 3


def test_notify_users_partial_failure():
    from ai_server.integrations.bitrix.client import BitrixApiError

    client = AsyncMock()
    client.notify_user = AsyncMock(side_effect=[None, BitrixApiError("im.notify", "USER_NOT_FOUND")])
    tool = NotifyUsersTool(client=client)

    result = anyio_run(tool.execute({"user_ids": [9, 99], "message": "тест"}))

    assert result.status == ToolStatus.ERROR
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["user_id"] == 99


def test_notify_users_custom_tag():
    client = AsyncMock()
    client.notify_user = AsyncMock(return_value=1)
    tool = NotifyUsersTool(client=client)

    anyio_run(tool.execute({"user_ids": [9], "message": "тест", "tag": "supervisor"}))
    client.notify_user.assert_called_once_with(user_id=9, message="тест", tag="supervisor")


def test_notify_users_definition():
    tool = NotifyUsersTool()
    defn = tool.definition()
    assert defn.name == "bitrix_notify_users"
    assert "user_ids" in defn.parameters["properties"]


# ---------------------------------------------------------------------------
# ResolveUserTool
# ---------------------------------------------------------------------------


def test_resolve_user_empty_query():
    tool = ResolveUserTool()
    result = anyio_run(tool.execute({"query": ""}))
    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_resolve_user_not_configured():
    tool = ResolveUserTool(client=None)
    result = anyio_run(tool.execute({"query": "Иван"}))
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_resolve_user_not_found():
    client = AsyncMock()
    client.search_users = AsyncMock(return_value=[])
    tool = ResolveUserTool(client=client)

    result = anyio_run(tool.execute({"query": "НетТакого"}))
    assert result.status == ToolStatus.NOT_FOUND
    assert result.data["candidates"] == []


def test_resolve_user_single_match():
    client = AsyncMock()
    client.search_users = AsyncMock(
        return_value=[{"ID": 9, "NAME": "Иван", "LAST_NAME": "Петров", "EMAIL": "ivan@test.com"}]
    )
    tool = ResolveUserTool(client=client)

    result = anyio_run(tool.execute({"query": "Иван"}))

    assert result.status == ToolStatus.OK
    assert result.data["candidate"]["id"] == 9
    assert "Петров" in result.data["candidate"]["label"]
    assert result.data["candidate"]["email"] == "ivan@test.com"


def test_resolve_user_ambiguous():
    client = AsyncMock()
    client.search_users = AsyncMock(
        return_value=[
            {"ID": 9, "NAME": "Иван", "LAST_NAME": "Петров"},
            {"ID": 10, "NAME": "Иван", "LAST_NAME": "Сидоров"},
        ]
    )
    tool = ResolveUserTool(client=client)

    result = anyio_run(tool.execute({"query": "Иван"}))
    assert result.status == ToolStatus.AMBIGUOUS
    assert len(result.data["candidates"]) == 2


def test_resolve_user_skips_entry_without_id():
    client = AsyncMock()
    client.search_users = AsyncMock(
        return_value=[
            {"NAME": "Без ID", "LAST_NAME": "Пример"},  # no ID
            {"ID": 9, "NAME": "С ID", "LAST_NAME": "Нормальный"},
        ]
    )
    tool = ResolveUserTool(client=client)

    result = anyio_run(tool.execute({"query": "тест"}))
    assert result.status == ToolStatus.OK
    assert result.data["candidate"]["id"] == 9


def test_resolve_user_api_error():
    from ai_server.integrations.bitrix.client import BitrixApiError

    client = AsyncMock()
    client.search_users = AsyncMock(side_effect=BitrixApiError("user.search", "ACCESS_DENIED"))
    tool = ResolveUserTool(client=client)

    result = anyio_run(tool.execute({"query": "Иван"}))
    assert result.status == ToolStatus.ERROR


# ---------------------------------------------------------------------------
# ResolveProjectTool
# ---------------------------------------------------------------------------


def test_resolve_project_empty_query():
    tool = ResolveProjectTool()
    result = anyio_run(tool.execute({"query": ""}))
    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_resolve_project_not_configured():
    tool = ResolveProjectTool(client=None)
    result = anyio_run(tool.execute({"query": "Склад"}))
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_resolve_project_not_found():
    client = AsyncMock()
    client.search_projects = AsyncMock(return_value=[])
    tool = ResolveProjectTool(client=client)

    result = anyio_run(tool.execute({"query": "НетТакого"}))
    assert result.status == ToolStatus.NOT_FOUND


def test_resolve_project_single_match():
    client = AsyncMock()
    client.search_projects = AsyncMock(return_value=[{"ID": 17, "NAME": "Склад"}])
    tool = ResolveProjectTool(client=client)

    result = anyio_run(tool.execute({"query": "Склад"}))
    assert result.status == ToolStatus.OK
    assert result.data["candidate"]["id"] == 17
    assert result.data["candidate"]["label"] == "Склад"


def test_resolve_project_ambiguous():
    client = AsyncMock()
    client.search_projects = AsyncMock(
        return_value=[
            {"ID": 17, "NAME": "Склад 1"},
            {"ID": 18, "NAME": "Склад 2"},
        ]
    )
    tool = ResolveProjectTool(client=client)

    result = anyio_run(tool.execute({"query": "Склад"}))
    assert result.status == ToolStatus.AMBIGUOUS
    assert len(result.data["candidates"]) == 2


def test_resolve_project_fallback_label():
    client = AsyncMock()
    client.search_projects = AsyncMock(return_value=[{"ID": 5, "NAME": ""}])
    tool = ResolveProjectTool(client=client)

    result = anyio_run(tool.execute({"query": "тест"}))
    assert result.status == ToolStatus.OK
    assert "5" in result.data["candidate"]["label"]
