import asyncio
import json
from datetime import date

import pytest

import ai_server.agents.bitrix24.llm as bitrix_llm
from ai_server.agents.bitrix24 import Bitrix24Specialist, BitrixLLMDecision, BitrixLLMService, BitrixLLMToolCall
from ai_server.agents.bitrix24.llm import ALLOWED_TOOL_NAMES
from ai_server.agents.bitrix24.tools.task_close import TaskCloseDraftTool
from ai_server.agents.bitrix24.tools.task_close_control import TaskCloseControlUpdateTool
from ai_server.agents.bitrix24.tools.task_create import TaskCreateDraftTool
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentTask, ToolDefinition, ToolResult
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.skills import SkillStore
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, FakeTaskDraftStore, RecordingLLMClient


def _writable_profile(user_id: int = 13) -> dict:
    return {
        "status": "ok",
        "data": {
            "user_id": user_id,
            "profile": {
                "id": user_id,
                "label": "Иванов Иван",
                "active": True,
                "is_admin": False,
                "user_type": "employee",
                "work_position": "Монтажник",
            },
        },
    }


def _bitrix_specialist(*, tools=None, llm=None, settings=None, conversation_trace=None) -> Bitrix24Specialist:
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    return Bitrix24Specialist(
        manifest,
        retriever=retriever,
        skill_store=SkillStore(),
        agent_tools=tools.agent_tools() if tools is not None else None,
        bitrix_user_client=tools.make_user_client() if tools is not None else None,
        llm=llm or FakeBitrixLLM(),
        settings=settings,
        conversation_trace=conversation_trace,
    )


class _RecordingConversationTrace:
    def __init__(self) -> None:
        self.events = []

    async def record_timing(self, **kwargs):
        self.events.append(kwargs)


def _bitrix_read_tool_definitions() -> list[dict]:
    return [
        ToolDefinition(name=name, description="", parameters={}).model_dump()
        for name in (
            "bitrix_my_tasks",
            "bitrix_task_search",
            "bitrix_project_search",
            "bitrix_api",
            "portal_search",
            "none",
        )
    ]


def test_bitrix_llm_selects_flash_only_for_read_only_requests(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("AI_SERVER_LLM_ROUTING_ENABLED", "true")
    monkeypatch.setenv("AI_SERVER_LLM_FLASH_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("AI_SERVER_LLM_PRO_MODEL", "deepseek-v4-pro")

    service = BitrixLLMService(settings=get_settings())

    read_client = service._decision_client_for_request("Битрикс покажи мои задачи")
    write_client = service._decision_client_for_request("Битрикс создай задачу на меня")

    assert read_client._model == "deepseek-v4-flash"
    assert write_client._model == "deepseek-v4-pro"


def test_bitrix_specialist_loads_available_skills_and_rag_context():
    result = asyncio.run(
        _bitrix_specialist(tools=FakeResolverTools()).handle(
            AgentTask(task_id="t1", request="Найди просроченные задачи в Битриксе")
        )
    )

    assert result.status == "completed"
    assert result.handoff_to == []
    details = result.actions_taken[0].details
    assert "skills" not in details
    assert "knowledge_topics" not in details
    assert {skill["id"] for skill in details["available_skills"]} >= {"tasks_search", "safe_bitrix_write"}
    assert details["retrieval_topics"]
    assert details["retrieval_hits"]
    assert result.actions_taken[1].name == "bitrix_llm_decision"


def test_bitrix_specialist_forwards_dialog_history_to_decide():
    llm = FakeBitrixLLM()
    history = [
        {"role": "user", "content": "найди задачи Иванова"},
        {"role": "assistant", "content": "Уточните, о ком из Ивановых речь?"},
    ]

    asyncio.run(
        _bitrix_specialist(tools=FakeResolverTools(), llm=llm).handle(
            AgentTask(task_id="t1", request="который Павел", context={"dialog_history": history})
        )
    )

    assert llm.decide_calls[0]["dialog_history"] == history


def test_bitrix_specialist_defaults_dialog_history_to_none_when_absent():
    llm = FakeBitrixLLM()

    asyncio.run(
        _bitrix_specialist(tools=FakeResolverTools(), llm=llm).handle(AgentTask(task_id="t1", request="привет"))
    )

    assert llm.decide_calls[0]["dialog_history"] is None


def test_bitrix_specialist_searches_task_by_id():
    result = asyncio.run(
        _bitrix_specialist(
            tools=FakeResolverTools(
                bitrix_api_result=ToolResult(
                    status="ok",
                    tool="bitrix_api",
                    data={
                        "method": "tasks.task.get",
                        "params": {"taskId": 8413},
                        "result": {
                            "task": {
                                "id": "8413",
                                "title": "AI-server private smoke test",
                                "status": "2",
                                "responsibleId": "9",
                                "deadline": None,
                            }
                        },
                    },
                )
            ),
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="bitrix_api",
                        args={"action": "call", "method": "tasks.task.get", "params": {"taskId": 8413}},
                    )
                ],
                final_answer="Нашёл задачу #8413 без срока.",
            ),
        ).handle(AgentTask(task_id="t1", request="Найди задачу #8413"))
    )

    assert result.status == "completed"
    assert result.actions_taken[2].name == "bitrix_api"
    assert result.actions_taken[2].status == "ok"
    assert "#8413" in result.answer
    assert "без срока" in result.answer
    assert "bitrix_tool_loop_guardrail" not in {action.name for action in result.actions_taken}


def test_bitrix_specialist_searches_my_open_tasks():
    tools = FakeResolverTools(
        my_tasks_result=ToolResult(
            status="ok",
            tool="bitrix_my_tasks",
            data={
                "status": "open",
                "items": [
                    {
                        "id": "101",
                        "title": "Проверить камеру",
                        "status": "3",
                        "status_label": "выполняется",
                        "deadline_label": "10.07.2026 19:00",
                        "roles": ["исполнитель"],
                    }
                ],
                "total": 1,
                "limit": 10,
                "offset": 0,
            },
        )
    )

    result = asyncio.run(
        _bitrix_specialist(
            tools=tools,
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="bitrix_my_tasks",
                        args={"status": "open", "limit": 10},
                    )
                ],
                final_answer="Нашёл задачу: Проверить камеру.",
            ),
        ).handle(AgentTask(task_id="t1", request="Покажи мои открытые задачи", user={"id": "9"}))
    )

    assert result.status == "completed"
    assert tools.my_tasks_calls[0] == {"status": "open", "limit": 10}
    assert "Проверить камеру" in result.answer


def test_bitrix_specialist_fast_returns_portal_search_result_and_traces_tool():
    trace = _RecordingConversationTrace()
    tools = FakeResolverTools(
        portal_search_result=ToolResult(
            status="ok",
            tool="portal_search",
            data={
                "query": "dogovor azimut",
                "items": [{"title": "DOGOVOR Azimut", "path": "Dogovora/Azimut.docx"}],
            },
        )
    )
    llm = FakeBitrixLLM(
        tool_call_steps=[
            [BitrixLLMToolCall(name="portal_search", args={"query": "dogovor azimut", "limit": 10})],
            [BitrixLLMToolCall(name="portal_search", args={"query": "dogovor azimut", "limit": 10})],
        ],
        final_answer="Nashel dogovor Azimut.",
    )

    result = asyncio.run(
        _bitrix_specialist(tools=tools, llm=llm, conversation_trace=trace).handle(
            AgentTask(task_id="t1", request="Bitrix find dogovor Azimut", user={"id": "1"})
        )
    )

    assert result.status == "completed"
    assert result.answer == "Nashel dogovor Azimut."
    assert len(llm.decide_calls) == 1
    assert tools.portal_search_calls == [{"query": "dogovor azimut", "limit": 10}]
    fast_return = [action for action in result.actions_taken if action.name == "bitrix_fast_return"]
    assert fast_return
    assert fast_return[0].details["terminal_tool"] == "portal_search"
    tool_events = [event for event in trace.events if event["stage"] == "tool_execute"]
    assert len(tool_events) == 1
    assert tool_events[0]["tool"] == "portal_search"
    assert tool_events[0]["details"]["tool"] == "portal_search"
    assert tool_events[0]["details"]["tool_result_tool"] == "portal_search"
    assert tool_events[0]["details"]["tool_result_status"] == "ok"


def test_bitrix_specialist_treats_show_warehouse_as_stock_request():
    tools = FakeResolverTools(
        warehouse_result=ToolResult(
            status="ok",
            tool="bitrix_warehouse_search",
            data={
                "matches": [{"id": "12", "title": "Борисов А.А.", "address": "Российская,8"}],
                "products": {"items": [], "total": 0, "limit": 10, "offset": 0},
            },
        )
    )

    result = asyncio.run(
        _bitrix_specialist(
            tools=tools,
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="bitrix_warehouse_search",
                        args={"query": "Борисов"},
                    )
                ],
                final_answer="Остатки проверены.",
            ),
        ).handle(AgentTask(task_id="t1", request="Битрикс покажи склад Борисов", user={"id": "1"}))
    )

    assert result.status == "completed"
    assert tools.warehouse_calls == [{"query": "Борисов", "include_products": True, "product_limit": 10}]


def test_bitrix_specialist_fast_returns_read_only_tools():
    cases = [
        (
            "bitrix_my_tasks",
            {"status": "open", "limit": 10},
            "my_tasks_result",
            "my_tasks_calls",
            ToolResult(status="ok", tool="bitrix_my_tasks", data={"items": [], "total": 0}),
        ),
        (
            "bitrix_task_search",
            {"scope": "responsible", "status": "active", "limit": 10},
            "task_search_result",
            "task_search_calls",
            ToolResult(status="ok", tool="bitrix_task_search", data={"items": [], "total": 0}),
        ),
        (
            "bitrix_project_search",
            {"query": "Garage", "limit": 10},
            "project_search_result",
            "project_search_calls",
            ToolResult(status="ok", tool="bitrix_project_search", data={"query": "Garage", "items": []}),
        ),
    ]

    for tool_name, args, result_kw, calls_attr, tool_result in cases:
        llm = FakeBitrixLLM(
            tool_calls=[BitrixLLMToolCall(name=tool_name, args=args)],
            final_answer="done",
        )
        tools = FakeResolverTools(**{result_kw: tool_result})

        result = asyncio.run(
            _bitrix_specialist(tools=tools, llm=llm).handle(
                AgentTask(task_id="t1", request=f"Bitrix read via {tool_name}", user={"id": "9"})
            )
        )

        assert result.status == "completed"
        assert getattr(tools, calls_attr)[0] == args
        assert len(llm.decide_calls) == 1
        assert result.metadata["fast_return"] is True
        assert result.metadata["terminal_tool"] == tool_name
        assert any(action.name == "bitrix_fast_return" for action in result.actions_taken)


def test_bitrix_specialist_fast_returns_read_oauth_authorization():
    llm = FakeBitrixLLM(
        tool_calls=[BitrixLLMToolCall(name="bitrix_my_tasks", args={"status": "open", "limit": 10})],
        final_answer="auth required",
    )
    tools = FakeResolverTools(
        my_tasks_result=ToolResult(
            status="denied",
            tool="bitrix_my_tasks",
            error="oauth required",
            data={"oauth_required": True, "authorization": {"message": "Authorize Bitrix24."}},
        )
    )

    result = asyncio.run(
        _bitrix_specialist(tools=tools, llm=llm).handle(
            AgentTask(task_id="t1", request="Bitrix show my tasks", user={"id": "27"})
        )
    )

    assert result.status == "completed"
    assert result.answer == "auth required"
    assert len(llm.decide_calls) == 1
    assert result.metadata["fast_return_reason"] == "read_only_oauth_authorization_required"


def test_bitrix_specialist_does_not_fast_return_ambiguous_project_search():
    llm = FakeBitrixLLM(
        tool_calls=[BitrixLLMToolCall(name="bitrix_project_search", args={"query": "Garage", "limit": 10})],
        final_answer="Choose a project.",
    )
    tools = FakeResolverTools(
        project_search_result=ToolResult(
            status="ok",
            tool="bitrix_project_search",
            data={
                "query": "Garage",
                "items": [{"id": "5", "name": "Garage A"}, {"id": "6", "name": "Garage B"}],
            },
        )
    )

    result = asyncio.run(
        _bitrix_specialist(tools=tools, llm=llm).handle(
            AgentTask(task_id="t1", request="Bitrix show project Garage", user={"id": "9"})
        )
    )

    assert result.answer == "Choose a project."
    assert len(llm.decide_calls) == 2
    assert result.metadata == {}


def test_bitrix_llm_compose_formats_read_oauth_authorization_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(json.dumps({"answer": "should not be used", "status": "completed"}))
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Bitrix show my tasks", user={"id": "27"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="denied",
                    tool="bitrix_my_tasks",
                    error="oauth required",
                    data={"authorization": {"message": "Authorize Bitrix24."}, "oauth_required": True},
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert result.answer == "Authorize Bitrix24."
    assert client.calls == []


def test_bitrix_llm_decide_routes_created_by_task_read_to_deterministic_tool(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.2,
                "tool_calls": [{"name": "bitrix_task_search", "args": {"scope": "created_by", "status": "all"}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Битрикс покажи задачи, поставленные мной.", user={"id": "13"}),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_task_search"]
    assert result.decision.tool_calls[0].args == {"scope": "created_by", "status": "active", "limit": 10}


def test_bitrix_llm_decide_routes_contract_documents_to_portal_search(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс, покажи последние 10 договоров по Азимуту.",
                user={"id": "1"},
            ),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["portal_search"]
    assert result.decision.tool_calls[0].args == {
        "query": "договор Азимут",
        "scope": "documents",
        "limit": 10,
    }


def test_bitrix_llm_decide_routes_disk_scan_to_file_portal_search(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Битрикс, найди на диске скан счета Азимут.", user={"id": "1"}),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["portal_search"]
    assert result.decision.tool_calls[0].args == {
        "query": "счет Азимут",
        "scope": "files",
        "limit": 10,
    }


def test_bitrix_llm_decide_routes_task_text_search_even_when_model_answers_from_history(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "Задача уже найдена. ID 139.",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Битрикс найди задачу Обучение сотрудников.", user={"id": "13"}),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_task_search"]
    assert result.decision.tool_calls[0].args == {
        "scope": "all",
        "status": "active",
        "limit": 10,
        "query": "Обучение сотрудников",
    }


def test_bitrix_llm_decide_routes_advanced_comment_closed_search(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.2,
                "tool_calls": [{"name": "bitrix_task_search", "args": {"scope": "my", "status": "closed"}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request=(
                    "Битрикс найди закрытые задачи, закрытые с 01.01.2025 по 31.12.2025, "
                    "где в комментариях есть слово наблюдателем. Покажи не больше 5."
                ),
                user={"id": "13"},
            ),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_task_search"]
    assert result.decision.tool_calls[0].args == {
        "scope": "all",
        "status": "closed",
        "limit": 5,
        "include_closed": True,
        "comment_query": "наблюдателем",
        "include_comments": True,
        "comment_lookup_task_limit": 200,
        "closed_from": "2025-01-01",
        "closed_to": "2025-12-31",
    }


def test_bitrix_llm_decide_routes_ai_close_problem_search(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс найди задачи с невыполненными пунктами после AI-закрытия.",
                user={"id": "13"},
            ),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_task_search"]
    assert result.decision.tool_calls[0].args == {
        "scope": "member",
        "status": "closed",
        "limit": 10,
        "include_closed": True,
        "ai_close_problem_type": "not_done",
    }


def test_bitrix_llm_decide_routes_task_close_report_incident_reply(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        ToolDefinition(name="task_close_report_incident", description="", parameters={}).model_dump(),
        ToolDefinition(name="none", description="", parameters={}).model_dump(),
    ]
    history = [
        {
            "role": "assistant",
            "content": (
                "Контроль AI-закрытия: файл отчёта исчез из задачи.\n"
                "Задача #303: Проверить архив\n"
                "Файл: AI-close-303-unconfirmed.txt\n"
                "1 — восстановить файл в задаче\n"
                "2 — всё в порядке, удалить эти данные из индекса"
            ),
        }
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="1", user={"id": "1"}),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
            dialog_history=history,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_report_incident"]
    assert result.decision.tool_calls[0].args == {
        "task_id": 303,
        "action": "restore",
        "file_name": "AI-close-303-unconfirmed.txt",
    }

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t2", request="второй вариант", user={"id": "1"}),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
            dialog_history=history,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_report_incident"]
    assert result.decision.tool_calls[0].args == {
        "task_id": 303,
        "action": "accept_missing",
        "file_name": "AI-close-303-unconfirmed.txt",
    }


def test_bitrix_llm_decide_routes_task_close_control_get(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        ToolDefinition(name="task_close_control_get", description="", parameters={}).model_dump(),
        ToolDefinition(name="task_close_control_update", description="", parameters={}).model_dump(),
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Покажи настройки контроля закрытия задач", user={"id": "1"}),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_control_get"]
    assert result.decision.tool_calls[0].args == {}

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t2", request="Покажи список операторов и контролируемых пользователей", user={"id": "1"}
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_control_get"]
    assert result.decision.tool_calls[0].args == {}

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t3",
                request="Покажи список операторов и обычных пользователей закрытия задач",
                user={"id": "1"},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_control_get"]
    assert result.decision.tool_calls[0].args == {}


def test_bitrix_llm_does_not_route_ambiguous_operator_panel_request(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    tool_definitions = [
        ToolDefinition(name="task_close_control_get", description="", parameters={}).model_dump(),
        ToolDefinition(name="task_close_control_update", description="", parameters={}).model_dump(),
    ]
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    service = BitrixLLMService(client, settings=get_settings())

    for request in (
        "Покажи список операторов",
        "Покажи список пользователей",
        "Покажи админ-панель",
        "Кто операторы",
    ):
        result = asyncio.run(
            service.decide(
                manifest=get_agent_manifest("bitrix24"),
                task=AgentTask(task_id="ambiguous", request=request, user={"id": "1"}),
                retrieval_hits=[],
                tool_definitions=tool_definitions,
            )
        )

        assert client.calls == []
        assert result.decision.status == "needs_clarification"
        assert "закрытие задач или отчет по машинам и людям" in result.decision.answer
        assert [call.name for call in result.decision.tool_calls] == ["none"]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="vehicle-panel",
                request="Покажи список операторов и пользователей отчета по машинам и людям",
                user={"id": "1"},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "unsupported_vehicle_usage_route"}
    assert result.decision.status == "needs_clarification"
    assert "Bitrix" in result.decision.answer
    assert "Логист" in result.decision.answer
    assert "машинам и людям" in result.decision.answer
    assert "всё равно обработай" in result.decision.answer
    assert [call.name for call in result.decision.tool_calls] == ["none"]


def test_bitrix_llm_allows_forced_bitrix_vehicle_request_to_reach_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        ToolDefinition(name="task_close_control_get", description="", parameters={}).model_dump(),
        ToolDefinition(name="none", description="", parameters={}).model_dump(),
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="forced-bitrix-vehicle-panel",
                request="Битрикс, всё равно обработай этот запрос: покажи отчёт по машинам и людям",
                user={"id": "1"},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls
    assert result.raw == {}
    assert [call.name for call in result.decision.tool_calls] == ["none"]


def test_bitrix_llm_decide_routes_task_close_control_update(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        ToolDefinition(name="task_close_control_get", description="", parameters={}).model_dump(),
        ToolDefinition(name="task_close_control_update", description="", parameters={}).model_dump(),
        ToolDefinition(name="none", description="", parameters={}).model_dump(),
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Добавь пользователя 231 в контролируемые", user={"id": "1"}),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_control_update"]
    assert result.decision.tool_calls[0].args == {"action": "add_controlled_user", "target_user_id": 231}

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t2", request="Поставь автозакрытие задач на 19:30", user={"id": "1"}),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["task_close_control_update"]
    assert result.decision.tool_calls[0].args == {"action": "set_auto_close_time", "auto_close_time": "19:30"}


def test_bitrix_specialist_injects_admin_context_for_task_close_control_tool(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeTaskDraftStore()

    class TaskCloseControlTools:
        def agent_tools(self):
            return [TaskCloseControlUpdateTool(store=store)]

        def make_user_client(self):
            class UserClient:
                async def get_user(self, user_id: int):
                    return {"ID": user_id, "NAME": "Admin", "LAST_NAME": "User", "IS_ADMIN": True}

            return UserClient()

    llm = FakeBitrixLLM(
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_control_update",
                args={"action": "add_operator", "target_user_id": 13},
            )
        ]
    )

    result = asyncio.run(
        _bitrix_specialist(tools=TaskCloseControlTools(), llm=llm).handle(
            AgentTask(task_id="t1", request="добавь 13 оператором контроля закрытия задач", user={"id": "1"})
        )
    )

    assert result.status == "completed"
    assert store.task_close_operator_ids() == {13}
    assert store.task_close_controlled_user_ids() == set()


def test_bitrix_specialist_uses_configured_task_close_control_admin_ids(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_TASK_CLOSE_CONTROL_ADMIN_USER_IDS", "1")
    store = FakeTaskDraftStore()

    class TaskCloseControlTools:
        def agent_tools(self):
            return [TaskCloseControlUpdateTool(store=store)]

        def make_user_client(self):
            class UserClient:
                async def get_user(self, user_id: int):
                    return {
                        "ID": user_id,
                        "NAME": "Configured",
                        "LAST_NAME": "Admin",
                    }

            return UserClient()

    llm = FakeBitrixLLM(
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_control_update",
                args={"action": "add_operator", "target_user_id": 13},
            )
        ]
    )

    result = asyncio.run(
        _bitrix_specialist(tools=TaskCloseControlTools(), llm=llm, settings=get_settings()).handle(
            AgentTask(task_id="t1", request="добавь 13 оператором контроля закрытия задач", user={"id": "1"})
        )
    )

    assert result.status == "completed"
    assert store.task_close_operator_ids() == {13}
    assert store.task_close_controlled_user_ids() == set()


def test_bitrix_llm_decide_routes_project_search_to_deterministic_tool(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.2,
                "tool_calls": [{"name": "bitrix_api", "args": {"method": "sonet_group.get", "params": {}}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Битрикс найди проект Ларгус 2.", user={"id": "13"}),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_project_search"]
    assert result.decision.tool_calls[0].args == {"query": "Ларгус 2", "limit": 10}


def test_bitrix_llm_decide_routes_hyphenated_project_search_to_deterministic_tool(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(task_id="t1", request="Битрикс найди проект Ларгус-2.", user={"id": "13"}),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_project_search"]
    assert result.decision.tool_calls[0].args == {"query": "Ларгус-2", "limit": 10}
    assert client.calls == []


def test_bitrix_llm_decide_strips_project_search_tail_instruction(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс найди проект Ларгус-2. Ответь кратко, только название и ссылку.",
                user={"id": "13"},
            ),
            retrieval_hits=[],
            tool_definitions=_bitrix_read_tool_definitions(),
        )
    )

    assert [call.name for call in result.decision.tool_calls] == ["bitrix_project_search"]
    assert result.decision.tool_calls[0].args == {"query": "Ларгус-2", "limit": 10}
    assert client.calls == []


def test_bitrix_specialist_passes_user_profile_and_permission_rag_to_llm():
    llm = FakeBitrixLLM()
    tools = FakeResolverTools(
        raw_user={
            "ID": "9",
            "ACTIVE": "Y",
            "NAME": "Иван",
            "LAST_NAME": "Иванов",
            "WORK_POSITION": "Монтажник",
            "UF_DEPARTMENT": ["77"],
            "IS_ADMIN": "N",
            "USER_TYPE": "employee",
        }
    )

    result = asyncio.run(
        _bitrix_specialist(tools=tools, llm=llm).handle(
            AgentTask(task_id="t1", request="Создай задачу на меня проверить камеру", user={"id": "9"})
        )
    )

    context = llm.decide_calls[0]["task"].context
    assert result.actions_taken[0].details["bitrix_current_user_profile_status"] == "ok"
    assert context["bitrix_current_user_profile"]["status"] == "ok"
    assert context["bitrix_current_user_profile"]["data"]["profile"]["work_position"] == "Монтажник"
    assert context["permission_policy_context"]
    assert {item["topic"] for item in context["permission_policy_context"]} == {"user_permissions"}
    assert "user_permissions" in result.actions_taken[0].details["permission_policy_topics"]


def test_bitrix_specialist_task_create_draft_saves_to_store():
    store = FakeTaskDraftStore()
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    specialist = Bitrix24Specialist(
        manifest,
        retriever=retriever,
        agent_tools=[TaskCreateDraftTool(store=store)],
        llm=FakeBitrixLLM(
            tool_calls=[
                BitrixLLMToolCall(
                    name="task_create_draft",
                    args={
                        "title": "проверить IP-камеру",
                        "responsible_self": True,
                        "deadline_iso": "2026-06-05T19:00:00+03:00",
                    },
                )
            ],
            final_status="needs_human",
            final_answer="Подготовил черновик задачи, нужно подтверждение.",
        ),
    )
    result = asyncio.run(
        specialist.handle(
            AgentTask(
                task_id="t1",
                request="Создай задачу на меня проверить IP-камеру завтра",
                user={"id": "9"},
                context={"dialog_key": "d:9"},
            )
        )
    )

    assert result.status == "needs_human"
    assert result.actions_requiring_approval == []
    assert "d:9" in store._drafts
    draft_fields = store._drafts["d:9"]["fields"]
    assert draft_fields["TITLE"] == "проверить IP-камеру"
    assert draft_fields["RESPONSIBLE_ID"] == 9
    assert draft_fields["CREATED_BY"] == 9
    assert draft_fields["DEADLINE"]


def test_bitrix_specialist_exact_release_task_draft_is_deterministic_and_persisted():
    store = FakeTaskDraftStore()
    client = RecordingLLMClient("not-json")

    class _UserClient:
        async def get_user(self, user_id: int):
            return {
                "ID": str(user_id),
                "NAME": "Иван",
                "LAST_NAME": "Иванов",
                "ACTIVE": "Y",
                "IS_ADMIN": "N",
                "USER_TYPE": "employee",
                "WORK_POSITION": "Монтажник",
            }

        async def search_projects(self, query: str, *, limit: int = 10):
            assert query == "Иванов Иван"
            return [{"ID": "43", "NAME": "Иванов Иван", "PROJECT": "Y"}]

    user_client = _UserClient()
    specialist = Bitrix24Specialist(
        get_agent_manifest("bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        agent_tools=[TaskCreateDraftTool(store=store, project_client=user_client)],
        bitrix_user_client=user_client,
        llm=BitrixLLMService(client, settings=get_settings()),
    )

    result = asyncio.run(
        specialist.handle(
            AgentTask(
                task_id="t1",
                request=(
                    "Битрикс создай задачу на меня: подготовить тестовый отчет. "
                    "Не создавай сразу, только покажи черновик для подтверждения."
                ),
                user={"id": "9"},
                context={"dialog_key": "d:9", "dialog_id": "chat9"},
            )
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert all(phrase in result.answer for phrase in ("Черновик задачи", "Срок", "Если всё верно"))
    assert store._drafts["d:9"]["_draft_type"] == "task_create"
    assert store._drafts["d:9"]["fields"]["TITLE"] == "подготовить тестовый отчет"
    assert store._drafts["d:9"]["fields"]["RESPONSIBLE_ID"] == 9


def test_bitrix_specialist_enforces_structured_task_close_draft_response():
    store = FakeTaskDraftStore()
    specialist = Bitrix24Specialist(
        get_agent_manifest("bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        agent_tools=[TaskCloseDraftTool(store=store)],
        llm=FakeBitrixLLM(
            tool_calls=[
                BitrixLLMToolCall(
                    name="task_close_draft",
                    args={
                        "task_id": 8869,
                        "task_title": "Проверить камеру",
                        "completion_summary": "Камеру проверили, архив не проверяли.",
                        "task_points": ["камера проверена", "архив не проверен"],
                        "equipment_consumables": "не указано",
                        "overall_status": "partial",
                        "not_done_items": ["архив не проверен"],
                        "unconfirmed_items": ["нет фото результата"],
                        "status_reasons": ["нет доступа к архиву"],
                    },
                )
            ],
            final_status="needs_clarification",
            final_answer="Черновик закрытия уже подготовлен: короткий свободный текст.",
        ),
    )

    result = asyncio.run(
        specialist.handle(
            AgentTask(
                task_id="t1",
                request="закрой задачу 8869",
                user={"id": "13"},
                context={"dialog_key": "d:13", "dialog_id": "chat4321"},
            )
        )
    )

    assert result.status == "needs_human"
    assert "d:13" in store._drafts
    assert "Черновик #8869: Проверить камеру" in result.answer
    assert "1. Выполняемые работы" in result.answer
    assert "1.1 камера проверена - ... ???" in result.answer
    assert "1.3 Еще работы - ... ???" in result.answer
    assert "2. Использовано материалов, оборудование" in result.answer
    assert "2.1 не указано" in result.answer
    assert "3. Статус выполнения работ" in result.answer
    assert "[ВЫБРАНО: выполнено частично]" in result.answer
    assert "3.1 причина: нет доступа к архиву" in result.answer
    assert "Причина/что не выполнено: архив не проверен" not in result.answer
    assert "Не подтверждено: нет фото результата" not in result.answer
    assert "4. Дополнительная информация" in result.answer
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" not in result.answer
    assert "уже подготовлен" not in result.answer


def test_bitrix_specialist_task_create_draft_uses_current_user_profile_label():
    store = FakeTaskDraftStore()
    llm = FakeBitrixLLM(
        tool_calls=[
            BitrixLLMToolCall(
                name="task_create_draft",
                args={
                    "title": "проверить IP-камеру",
                    "responsible_self": True,
                },
            )
        ],
        final_status="needs_human",
        final_answer="Подготовил черновик задачи, нужно подтверждение.",
    )

    class _FakeUserClient:
        async def get_user(self, user_id: int):
            return {"ID": str(user_id), "NAME": "Дмитрий", "LAST_NAME": "Кулинич"}

        async def search_projects(self, query: str, *, limit: int = 10):
            assert query == "Кулинич Дмитрий"
            return [{"ID": "43", "NAME": "Кулинич Дмитрий", "PROJECT": "Y"}]

    user_client = _FakeUserClient()
    specialist = Bitrix24Specialist(
        get_agent_manifest("bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        agent_tools=[TaskCreateDraftTool(store=store, project_client=user_client)],
        bitrix_user_client=user_client,
        llm=llm,
    )

    result = asyncio.run(
        specialist.handle(
            AgentTask(
                task_id="t1",
                request="Создай задачу на меня проверить IP-камеру",
                user={"id": "13"},
                context={"dialog_key": "d:13"},
            )
        )
    )

    assert result.status == "needs_human"
    preview = llm.compose_calls[0]["tool_results"][0].data["preview"]
    fields = llm.compose_calls[0]["tool_results"][0].data["params"]["fields"]
    assert preview["responsible"] == "Кулинич Дмитрий"
    assert preview["project"] == "Кулинич Дмитрий"
    assert fields["GROUP_ID"] == 43
    assert "текущий пользователь" not in preview["responsible"]


def test_bitrix_specialist_asks_for_responsible_before_task_create():
    result = asyncio.run(
        _bitrix_specialist(
            llm=FakeBitrixLLM(
                tool_calls=[BitrixLLMToolCall(name="none")],
                decision_status="needs_clarification",
                final_status="needs_clarification",
                final_answer="Кого поставить ответственным?",
            )
        ).handle(AgentTask(task_id="t1", request="Создай задачу проверить IP-камеру"))
    )

    assert result.status == "needs_clarification"
    assert result.actions_requiring_approval == []
    assert "ответственн" in result.answer


def test_bitrix_specialist_rejects_incomplete_task_create_tool_call():
    store = FakeTaskDraftStore()
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    specialist = Bitrix24Specialist(
        manifest,
        retriever=retriever,
        agent_tools=[TaskCreateDraftTool(store=store)],
        llm=FakeBitrixLLM(
            tool_calls=[
                BitrixLLMToolCall(
                    name="task_create_draft",
                    args={"title": "проверить камеру"},
                )
            ],
            final_status="failed",
            final_answer="LLM-субагент вызвал task_create_draft с неполным контрактом.",
        ),
    )
    result = asyncio.run(specialist.handle(AgentTask(task_id="t1", request="Создай задачу проверить камеру")))

    assert result.status == "failed"
    assert result.actions_requiring_approval == []
    tool_action = next(a for a in result.actions_taken if a.name == "task_create_draft")
    assert tool_action.status == "contract_violation"
    assert tool_action.details["data"]["contract_errors"] == [
        "task_create_draft requires one of responsible_id or responsible_self",
    ]


def test_bitrix_specialist_task_create_stores_explicit_ids():
    store = FakeTaskDraftStore()
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    specialist = Bitrix24Specialist(
        manifest,
        retriever=retriever,
        agent_tools=[TaskCreateDraftTool(store=store)],
        llm=FakeBitrixLLM(
            tool_calls=[
                BitrixLLMToolCall(
                    name="task_create_draft",
                    args={
                        "title": "проверить регистратор",
                        "responsible_id": 15,
                        "group_id": 44,
                        "deadline_iso": "2026-06-05T19:00:00+03:00",
                    },
                )
            ],
            final_status="needs_human",
            final_answer="Подготовил черновик задачи, нужно подтверждение.",
        ),
    )
    result = asyncio.run(
        specialist.handle(
            AgentTask(
                task_id="t1",
                request="Поставь задачу ответственному #15 проверить регистратор в проекте #44 до 05.06.2026",
                user={"id": "9"},
                context={"dialog_key": "d:9"},
            )
        )
    )

    assert result.status == "needs_human"
    assert result.actions_requiring_approval == []
    draft_fields = store._drafts["d:9"]["fields"]
    assert draft_fields["TITLE"] == "проверить регистратор"
    assert draft_fields["RESPONSIBLE_ID"] == 15
    assert draft_fields["GROUP_ID"] == 44
    assert draft_fields["DEADLINE"].startswith("2026-06-05T19:00:00")


def test_bitrix_llm_decision_payload_includes_permission_context(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        '{"status":"completed","answer":"","confidence":0.7,"tool_calls":[{"name":"none","args":{},"summary":""}]}'
    )
    manifest = get_agent_manifest("bitrix24")
    context = {
        "bitrix_current_user_profile": {
            "status": "ok",
            "tool": "current_user_profile",
            "data": {"user_id": 15, "profile": {"work_position": "Монтажник"}},
        },
        "permission_policy_context": [
            {"topic": "user_permissions", "section": "Монтажники", "text": "Монтажник может закрывать свои задачи."}
        ],
    }

    asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(task_id="t1", request="Создай задачу", user={"id": "15"}, context=context),
            retrieval_hits=[],
            tool_definitions=[],
        )
    )

    payload = json.loads(client.calls[0]["messages"][1]["content"])
    permission = payload["permission_context"]
    assert permission["current_user_id"] == 15
    assert permission["current_dialog_id"] == ""
    assert permission["current_user_write_profile"] == "member_write"
    assert permission["pending_task_draft"] is None
    assert permission["bitrix_current_user_profile"] == context["bitrix_current_user_profile"]
    assert permission["permission_policy_context"] == context["permission_policy_context"]
    assert any("business writes must be prepared as drafts" in rule for rule in permission["rules"])
    assert any("confirmed writes execute only through OAuth" in rule for rule in permission["rules"])
    assert any("Bitrix itself decides final permissions" in rule for rule in permission["rules"])


def test_bitrix_llm_routes_task_draft_confirmation_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.7,
                "tool_calls": [
                    {"name": "task_create_confirm", "args": {}, "summary": "confirmed"},
                    {"name": "task_draft_discard", "args": {}, "summary": "discarded"},
                ],
            }
        )
    )
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_create_draft", "description": "", "parameters": {}},
        {"name": "task_create_confirm", "description": "", "parameters": {}},
        {"name": "task_draft_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="да, создай",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "fields": {"TITLE": "test", "RESPONSIBLE_ID": 15, "CREATED_BY": 15, "NO_DEADLINE": True}
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert {"task_create_confirm", "task_draft_discard"} <= ALLOWED_TOOL_NAMES
    assert client.calls == []
    assert result.raw == {"source": "draft_confirm_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_create_confirm"]


def test_bitrix_llm_routes_task_close_confirmation_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.7,
                "tool_calls": [
                    {"name": "task_close_confirm", "args": {}, "summary": "confirmed"},
                    {"name": "task_close_discard", "args": {}, "summary": "discarded"},
                ],
            }
        )
    )
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="да, закрывай",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 139,
                        "task_title": "Обучение сотрудников",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert {"task_close_draft", "task_close_confirm", "task_close_discard"} <= ALLOWED_TOOL_NAMES
    assert client.calls == []
    assert result.raw == {"source": "draft_confirm_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_close_confirm"]


def test_bitrix_llm_routes_task_close_start_to_draft_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="Битрикс закрой задачу 8899.",
                user={"id": "15"},
                context={"dialog_id": "chat4321"},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_start_route"}
    assert result.decision.status == "completed"
    assert [call.name for call in result.decision.tool_calls] == ["task_close_draft"]
    args = result.decision.tool_calls[0].args
    assert args["task_id"] == 8899
    assert args["overall_status"] == "unconfirmed"
    assert args["unconfirmed_items"] == ["результат выполнения не указан"]
    assert "что сделано по задаче" in args["missing_fields"]


def test_bitrix_llm_routes_task_close_followup_to_active_draft_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request=(
                    "По задаче 8899: проверка выполнена частично, не все пункты подтверждены, "
                    "оборудование не использовалось."
                ),
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить камеры",
                        "task_points": ["Проверить камеру", "Проверить архив"],
                        "overall_status": "unconfirmed",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_close_draft"]
    args = result.decision.tool_calls[0].args
    assert args["task_id"] == 8899
    assert args["task_title"] == "Проверить камеры"
    assert args["overall_status"] == "partial"
    assert args["equipment_consumables"] == "не использовалось"
    assert "unconfirmed_items" not in args
    assert args["missing_fields"] == []


def test_bitrix_llm_routes_task_close_followup_extracts_four_block_fields(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request=(
                    "По задаче 8899: 1.1 Проверить камеру выполнено, "
                    "1.2 Забрать документы не выполнено, 1.3 Починить селектор не подтверждено. "
                    "По пункту 2 оборудование: 3 видеокамеры, 40 метров кабеля. "
                    "По пункту 3 выполнено частично, причина: не было доступа к архиву. "
                    "По пункту 4 нужно вернуться завтра с доступом."
                ),
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру", "Забрать документы", "Починить селектор"],
                        "overall_status": "unconfirmed",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["completion_summary"] == (
        "Проверить камеру выполнено. Забрать документы не выполнено. Починить селектор не подтверждено"
    )
    assert args["equipment_consumables"] == "3 видеокамеры, 40 метров кабеля"
    assert args["additional_info"] == "нужно вернуться завтра с доступом"
    assert args["overall_status"] == "partial"
    assert args["not_done_items"] == ["Забрать документы"]
    assert args["unconfirmed_items"] == ["Починить селектор"]
    assert args["status_reasons"] == ["не было доступа к архиву"]
    assert args["missing_fields"] == []


def test_bitrix_llm_routes_task_close_followup_extracts_compact_numbered_blocks(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request=(
                    "По задаче 8899: 1.1 выполнено полностью. "
                    "1.2 не выполнено. 1.3 не подтверждено. "
                    "2 использовал 2 камеры, 30 метров кабеля. "
                    "3 выполнено частично, причина не было доступа к архиву. "
                    "4 отсутствует."
                ),
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру", "Забрать документы", "Починить селектор"],
                        "overall_status": "unconfirmed",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["completion_summary"] == (
        "Проверить камеру выполнено полностью. Забрать документы не выполнено. Починить селектор не подтверждено"
    )
    assert args["equipment_consumables"] == "использовал 2 камеры, 30 метров кабеля"
    assert args["additional_info"] == "отсутствует"
    assert args["overall_status"] == "partial"
    assert args["not_done_items"] == ["Забрать документы"]
    assert args["unconfirmed_items"] == ["Починить селектор"]
    assert args["status_reasons"] == ["не было доступа к архиву"]
    assert args["missing_fields"] == []


def test_bitrix_llm_routes_task_close_followup_keeps_work_block_isolated(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request=(
                    "По задаче 8899: 1.1 камеру не поставил, потому что не было оборудования "
                    "и не было доступа к архиву."
                ),
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру", "Забрать документы"],
                        "overall_status": "unconfirmed",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["completion_summary"] == "Проверить камеру не выполнено"
    assert args["not_done_items"] == ["Проверить камеру"]
    assert "equipment_consumables" not in args
    assert "overall_status" not in args
    assert "status_reasons" not in args
    assert "additional_info" not in args


def test_bitrix_llm_routes_task_close_followup_keeps_status_block_out_of_equipment(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="По задаче 8899: 3 выполнено частично, причина: не хватило 30 метров кабеля.",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру"],
                        "overall_status": "unconfirmed",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["overall_status"] == "partial"
    assert args["status_reasons"] == ["не хватило 30 метров кабеля"]
    assert "equipment_consumables" not in args
    assert "not_done_items" not in args


def test_bitrix_llm_routes_task_close_followup_filters_wrong_extra_info_block_parts(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request=("По задаче 8899: 4 нужно вернуться завтра, использовал 5 датчиков и работа не выполнена."),
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру"],
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["additional_info"] == "нужно вернуться завтра"
    assert "equipment_consumables" not in args
    assert "overall_status" not in args


def test_bitrix_llm_routes_task_close_followup_keeps_future_extra_info_with_equipment_words(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="По задаче 8899: 4 нужно взять с собой 5 камер, вернуться завтра.",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру"],
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["additional_info"] == "нужно взять с собой 5 камер, вернуться завтра"
    assert "equipment_consumables" not in args


def test_bitrix_llm_routes_task_close_followup_completed_status_has_no_fake_reason(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request=(
                    "По задаче 8899: 1.1 выполнено полностью. "
                    "2 не использовалось. 3 выполнено полностью. 4 отсутствует."
                ),
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 8899,
                        "task_title": "Проверить объект",
                        "task_points": ["Проверить камеру"],
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_active_update_route"}
    args = result.decision.tool_calls[0].args
    assert args["overall_status"] == "completed"
    assert "status_reasons" not in args


def test_bitrix_llm_routes_task_close_conflict_switch_to_requested_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "task_close_draft", "description": "", "parameters": {}},
        {"name": "task_close_confirm", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="3",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 139,
                        "task_title": "Проверить камеры",
                        "_task_close_conflict_pending": {
                            "_draft_type": "task_close",
                            "task_id": 140,
                            "task_title": "Проверить регистратор",
                            "completion_summary": "Регистратор проверен.",
                            "task_points": ["регистратор проверен"],
                        },
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_close_draft_conflict_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_close_discard", "task_close_draft"]
    assert result.decision.tool_calls[1].args["task_id"] == 140
    assert result.decision.tool_calls[1].args["task_title"] == "Проверить регистратор"
    assert result.decision.tool_calls[1].args["task_points"] == ["регистратор проверен"]


def test_bitrix_llm_routes_calendar_confirmation_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.7,
                "tool_calls": [
                    {"name": "calendar_event_confirm", "args": {}, "summary": "confirmed"},
                    {"name": "calendar_event_discard", "args": {}, "summary": "discarded"},
                ],
            }
        )
    )
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "calendar_event_draft", "description": "", "parameters": {}},
        {"name": "calendar_event_confirm", "description": "", "parameters": {}},
        {"name": "calendar_event_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="да, добавь в календарь",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "calendar_event",
                        "title": "позвонить Борисову",
                        "start_iso": "2026-07-09T12:00:00+03:00",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert {"calendar_event_draft", "calendar_event_confirm", "calendar_event_discard"} <= ALLOWED_TOOL_NAMES
    assert client.calls == []
    assert result.raw == {"source": "draft_confirm_route"}
    assert [call.name for call in result.decision.tool_calls] == ["calendar_event_confirm"]


def test_bitrix_llm_decide_routes_simple_reminder_to_calendar_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setattr(bitrix_llm, "_moscow_today", lambda: date(2026, 7, 10))
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "Черновик события календаря: ...",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "calendar_event_draft", "description": "", "parameters": {}},
        {"name": "calendar_event_confirm", "description": "", "parameters": {}},
        {"name": "calendar_event_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request=(
                    "Битрикс напомни мне завтра позвонить Борисову. "
                    "Не добавляй сразу, только покажи черновик календаря для подтверждения."
                ),
                user={"id": "13"},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "calendar_reminder_route"}
    assert [call.name for call in result.decision.tool_calls] == ["calendar_event_draft"]
    assert result.decision.tool_calls[0].args == {
        "title": "Позвонить Борисову",
        "date_iso": "2026-07-11",
        "description": "Позвонить Борисову",
    }


def test_bitrix_llm_decide_routes_task_draft_discard_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "Нет активного черновика задачи для отмены.",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "task_draft_discard", "description": "", "parameters": {}},
        {"name": "calendar_event_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс отмени черновик задачи.",
                user={"id": "13"},
                context={"dialog_id": "chat4321", "pending_task_draft": {"fields": {"TITLE": "тест"}}},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "draft_discard_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_draft_discard"]
    assert result.decision.tool_calls[0].args == {}


def test_bitrix_llm_does_not_treat_preview_request_as_task_draft_discard(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.7,
                "tool_calls": [
                    {
                        "name": "task_create_draft",
                        "args": {"title": "подготовить тестовый отчет", "responsible_self": True},
                    }
                ],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "task_create_draft", "description": "", "parameters": {}},
        {"name": "task_draft_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request=(
                    "Битрикс создай задачу на меня: подготовить тестовый отчет. "
                    "Не создавай сразу, только покажи черновик для подтверждения."
                ),
                user={"id": "13"},
                context={"dialog_id": "chat4321", "bitrix_current_user_profile": _writable_profile()},
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert len(client.calls) == 0
    assert result.raw == {"source": "task_create_draft_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_create_draft"]
    assert result.decision.tool_calls[0].args == {
        "title": "подготовить тестовый отчет",
        "responsible_self": True,
    }


@pytest.mark.parametrize(
    ("request_text", "expected_title"),
    [
        ("Создай задачу на меня: проверить камеры.", "проверить камеры"),
        ("Поставьте задачу для меня: подготовить акт! Только покажи черновик.", "подготовить акт"),
        ("Создать задачу: сверить остатки. Не создавай сразу.", "сверить остатки"),
    ],
)
def test_bitrix_llm_routes_unambiguous_self_task_draft_without_llm(monkeypatch, request_text, expected_title):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient("not-json")
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request=request_text,
                user={"id": "13"},
                context={"dialog_id": "chat4321", "bitrix_current_user_profile": _writable_profile()},
            ),
            retrieval_hits=[],
            tool_definitions=[{"name": "task_create_draft", "description": "", "parameters": {}}],
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "task_create_draft_route"}
    assert result.decision.tool_calls[0].args == {"title": expected_title, "responsible_self": True}


@pytest.mark.parametrize(
    "request_text",
    [
        "Создай задачу в проекте Ларгус-2 на меня: проверить документы.",
        "Создай задачу на меня: подготовить документы для проекта Ларгус-2.",
        "Создай задачу для Коверги Дмитрия: проверить документы.",
        "Создай задачу: проверить документы, назначь Иванова исполнителем.",
        "Создай задачу: поручи Иванову проверить документы.",
        "Создай задачу: проверить документы, ответственный Иванов.",
        "Создай задачу на меня без сформулированного заголовка",
    ],
)
def test_bitrix_llm_leaves_lookup_dependent_or_incomplete_task_drafts_to_model(monkeypatch, request_text):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps({"status": "needs_clarification", "answer": "Уточните детали.", "confidence": 0.5, "tool_calls": []})
    )
    service = BitrixLLMService(client, settings=get_settings())

    asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request=request_text,
                user={"id": "13"},
                context={"dialog_id": "chat4321", "bitrix_current_user_profile": _writable_profile()},
            ),
            retrieval_hits=[],
            tool_definitions=[{"name": "task_create_draft", "description": "", "parameters": {}}],
        )
    )

    assert len(client.calls) == 1


def test_bitrix_llm_decide_routes_task_close_draft_discard_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "Нет активного черновика закрытия задачи для отмены.",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "task_draft_discard", "description": "", "parameters": {}},
        {"name": "task_close_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс отмени черновик закрытия задачи.",
                user={"id": "13"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "task_close",
                        "task_id": 139,
                        "task_title": "Обучение сотрудников",
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "draft_discard_route"}
    assert [call.name for call in result.decision.tool_calls] == ["task_close_discard"]
    assert result.decision.tool_calls[0].args == {}


def test_bitrix_llm_decide_routes_calendar_draft_discard_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "Нет активного черновика события календаря для отмены.",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "task_draft_discard", "description": "", "parameters": {}},
        {"name": "calendar_event_discard", "description": "", "parameters": {}},
        {"name": "project_create_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс отмени черновик события календаря.",
                user={"id": "13"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {"_draft_type": "calendar_event", "title": "Позвонить Борисову"},
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "draft_discard_route"}
    assert [call.name for call in result.decision.tool_calls] == ["calendar_event_discard"]
    assert result.decision.tool_calls[0].args == {}


def test_bitrix_llm_decide_routes_project_draft_discard_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "Нет активного черновика проекта для отмены.",
                "confidence": 0.2,
                "tool_calls": [{"name": "none", "args": {}}],
            }
        )
    )
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "task_draft_discard", "description": "", "parameters": {}},
        {"name": "calendar_event_discard", "description": "", "parameters": {}},
        {"name": "project_create_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс отмени черновик проекта.",
                user={"id": "13"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {"_draft_type": "project_create", "params": {"fields": {"NAME": "Иванов"}}},
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert result.raw == {"source": "draft_discard_route"}
    assert [call.name for call in result.decision.tool_calls] == ["project_create_discard"]
    assert result.decision.tool_calls[0].args == {}


def test_bitrix_llm_uses_active_project_draft_type_when_cancel_text_says_task(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient("{}")
    service = BitrixLLMService(client, settings=get_settings())
    tool_definitions = [
        {"name": "task_draft_discard", "description": "", "parameters": {}},
        {"name": "project_create_discard", "description": "", "parameters": {}},
        {"name": "none", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("bitrix24"),
            task=AgentTask(
                task_id="t1",
                request="Битрикс отмени черновик задачи.",
                user={"id": "13"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {"_draft_type": "project_create", "params": {"fields": {"NAME": "Тест"}}},
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert client.calls == []
    assert [call.name for call in result.decision.tool_calls] == ["project_create_discard"]


def test_bitrix_llm_routes_project_confirmation_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        json.dumps(
            {
                "status": "completed",
                "answer": "",
                "confidence": 0.7,
                "tool_calls": [
                    {"name": "project_create_confirm", "args": {}, "summary": "confirmed"},
                    {"name": "project_create_discard", "args": {}, "summary": "discarded"},
                ],
            }
        )
    )
    manifest = get_agent_manifest("bitrix24")
    tool_definitions = [
        {"name": "project_create_draft", "description": "", "parameters": {}},
        {"name": "project_create_confirm", "description": "", "parameters": {}},
        {"name": "project_create_discard", "description": "", "parameters": {}},
    ]

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="t1",
                request="да, создай проект",
                user={"id": "15"},
                context={
                    "dialog_id": "chat4321",
                    "pending_task_draft": {
                        "_draft_type": "project_create",
                        "method": "sonet_group.create",
                        "params": {"fields": {"NAME": "Кулинич Валерий"}},
                    },
                },
            ),
            retrieval_hits=[],
            tool_definitions=tool_definitions,
        )
    )

    assert {"project_create_draft", "project_create_confirm", "project_create_discard"} <= ALLOWED_TOOL_NAMES
    assert client.calls == []
    assert result.raw == {"source": "draft_confirm_route"}
    assert [call.name for call in result.decision.tool_calls] == ["project_create_confirm"]


def test_bitrix_llm_compose_formats_project_create_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="создай проект", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="project_create_draft",
                    data={
                        "params": {"params": {"fields": {"NAME": "Кулинич Валерий"}}},
                        "preview": {
                            "name": "Кулинич Валерий",
                            "type": "личный проект",
                            "visibility": "открытый",
                            "description": "Личный проект Кулинич Валерий",
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Черновик проекта:" in result.answer
    assert "Название: Кулинич Валерий" in result.answer
    assert "Тип: личный проект" in result.answer
    assert "Открытость: открытый" in result.answer
    assert "Если всё верно" in result.answer


def test_bitrix_llm_compose_formats_project_create_confirm_with_project_link(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    settings = get_settings()
    service = BitrixLLMService(client, settings=settings)

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="да, создай проект", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="project_create_confirm",
                    data={
                        "result": {"result": 777},
                        "params": {"fields": {"NAME": "Кулинич Валерий"}},
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert "Проект создан:" in result.answer
    assert "Кулинич Валерий" in result.answer
    assert "/workgroups/group/777/" in result.answer
    assert "/company/personal/user/" not in result.answer


def test_bitrix_llm_compose_formats_project_create_confirm_with_followup_task(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="да, создай проект", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="project_create_confirm",
                    data={
                        "result": {"result": 777},
                        "params": {"fields": {"NAME": "Кулинич Валерий"}},
                        "followup_task_draft": {
                            "params": {
                                "fields": {
                                    "TITLE": "проверить IP-камеру",
                                    "RESPONSIBLE_ID": 13,
                                    "GROUP_ID": 777,
                                    "DEADLINE": "2026-07-20T19:00:00+03:00",
                                    "DESCRIPTION": "Краткое содержание: проверить IP-камеру",
                                }
                            },
                            "preview": {
                                "title": "проверить IP-камеру",
                                "responsible": "Кулинич Валерий",
                                "project": "Кулинич Валерий",
                                "deadline": "20.07.2026 19:00 МСК",
                                "description": "Краткое содержание: проверить IP-камеру",
                            },
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Проект создан:" in result.answer
    assert "Черновик задачи:" in result.answer
    assert "Проект: Кулинич Валерий" in result.answer
    assert "Если всё верно, напишите: да, создай." in result.answer


def test_bitrix_llm_compose_formats_project_create_discard_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="отмени черновик проекта", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[ToolResult(status="ok", tool="project_create_discard", data={"discarded": True})],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert result.answer == "Черновик проекта удалён."


def test_bitrix_llm_compose_formats_task_draft_without_profile_links(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="создай задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_create_draft",
                    data={
                        "params": {
                            "fields": {
                                "TITLE": "тест подтверждения",
                                "DESCRIPTION": "Краткое содержание: тест подтверждения",
                                "RESPONSIBLE_ID": 13,
                                "DEADLINE": "2026-07-13T19:00:00+03:00",
                            }
                        },
                        "preview": {
                            "title": "тест подтверждения",
                            "responsible": "Дмитрий",
                            "project": "Ларгус 2",
                            "deadline": "13.07.2026 19:00 МСК",
                            "description": "Краткое содержание: тест подтверждения",
                        },
                        "notes": ["Срок по умолчанию: три рабочих дня от даты создания, 19:00 МСК."],
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Черновик задачи" in result.answer
    assert "по умолчанию: 3 рабочих дня" not in result.answer
    assert "13.07.2026 19:00 МСК" in result.answer
    assert "Проект: Ларгус 2" in result.answer
    assert "Описание: тест подтверждения" in result.answer
    assert "Описание: Краткое содержание:" not in result.answer
    assert "[URL" not in result.answer
    assert "company/personal/user" not in result.answer
    assert "#13" not in result.answer


def test_bitrix_llm_compose_formats_task_confirm_with_task_link_only(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://asutp-expert.bitrix24.ru/rest/1/token/")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="да, создай", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_create_confirm",
                    data={
                        "result": {"result": {"task": {"id": 8851}}},
                        "params": {"fields": {"TITLE": "тест подтверждения", "RESPONSIBLE_ID": 13}},
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert result.answer == (
        "Задача создана: "
        "[URL=https://asutp-expert.bitrix24.ru/company/personal/user/0/tasks/task/view/8851/]"
        "тест подтверждения[/URL]."
    )
    assert "/company/personal/user/13/" not in result.answer


def test_bitrix_llm_compose_formats_task_close_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {
                            "_draft_type": "task_close",
                            "task_id": 139,
                            "task_title": "Обучение сотрудников",
                            "action": "complete",
                            "completion_summary": "Пользователь подтвердил выполнение.",
                            "unresolved_items": ["не приложен акт проверки"],
                        },
                        "preview": {
                            "task_title": "Обучение сотрудников",
                            "action_label": "отметить задачу выполненной",
                            "completion_summary": "Пользователь подтвердил выполнение.",
                            "unresolved_items": ["не приложен акт проверки"],
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Черновик #139: Обучение сотрудников" in result.answer
    assert "Обучение сотрудников" in result.answer
    assert "1.1 Пользователь подтвердил выполнение" in result.answer
    assert "Не подтверждено: не приложен акт проверки" not in result.answer
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" not in result.answer
    assert "[URL" not in result.answer


def test_bitrix_llm_compose_formats_structured_task_close_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {"_draft_type": "task_close", "task_id": 139, "task_title": "Проверить камеры"},
                        "preview": {
                            "task_title": "Проверить камеры",
                            "action_label": "отметить задачу выполненной",
                            "completion_summary": "Часть работ выполнена.",
                            "task_points": ["камеры подключены", "архив не проверен"],
                            "equipment_consumables": "4 камеры, 30 метров кабеля",
                            "additional_info": "вернуться завтра, проверить доступ",
                            "overall_status_label": "выполнена частично",
                            "not_done_items": ["не проверен архив"],
                            "unconfirmed_items": ["нет фото результата"],
                            "status_reasons": ["не было доступа к архиву"],
                            "missing_fields": ["причина, почему архив не проверен"],
                            "ai_close_marker": "AI_SERVER_TASK_CLOSE_INCOMPLETE",
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "1. Выполняемые работы" in result.answer
    assert "1.1 камеры подключены - ... ???" in result.answer
    assert "1.3 Еще работы - ... ???" in result.answer
    assert "2. Использовано материалов, оборудование" in result.answer
    assert "2.1 4 камеры" in result.answer
    assert "2.2 30 метров кабеля" in result.answer
    assert "[ВЫБРАНО: выполнено частично]" in result.answer
    assert "3.1 причина: не было доступа к архиву" in result.answer
    assert "Причина/что не выполнено: не проверен архив" not in result.answer
    assert "1.2 архив не проверен - не выполнено" in result.answer
    assert "Не подтверждено: нет фото результата" not in result.answer
    assert "4. Дополнительная информация" in result.answer
    assert "4.1 вернуться завтра" in result.answer
    assert "4.2 проверить доступ" in result.answer
    assert "да, закрывай как есть" in result.answer
    assert "[URL" not in result.answer


def test_bitrix_llm_compose_formats_task_close_point_status_labels(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {"_draft_type": "task_close", "task_id": 139, "task_title": "Проверить объект"},
                        "preview": {
                            "task_title": "Проверить объект",
                            "completion_summary": (
                                "Проверить камеру выполнено полностью. "
                                "Забрать документы не выполнено. "
                                "Починить селектор не подтверждено"
                            ),
                            "task_points": ["Проверить камеру", "Забрать документы", "Починить селектор"],
                            "not_done_items": ["Забрать документы"],
                            "unconfirmed_items": ["Починить селектор"],
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "1.1 Проверить камеру - выполнено полностью" in result.answer
    assert "1.2 Забрать документы - не выполнено" in result.answer
    assert "1.3 Починить селектор - не подтверждено" in result.answer
    assert "1.2 Забрать документы - не выполнено: Забрать документы" not in result.answer
    assert "1.3 Починить селектор - не подтверждено: Починить селектор" not in result.answer
    assert "2.1 выполнено" not in result.answer
    assert "3.1 причина: Забрать документы" not in result.answer


def test_bitrix_llm_compose_hides_unanswered_unconfirmed_task_points(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {
                            "_draft_type": "task_close",
                            "task_id": 139,
                            "task_title": "Проверить объект",
                        },
                        "preview": {
                            "task_title": "Проверить объект",
                            "completion_summary": "Установить камеру не выполнено",
                            "task_points": ["Установить камеру", "Забрать документы", "Проверить архив"],
                            "not_done_items": ["Установить камеру"],
                            "unconfirmed_items": ["Забрать документы", "Проверить архив"],
                            "overall_status": "partial",
                            "status_reasons": ["не было доступа к архиву"],
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "1.1 Установить камеру - не выполнено" in result.answer
    assert "1.2 Забрать документы - ... ???" in result.answer
    assert "1.3 Проверить архив - ... ???" in result.answer
    assert "1.2 Забрать документы - не подтверждено" not in result.answer
    assert "1.3 Проверить архив - не подтверждено" not in result.answer


def test_bitrix_llm_compose_formats_empty_description_task_close_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {
                            "_draft_type": "task_close",
                            "task_id": 139,
                            "task_title": "Пустая задача",
                            "source_task_description_empty": True,
                        },
                        "preview": {
                            "task_title": "Пустая задача",
                            "action_label": "отметить задачу выполненной",
                            "source_task_description_empty": True,
                            "completion_summary": "Проверил объект, устранил замечания.",
                            "equipment_consumables": "не использовались",
                            "overall_status_label": "выполнена полностью",
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "1. Выполняемые работы" in result.answer
    assert "1.1 Проверил объект" in result.answer
    assert "1.2 устранил замечания" in result.answer
    assert "1. Пункты задачи" not in result.answer
    assert "[ВЫБРАНО: не использовалось]" in result.answer
    assert "2.1 не использовались" not in result.answer
    assert "[ВЫБРАНО: выполнено полностью]" in result.answer


def test_bitrix_llm_compose_hides_initial_task_close_placeholder_summary(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {
                            "_draft_type": "task_close",
                            "task_id": 139,
                            "task_title": "Пустая задача",
                            "source_task_description_empty": True,
                        },
                        "preview": {
                            "task_title": "Пустая задача",
                            "source_task_description_empty": True,
                            "completion_summary": (
                                "Общий итог: результат не подтверждён "
                                "Статус AI-закрытия: неподтвержденная "
                                "Неподтверждённые пункты: - результат выполнения не указан"
                            ),
                            "overall_status_label": "результат не подтверждён",
                            "unconfirmed_items": ["результат выполнения не указан"],
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "1. Выполняемые работы" in result.answer
    assert "1.1 Еще работы - ... ???" in result.answer
    assert "2. Использовано материалов, оборудование" in result.answer
    assert "2.1" not in result.answer
    assert "3. Статус выполнения работ" in result.answer
    assert "[ВЫБРАНО:" not in result.answer
    assert "результат не подтверждён" not in result.answer
    assert "результат выполнения не указан" not in result.answer
    assert "Статус AI-закрытия" not in result.answer


def test_bitrix_llm_compose_formats_task_close_active_draft_conflict(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу 140", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "status": "active_draft_conflict",
                        "draft": {"_draft_type": "task_close", "task_id": 139, "task_title": "Проверить камеры"},
                        "incoming_draft": {
                            "_draft_type": "task_close",
                            "task_id": 140,
                            "task_title": "Проверить регистратор",
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Уже есть активный черновик закрытия." in result.answer
    assert "Текущий: Проверить камеры" in result.answer
    assert "Новая задача: Проверить регистратор" in result.answer
    assert "1 — продолжить текущий черновик" in result.answer
    assert "3 — отменить текущий черновик и перейти к новой задаче" in result.answer


def test_bitrix_llm_compose_formats_task_close_draft_with_id_fallback(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="закрой задачу 139", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_draft",
                    data={
                        "draft": {
                            "_draft_type": "task_close",
                            "task_id": 139,
                            "task_title": "",
                            "action": "complete",
                            "completion_summary": "Готово.",
                            "unresolved_items": [],
                        },
                        "preview": {
                            "task_title": "",
                            "action_label": "отметить задачу выполненной",
                            "completion_summary": "Готово.",
                            "unresolved_items": [],
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Черновик #139: задача #139" in result.answer
    assert "Задача: указанная задача" not in result.answer


def test_bitrix_llm_compose_formats_task_close_confirm_with_task_link(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://asutp-expert.bitrix24.ru/rest/1/token/")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="да, закрывай", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="task_close_confirm",
                    data={
                        "task_id": 139,
                        "task_title": "Обучение сотрудников",
                        "action": "complete",
                        "unresolved_items": ["не приложен акт проверки"],
                        "draft": {"task_id": 139, "task_title": "Обучение сотрудников"},
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert result.answer == (
        "Задача отмечена выполненной: "
        "[URL=https://asutp-expert.bitrix24.ru/company/personal/user/0/tasks/task/view/139/]"
        "Обучение сотрудников[/URL]. Статус: неподтвержденная."
    )
    assert "/company/personal/user/13/" not in result.answer


def test_bitrix_llm_compose_formats_calendar_event_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="напомни завтра позвонить Борисову", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="calendar_event_draft",
                    data={
                        "draft": {
                            "_draft_type": "calendar_event",
                            "title": "позвонить Борисову",
                            "description": "Позвонить Борисову",
                            "start_iso": "2026-07-09T12:00:00+03:00",
                            "end_iso": "2026-07-09T12:30:00+03:00",
                        },
                        "preview": {
                            "title": "позвонить Борисову",
                            "description": "Позвонить Борисову",
                            "start": "09.07.2026 12:00 МСК",
                            "end": "09.07.2026 12:30 МСК",
                            "participants": "только текущий пользователь",
                            "reminder": "по настройкам календаря Bitrix",
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Черновик события календаря" in result.answer
    assert "09.07.2026 12:00 МСК" in result.answer
    assert "Описание: Позвонить Борисову" in result.answer
    assert "по настройкам календаря Bitrix" not in result.answer
    assert "Участники: только текущий пользователь" not in result.answer
    assert "[URL" not in result.answer


def test_bitrix_llm_compose_hides_generic_calendar_event_description(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="напомни завтра позвонить Борисову", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="calendar_event_draft",
                    data={
                        "draft": {
                            "_draft_type": "calendar_event",
                            "title": "Позвонить Борисову",
                            "description": "Напоминание",
                            "start_iso": "2026-07-09T12:00:00+03:00",
                            "end_iso": "2026-07-09T12:30:00+03:00",
                        },
                        "preview": {
                            "title": "Позвонить Борисову",
                            "description": "Напоминание",
                            "start": "09.07.2026 12:00 МСК",
                            "end": "09.07.2026 12:30 МСК",
                            "participants": "Коверга Дмитрий Владимирович",
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "needs_human"
    assert "Участники: Коверга Дмитрий Владимирович" in result.answer
    assert "Описание: Напоминание" not in result.answer


def test_bitrix_llm_compose_formats_calendar_event_confirm_without_fake_link(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="да, добавь в календарь", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="calendar_event_confirm",
                    data={
                        "event_id": "444",
                        "title": "позвонить Борисову",
                        "start_label": "09.07.2026 12:00 МСК",
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert result.answer == "Событие добавлено в календарь: позвонить Борисову. Начало: 09.07.2026 12:00 МСК."
    assert "[URL" not in result.answer


def test_bitrix_llm_compose_formats_my_tasks_with_task_links(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    service = BitrixLLMService(client=RecordingLLMClient("{}"), settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="покажи мои задачи", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_my_tasks",
                    data={
                        "status": "open",
                        "items": [
                            {
                                "id": "101",
                                "title": "Проверить камеру",
                                "deadline_label": "10.07.2026 19:00",
                                "status_label": "выполняется",
                                "roles": ["исполнитель"],
                            }
                        ],
                        "total": 1,
                        "limit": 10,
                        "offset": 0,
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/company/personal/user/0/tasks/task/view/101/]Проверить камеру[/URL]"
    ) in result.answer
    assert "срок: 10.07.2026 19:00" in result.answer
    assert "/company/personal/user/13/" not in result.answer


def test_bitrix_llm_compose_formats_task_search_comment_snippet(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    service = BitrixLLMService(client=RecordingLLMClient("{}"), settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="найди задачу с комментарием акт сверки", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_task_search",
                    data={
                        "mode": "list",
                        "scope": "my",
                        "scope_label": "мои задачи",
                        "status": "active",
                        "comment_query": "акт сверки",
                        "items": [
                            {
                                "id": "101",
                                "title": "Проверить документы",
                                "deadline_label": "10.07.2026 19:00",
                                "status_label": "выполняется",
                                "roles": ["исполнитель"],
                                "responsible_label": "Марат",
                                "creator_label": "Валерий Кулинич",
                                "comment_snippets": ["Нужен акт сверки по объекту."],
                            }
                        ],
                        "total": 1,
                        "limit": 10,
                        "offset": 0,
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert "с комментарием «акт сверки»" in result.answer
    assert "исполнитель: Марат" in result.answer
    assert "постановщик: Валерий Кулинич" in result.answer
    assert "комментарий: Нужен акт сверки по объекту." in result.answer
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/company/personal/user/0/tasks/task/view/101/]Проверить документы[/URL]"
        in result.answer
    )


def test_bitrix_llm_compose_formats_ai_close_problem_task_search(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    service = BitrixLLMService(client=RecordingLLMClient("{}"), settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="найди задачи с невыполненными пунктами", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_task_search",
                    data={
                        "mode": "list",
                        "scope": "member",
                        "scope_label": "мои задачи",
                        "status": "closed",
                        "ai_close_problem_type": "not_done",
                        "items": [
                            {
                                "id": "101",
                                "title": "Проверить камеры",
                                "deadline_label": "10.07.2026 19:00",
                                "status_label": "завершена",
                                "ai_close_incomplete": True,
                                "ai_close_problem_types": ["not_done", "unconfirmed"],
                            }
                        ],
                        "total": 1,
                        "limit": 10,
                        "offset": 0,
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert "Завершённые мои задачи с невыполненными пунктами:" in result.answer
    assert "AI-закрытие: есть невыполненные пункты, результат не подтверждён" in result.answer
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" not in result.answer


def test_bitrix_llm_compose_formats_task_search_list_without_duplicate_detail_or_ids(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    service = BitrixLLMService(client=RecordingLLMClient("{}"), settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="покажи задачи на мне", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_task_search",
                    data={
                        "mode": "list",
                        "scope": "responsible",
                        "scope_label": "задачи на мне",
                        "status": "active",
                        "items": [
                            {
                                "id": "101",
                                "title": "Обучение сотрудников",
                                "deadline_label": "10.07.2026 19:00",
                                "status_label": "выполняется",
                                "roles": ["исполнитель"],
                            }
                        ],
                        "total": 11,
                        "limit": 10,
                        "offset": 0,
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/company/personal/user/0/tasks/task/view/101/]Обучение сотрудников[/URL]"
        in result.answer
    )
    assert result.answer.count("Обучение сотрудников") == 1
    assert "Показаны первые 1 задач из 11" in result.answer
    assert "(ID:" not in result.answer
    assert "#101" not in result.answer
    assert "]([URL=" not in result.answer
    assert "/company/personal/user/13/" not in result.answer


def test_bitrix_llm_compose_formats_task_search_detail_without_visible_id(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    service = BitrixLLMService(client=RecordingLLMClient("{}"), settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="найди задачу 139", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_task_search",
                    data={
                        "mode": "detail",
                        "task_id": "139",
                        "item": {
                            "id": "139",
                            "title": "Обучение сотрудников",
                            "deadline_label": "без срока",
                            "status_label": "ждёт выполнения",
                            "roles": ["наблюдатель"],
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert "Задача найдена" in result.answer
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/company/personal/user/0/tasks/task/view/139/]Обучение сотрудников[/URL]"
        in result.answer
    )
    assert "(ID:" not in result.answer
    assert "#139" not in result.answer
    assert "наблюдатель" not in result.answer.casefold()


def test_bitrix_llm_compose_formats_project_search_with_project_link(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    service = BitrixLLMService(client=RecordingLLMClient("{}"), settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="найди проект Ларгус 2", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_project_search",
                    data={"query": "Ларгус 2", "items": [{"id": "45", "name": "Ларгус 2"}], "total": 1},
                )
            ],
            approval_actions=[],
        )
    )

    assert result.status == "completed"
    assert "[URL=https://asutp-expert.bitrix24.ru/workgroups/group/45/]Ларгус 2[/URL]" in result.answer
    assert "(ID:" not in result.answer


def test_bitrix_llm_compose_formats_portal_search_document_with_snippet(monkeypatch):
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client=client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="найди документ договор транзит", user={"id": "13"}),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="portal_search",
                    data={
                        "query": "договор транзит",
                        "scope": "documents",
                        "limit": 10,
                        "results": [
                            {
                                "entity_type": "disk_file",
                                "entity_id": "77",
                                "title": "Договор Транзит.pdf",
                                "body": (
                                    "Карточка файла.\n\nТекст файла:\n"
                                    "Это договор с компанией Транзит-Экспресс по поставке оборудования."
                                ),
                                "url": "/docs/path/Dogovor-Tranzit.pdf",
                                "score": 42,
                                "metadata": {},
                            }
                        ],
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert "Документы по запросу «договор транзит»:" in result.answer
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/docs/path/Dogovor-Tranzit.pdf]Договор Транзит.pdf[/URL]" in result.answer
    )
    assert "Фрагмент:" in result.answer
    assert "Транзит-Экспресс" in result.answer
    assert "(ID:" not in result.answer


def test_bitrix_llm_compose_formats_warehouse_products_with_links_and_more(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://asutp-expert.bitrix24.ru/rest/1/token/")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="покажи остатки по складу Борисов"),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_warehouse_search",
                    data={
                        "query": "Борисов",
                        "matches": [{"id": 10, "title": "Борисов А.А.", "address": "Российская,8"}],
                        "products": {
                            "status": "ok",
                            "items": [
                                {
                                    "product_id": 1001,
                                    "product_name": "Коробка ответвительная",
                                    "product_url": "/shop/documents-catalog/7/product/1001/",
                                    "amount": "1.0000",
                                },
                                {
                                    "product_id": 1002,
                                    "product_name": "Селектор",
                                    "product_url": "/shop/documents-catalog/7/product/1002/",
                                    "amount": "9",
                                },
                            ],
                            "limit": 2,
                            "offset": 0,
                            "available_items_with_names": 3,
                            "available_items_seen": 3,
                            "has_more": True,
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/shop/documents-catalog/7/product/1001/]Коробка ответвительная[/URL]"
        in result.answer
    )
    assert (
        "[URL=https://asutp-expert.bitrix24.ru/shop/documents-catalog/7/product/1002/]Селектор[/URL]" in result.answer
    )
    assert "Показаны первые 2 позиции из 3" in result.answer
    assert "product/1001" in result.answer
    assert "#1001" not in result.answer


def test_bitrix_llm_compose_marks_limit_sized_warehouse_page_as_complete_when_total_known(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://asutp-expert.bitrix24.ru/rest/1/token/")
    client = RecordingLLMClient('{"status":"completed","answer":"should not be used"}')
    service = BitrixLLMService(client, settings=get_settings())

    result = asyncio.run(
        service.compose(
            task=AgentTask(task_id="t1", request="покажи остатки по складу Борисов"),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="bitrix_warehouse_search",
                    data={
                        "query": "Борисов",
                        "matches": [{"id": 10, "title": "Борисов А.А.", "address": "Российская,8"}],
                        "products": {
                            "status": "ok",
                            "items": [
                                {
                                    "product_id": 1001,
                                    "product_name": "Коробка ответвительная",
                                    "product_url": "/shop/documents-catalog/7/product/1001/",
                                    "amount": "1",
                                },
                                {
                                    "product_id": 1002,
                                    "product_name": "Селектор",
                                    "product_url": "/shop/documents-catalog/7/product/1002/",
                                    "amount": "9",
                                },
                            ],
                            "limit": 2,
                            "offset": 0,
                            "available_items_with_names": 2,
                            "available_items_seen": 2,
                            "has_more": False,
                        },
                    },
                )
            ],
            approval_actions=[],
        )
    )

    assert client.calls == []
    assert result.status == "completed"
    assert "Показаны все 2 позиции с положительным остатком" in result.answer
    assert "следующие позиции" not in result.answer
    assert "Остальные позиции есть" not in result.answer


def test_bitrix_llm_service_uses_injected_settings_not_global_at_call_time(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://injected.bitrix24.ru/rest/1/token/")
    injected_settings = get_settings()

    # Change ambient env after capturing the snapshot to prove decide()/compose() use the
    # injected Settings instance, not a fresh get_settings() call made from inside them.
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://different.bitrix24.ru/rest/1/token/")

    client = RecordingLLMClient('{"status":"completed","answer":"ok","confidence":0.7,"tool_calls":[]}')
    manifest = get_agent_manifest("bitrix24")
    service = BitrixLLMService(client, settings=injected_settings)

    asyncio.run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="t1", request="x"),
            decision=BitrixLLMDecision(status="completed", answer="", tool_calls=[BitrixLLMToolCall(name="none")]),
            tool_results=[],
            approval_actions=[],
        )
    )

    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["portal_base_url"] == "https://injected.bitrix24.ru"


def test_bitrix_policy():
    assert decide_bitrix_method_policy("tasks.task.list").decision == "allow"
    assert decide_bitrix_method_policy("tasks.task.add").decision == "confirm"
    assert decide_bitrix_method_policy("user.delete").decision == "deny"


def test_bitrix_skills_and_knowledge_loaded():
    manifest = get_agent_manifest("bitrix24")

    skill_ids = {skill.id for skill in SkillStore().list_skills(manifest)}
    topic_ids = {topic.name for topic in MarkdownKnowledgeBase().list_topics(manifest)}

    assert "tasks_search" in skill_ids
    assert "safe_bitrix_write" in skill_ids
    assert "tasks_search" in topic_ids
    assert "bitrix_rest" in topic_ids
    assert "user_permissions" in topic_ids


class _FakePortalSearchTool:
    name = "portal_search"

    def __init__(self, result: ToolResult | None = None, calls: list | None = None):
        self._result = result or ToolResult(status="not_configured", tool="portal_search")
        self._calls = calls

    def definition(self):
        return ToolDefinition(name="portal_search", description="", parameters={})

    async def execute(self, args, *, user_id=None, dialog_key=None, dialog_id=None):
        if self._calls is not None:
            self._calls.append(args)
        return self._result


class _FakeBitrixApiTool:
    name = "bitrix_api"

    def __init__(self, result: ToolResult, calls: list):
        self._result = result
        self._calls = calls

    def definition(self):
        return ToolDefinition(name="bitrix_api", description="", parameters={})

    async def execute(self, args, *, user_id=None, dialog_key=None, dialog_id=None):
        self._calls.append(args)
        return self._result


class _FakeMyTasksTool:
    name = "bitrix_my_tasks"

    def __init__(self, result: ToolResult, calls: list):
        self._result = result
        self._calls = calls

    def definition(self):
        return ToolDefinition(name="bitrix_my_tasks", description="", parameters={})

    async def execute(self, args, *, user_id=None, dialog_key=None, dialog_id=None):
        self._calls.append(args)
        return self._result


class _FakeWarehouseTool:
    name = "bitrix_warehouse_search"

    def __init__(self, result: ToolResult, calls: list):
        self._result = result
        self._calls = calls

    def definition(self):
        return ToolDefinition(name="bitrix_warehouse_search", description="", parameters={})

    async def execute(self, args, *, user_id=None, dialog_key=None, dialog_id=None):
        self._calls.append(args)
        return self._result


class _FakeNamedTool:
    def __init__(self, name: str, result: ToolResult, calls: list):
        self.name = name
        self._result = result
        self._calls = calls

    def definition(self):
        return ToolDefinition(name=self.name, description="", parameters={})

    async def execute(self, args, *, user_id=None, dialog_key=None, dialog_id=None):
        self._calls.append(args)
        return self._result


class FakeResolverTools:
    def __init__(
        self,
        *,
        bitrix_api_result: ToolResult | None = None,
        my_tasks_result: ToolResult | None = None,
        portal_search_result: ToolResult | None = None,
        task_search_result: ToolResult | None = None,
        project_search_result: ToolResult | None = None,
        warehouse_result: ToolResult | None = None,
        raw_user: dict | None = None,
    ) -> None:
        self.bitrix_api_calls: list = []
        self.my_tasks_calls: list = []
        self.task_search_calls: list = []
        self.project_search_calls: list = []
        self.warehouse_calls: list = []
        self.portal_search_calls: list = []
        self._raw_user = raw_user
        self._tools = [
            _FakePortalSearchTool(portal_search_result, self.portal_search_calls),
            _FakeMyTasksTool(
                my_tasks_result or ToolResult(status="not_configured", tool="bitrix_my_tasks"),
                self.my_tasks_calls,
            ),
            _FakeWarehouseTool(
                warehouse_result or ToolResult(status="not_configured", tool="bitrix_warehouse_search"),
                self.warehouse_calls,
            ),
            _FakeNamedTool(
                "bitrix_task_search",
                task_search_result or ToolResult(status="not_configured", tool="bitrix_task_search"),
                self.task_search_calls,
            ),
            _FakeNamedTool(
                "bitrix_project_search",
                project_search_result or ToolResult(status="not_configured", tool="bitrix_project_search"),
                self.project_search_calls,
            ),
            _FakeBitrixApiTool(
                bitrix_api_result or ToolResult(status="not_configured", tool="bitrix_api"),
                self.bitrix_api_calls,
            ),
        ]

    def agent_tools(self) -> list:
        return self._tools

    def make_user_client(self):
        raw_user = self._raw_user

        class _FakeUserClient:
            async def get_user(self, user_id: int):
                return raw_user

        return _FakeUserClient()


def test_specialist_preserves_needs_clarification_when_compose_returns_completed():
    """Если decide вернул needs_clarification, а compose вернул completed — status должен остаться needs_clarification.

    Воспроизводит баг: deepseek-chat в compose-шаге часто возвращает "completed"
    даже когда специалист задаёт уточняющий вопрос. base.py должен доверять decide-статусу.
    """
    llm = FakeBitrixLLM(
        decision_status="needs_clarification",
        decision_answer="Укажите срок выполнения задачи.",
        final_status="completed",  # LLM-баг: compose говорит completed вместо needs_clarification
        final_answer="Укажите срок выполнения задачи.",
    )
    result = asyncio.run(
        _bitrix_specialist(llm=llm).handle(AgentTask(task_id="t1", request="создай задачу в тестовом проекте"))
    )

    assert result.status == "needs_clarification"


def test_specialist_uses_compose_failed_even_if_decide_was_needs_clarification():
    """Если compose вернул failed — это перебивает decide-статус needs_clarification."""
    llm = FakeBitrixLLM(
        decision_status="needs_clarification",
        final_status="failed",
        final_answer="Ошибка при обработке.",
    )
    result = asyncio.run(_bitrix_specialist(llm=llm).handle(AgentTask(task_id="t1", request="создай задачу")))

    assert result.status == "failed"
