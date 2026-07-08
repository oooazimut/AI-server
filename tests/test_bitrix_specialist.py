import asyncio
import json

from ai_server.agents.bitrix24 import Bitrix24Specialist, BitrixLLMDecision, BitrixLLMService, BitrixLLMToolCall
from ai_server.agents.bitrix24.llm import ALLOWED_TOOL_NAMES
from ai_server.agents.bitrix24.tools.task_create import TaskCreateDraftTool
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentTask, ToolDefinition, ToolResult
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.skills import SkillStore
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, FakeTaskDraftStore, RecordingLLMClient


def _bitrix_specialist(*, tools=None, llm=None) -> Bitrix24Specialist:
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    return Bitrix24Specialist(
        manifest,
        retriever=retriever,
        skill_store=SkillStore(),
        agent_tools=tools.agent_tools() if tools is not None else None,
        bitrix_user_client=tools.make_user_client() if tools is not None else None,
        llm=llm or FakeBitrixLLM(),
    )


def _bitrix_read_tool_definitions() -> list[dict]:
    return [
        ToolDefinition(name=name, description="", parameters={}).model_dump()
        for name in ("bitrix_my_tasks", "bitrix_task_search", "bitrix_project_search", "bitrix_api", "none")
    ]


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

    specialist = Bitrix24Specialist(
        get_agent_manifest("bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        agent_tools=[TaskCreateDraftTool(store=store)],
        bitrix_user_client=_FakeUserClient(),
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
    assert preview["responsible"] == "Кулинич Дмитрий"
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


def test_bitrix_llm_exposes_and_accepts_task_draft_confirmation_tools(monkeypatch):
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
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    prompt = client.calls[0]["messages"][0]["content"]
    assert {"task_create_confirm", "task_draft_discard"} <= {tool["name"] for tool in payload["tools"]}
    assert payload["permission_context"]["current_dialog_id"] == "chat4321"
    assert payload["permission_context"]["pending_task_draft"]["fields"]["TITLE"] == "test"
    assert "task_create_confirm" in prompt
    assert "task_draft_discard" in prompt
    assert [call.name for call in result.decision.tool_calls] == ["task_create_confirm", "task_draft_discard"]


def test_bitrix_llm_exposes_and_accepts_task_close_confirmation_tools(monkeypatch):
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
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    prompt = client.calls[0]["messages"][0]["content"]
    assert {"task_close_confirm", "task_close_discard"} <= {tool["name"] for tool in payload["tools"]}
    assert payload["permission_context"]["pending_task_draft"]["_draft_type"] == "task_close"
    assert "task_close_confirm" in prompt
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" in prompt
    assert [call.name for call in result.decision.tool_calls] == ["task_close_confirm", "task_close_discard"]


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
    assert "Краткое содержание: тест подтверждения" in result.answer
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
    assert "Черновик закрытия задачи" in result.answer
    assert "Обучение сотрудников" in result.answer
    assert "не приложен акт проверки" in result.answer
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" in result.answer
    assert "[URL" not in result.answer


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
    assert "Задача: задача #139" in result.answer
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
        "Обучение сотрудников[/URL]. С непроверенными пунктами добавлена метка "
        "AI_SERVER_TASK_CLOSE_INCOMPLETE."
    )
    assert "/company/personal/user/13/" not in result.answer


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

    def definition(self):
        return ToolDefinition(name="portal_search", description="", parameters={})

    async def execute(self, args, *, user_id=None, dialog_key=None, dialog_id=None):
        return ToolResult(status="not_configured", tool="portal_search", data={"args": args})


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


class FakeResolverTools:
    def __init__(
        self,
        *,
        bitrix_api_result: ToolResult | None = None,
        my_tasks_result: ToolResult | None = None,
        raw_user: dict | None = None,
    ) -> None:
        self.bitrix_api_calls: list = []
        self.my_tasks_calls: list = []
        self._raw_user = raw_user
        self._tools = [
            _FakePortalSearchTool(),
            _FakeMyTasksTool(
                my_tasks_result or ToolResult(status="not_configured", tool="bitrix_my_tasks"),
                self.my_tasks_calls,
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
