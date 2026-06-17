import asyncio
import json

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix_llm import BitrixLLMDecision, BitrixLLMService, BitrixLLMToolCall
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentTask, ToolResult
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.skills import SkillStore
from ai_server.tools.bitrix import BitrixToolset
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, RecordingLLMClient


def _bitrix_specialist(*, tools=None, llm=None) -> Bitrix24Specialist:
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    return Bitrix24Specialist(
        manifest,
        retriever=retriever,
        tools=tools,
        llm=llm or FakeBitrixLLM(),
    )


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
        bitrix_api_result=ToolResult(
            status="ok",
            tool="bitrix_api",
            data={
                "method": "tasks.task.list",
                "params": {"filter": {"!STATUS": 5, "RESPONSIBLE_ID": 9}},
                "result": [
                    {
                        "id": "101",
                        "title": "Проверить камеру",
                        "status": "3",
                        "responsibleId": "9",
                    }
                ],
            },
        )
    )

    result = asyncio.run(
        _bitrix_specialist(
            tools=tools,
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="bitrix_api",
                        args={
                            "action": "call",
                            "method": "tasks.task.list",
                            "params": {"filter": {"!STATUS": 5, "RESPONSIBLE_ID": 9}},
                        },
                    )
                ],
                final_answer="Нашёл задачу: Проверить камеру.",
            ),
        ).handle(AgentTask(task_id="t1", request="Покажи мои открытые задачи", user={"id": "9"}))
    )

    assert result.status == "completed"
    assert tools.bitrix_api_calls[0]["method"] == "tasks.task.list"
    assert tools.bitrix_api_calls[0]["params"]["filter"] == {"!STATUS": 5, "RESPONSIBLE_ID": 9}
    assert "Проверить камеру" in result.answer


def test_bitrix_specialist_passes_user_profile_and_permission_rag_to_llm():
    llm = FakeBitrixLLM()
    tools = FakeResolverTools(
        profile_result=ToolResult(
            status="ok",
            tool="current_user_profile",
            data={
                "user_id": 9,
                "profile": {
                    "id": 9,
                    "label": "Иванов Иван",
                    "active": True,
                    "is_admin": False,
                    "department_ids": [77],
                    "work_position": "Монтажник",
                    "user_type": "employee",
                },
            },
        )
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


def test_bitrix_specialist_can_call_current_user_profile_tool():
    result = asyncio.run(
        _bitrix_specialist(
            tools=FakeResolverTools(
                profile_result=ToolResult(
                    status="ok",
                    tool="current_user_profile",
                    data={"user_id": 9, "profile": {"id": 9, "work_position": "Монтажник"}},
                )
            ),
            llm=FakeBitrixLLM(
                tool_calls=[BitrixLLMToolCall(name="current_user_profile")],
                final_answer="Профиль пользователя проверен.",
            ),
        ).handle(AgentTask(task_id="t1", request="Кто я в Битриксе?", user={"id": "9"}))
    )

    assert result.status == "completed"
    assert result.actions_taken[2].name == "bitrix_current_user_profile"
    assert result.actions_taken[2].status == "ok"


def test_bitrix_specialist_marks_write_for_approval():
    result = asyncio.run(
        _bitrix_specialist(
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
            )
        ).handle(AgentTask(task_id="t1", request="Создай задачу на меня проверить IP-камеру завтра", user={"id": "9"}))
    )

    assert result.status == "needs_human"
    assert result.actions_requiring_approval
    action = result.actions_requiring_approval[0]
    assert action.details["method"] == "tasks.task.add"
    assert action.details["params"]["fields"]["TITLE"] == "проверить IP-камеру"
    assert action.details["params"]["fields"]["RESPONSIBLE_ID"] == 9
    assert action.details["params"]["fields"]["CREATED_BY"] == 9
    assert action.details["params"]["fields"]["DEADLINE"]


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
    result = asyncio.run(
        _bitrix_specialist(
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="task_create_draft",
                        args={"title": "проверить камеру"},
                    )
                ],
                final_status="failed",
                final_answer="LLM-субагент вызвал task_create_draft с неполным контрактом.",
            )
        ).handle(AgentTask(task_id="t1", request="Создай задачу проверить камеру"))
    )

    assert result.status == "failed"
    assert result.actions_requiring_approval == []
    action = result.actions_taken[2]
    assert action.name == "bitrix_task_create_draft"
    assert action.status == "contract_violation"
    assert action.details["contract_errors"] == [
        "task_create_draft requires one of responsible_id, responsible_query, or responsible_self",
        "task_create_draft requires deadline_iso or no_deadline=true",
    ]


def test_bitrix_specialist_task_create_uses_explicit_ids():
    result = asyncio.run(
        _bitrix_specialist(
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
            )
        ).handle(
            AgentTask(
                task_id="t1",
                request="Поставь задачу ответственному #15 проверить регистратор в проекте #44 до 05.06.2026",
                user={"id": "9"},
            )
        )
    )

    assert result.status == "needs_human"
    params = result.actions_requiring_approval[0].details["params"]
    assert params["fields"]["TITLE"] == "проверить регистратор"
    assert params["fields"]["RESPONSIBLE_ID"] == 15
    assert params["fields"]["GROUP_ID"] == 44
    assert params["fields"]["DEADLINE"].startswith("2026-06-05T19:00:00")


def test_bitrix_specialist_resolves_responsible_name():
    result = asyncio.run(
        _bitrix_specialist(
            tools=FakeResolverTools(
                users={
                    "Иванову": [
                        {"id": 15, "label": "Иванов Иван"},
                    ]
                }
            ),
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="task_create_draft",
                        args={
                            "title": "проверить IP-камеру",
                            "responsible_query": "Иванову",
                            "deadline_iso": "2026-06-05T19:00:00+03:00",
                        },
                    )
                ],
                final_status="needs_human",
                final_answer="Подготовил черновик задачи, нужно подтверждение.",
            ),
        ).handle(
            AgentTask(
                task_id="t1",
                request="Создай задачу Иванову проверить IP-камеру завтра",
                user={"id": "9"},
            )
        )
    )

    assert result.status == "needs_human"
    params = result.actions_requiring_approval[0].details["params"]
    assert params["fields"]["TITLE"] == "проверить IP-камеру"
    assert params["fields"]["RESPONSIBLE_ID"] == 15


def test_bitrix_specialist_resolves_project_name():
    result = asyncio.run(
        _bitrix_specialist(
            tools=FakeResolverTools(
                projects={
                    "Монтаж": [
                        {"id": 44, "label": "Монтаж"},
                    ]
                }
            ),
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="task_create_draft",
                        args={
                            "title": "проверить камеру",
                            "responsible_self": True,
                            "project_query": "Монтаж",
                            "deadline_iso": "2026-06-05T19:00:00+03:00",
                        },
                    )
                ],
                final_status="needs_human",
                final_answer="Подготовил черновик задачи, нужно подтверждение.",
            ),
        ).handle(
            AgentTask(
                task_id="t1",
                request="Создай задачу на меня проверить камеру в проекте Монтаж завтра",
                user={"id": "9"},
            )
        )
    )

    assert result.status == "needs_human"
    params = result.actions_requiring_approval[0].details["params"]
    assert params["fields"]["TITLE"] == "проверить камеру"
    assert params["fields"]["RESPONSIBLE_ID"] == 9
    assert params["fields"]["GROUP_ID"] == 44


def test_bitrix_specialist_asks_when_responsible_name_is_ambiguous():
    result = asyncio.run(
        _bitrix_specialist(
            tools=FakeResolverTools(
                users={
                    "Иванову": [
                        {"id": 15, "label": "Иванов Иван"},
                        {"id": 16, "label": "Иванов Пётр"},
                    ]
                }
            ),
            llm=FakeBitrixLLM(
                tool_calls=[
                    BitrixLLMToolCall(
                        name="task_create_draft",
                        args={
                            "title": "проверить камеру",
                            "responsible_query": "Иванову",
                            "deadline_iso": "2026-06-05T19:00:00+03:00",
                        },
                    )
                ],
                final_status="needs_clarification",
                final_answer="Нашёл несколько сотрудников Ивановых. Уточните ответственного.",
            ),
        ).handle(
            AgentTask(
                task_id="t1",
                request="Создай задачу Иванову проверить камеру",
                user={"id": "9"},
            )
        )
    )

    assert result.status == "needs_clarification"
    assert result.actions_requiring_approval == []
    assert "несколько сотрудников" in result.answer


def test_bitrix_llm_decision_payload_includes_permission_context(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AGENT_WRITE_ALLOWED_USER_IDS", "1,9")
    monkeypatch.setenv("AGENT_LIMITED_TASK_CREATE_PROJECT_ID", "44")
    monkeypatch.setenv("AGENT_LIMITED_TASK_CREATE_USER_IDS", "15")
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
        BitrixLLMService(client).decide(
            manifest=manifest,
            task=AgentTask(task_id="t1", request="Создай задачу", user={"id": "15"}, context=context),
            retrieval_hits=[],
            tool_definitions=[],
        )
    )

    payload = json.loads(client.calls[0]["messages"][1]["content"])
    permission = payload["permission_context"]
    assert permission["current_user_id"] == 15
    assert permission["current_user_write_profile"] == "limited_task_create"
    assert permission["full_write_user_ids"] == [1, 9]
    assert permission["limited_task_create_user_ids"] == [15]
    assert permission["limited_task_create_project_id"] == 44
    assert permission["bitrix_current_user_profile"] == context["bitrix_current_user_profile"]
    assert permission["permission_policy_context"] == context["permission_policy_context"]
    assert any("read_only users should not prepare write-tools" in rule for rule in permission["rules"])


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


def test_bitrix_current_user_profile_tool_compacts_user_fields():
    class ProfileClient:
        async def get_user(self, user_id: int):
            assert user_id == 9
            return {
                "ID": "9",
                "ACTIVE": "Y",
                "NAME": "Иван",
                "LAST_NAME": "Иванов",
                "WORK_POSITION": "Монтажник",
                "UF_DEPARTMENT": ["77", "88"],
                "IS_ADMIN": "N",
                "USER_TYPE": "employee",
            }

    result = asyncio.run(BitrixToolset(client=ProfileClient(), user_id=9).current_user_profile({}))

    assert result.status == "ok"
    profile = result.data["profile"]
    assert profile["id"] == 9
    assert profile["label"] == "Иванов Иван"
    assert profile["department_ids"] == [77, 88]
    assert profile["work_position"] == "Монтажник"


class FakeResolverTools:
    def __init__(
        self,
        *,
        users: dict[str, list[dict]] | None = None,
        projects: dict[str, list[dict]] | None = None,
        bitrix_api_result: ToolResult | None = None,
        profile_result: ToolResult | None = None,
    ) -> None:
        self.users = users or {}
        self.projects = projects or {}
        self.bitrix_api_result = bitrix_api_result or ToolResult(status="not_configured", tool="bitrix_api")
        self.profile_result = profile_result or ToolResult(status="not_available", tool="current_user_profile")
        self.bitrix_api_calls: list[dict] = []

    def definitions(self) -> list:
        return []

    async def resolve_user(self, query: str, *, limit: int = 5) -> ToolResult:
        return _fake_resolution("resolve_user", query, self.users.get(query, []))

    async def resolve_project(self, query: str, *, limit: int = 5) -> ToolResult:
        return _fake_resolution("resolve_project", query, self.projects.get(query, []))

    def portal_search_contract(self, args: dict) -> ToolResult:
        return ToolResult(status="not_configured", tool="portal_search", data={"args": args})

    async def bitrix_api(self, args: dict) -> ToolResult:
        self.bitrix_api_calls.append(args)
        return self.bitrix_api_result

    async def current_user_profile(self, args: dict | None = None) -> ToolResult:
        return self.profile_result


def _fake_resolution(tool: str, query: str, candidates: list[dict]) -> ToolResult:
    if not candidates:
        return ToolResult(status="not_found", tool=tool, data={"query": query, "candidates": []})
    if len(candidates) == 1:
        return ToolResult(
            status="ok", tool=tool, data={"query": query, "candidate": candidates[0], "candidates": candidates}
        )
    return ToolResult(status="ambiguous", tool=tool, data={"query": query, "candidates": candidates})
