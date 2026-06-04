import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentTask, ToolResult
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from tests.fakes import FakeEmbeddingProvider


def _bitrix_specialist(*, tools=None) -> Bitrix24Specialist:
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    return Bitrix24Specialist(manifest, retriever=retriever, tools=tools)


def test_bitrix_specialist_selects_task_skill():
    result = asyncio.run(
        _bitrix_specialist(tools=FakeResolverTools()).handle(
            AgentTask(task_id="t1", request="Найди просроченные задачи в Битриксе")
        )
    )

    assert result.status == "completed"
    assert result.handoff_to == []
    assert result.actions_taken[0].details["skills"] == ["tasks_search"]


def test_bitrix_specialist_searches_task_by_id():
    result = asyncio.run(
        _bitrix_specialist(
            tools=FakeResolverTools(
                task_result=ToolResult(
                    status="ok",
                    tool="task_search",
                    data={
                        "mode": "get",
                        "task_id": 8413,
                        "task": {
                            "id": "8413",
                            "title": "AI-server private smoke test",
                            "status": "2",
                            "responsibleId": "9",
                            "deadline": None,
                        },
                    },
                )
            )
        ).handle(AgentTask(task_id="t1", request="Найди задачу #8413"))
    )

    assert result.status == "completed"
    assert result.actions_taken[1].name == "bitrix_task_search"
    assert result.actions_taken[1].status == "ok"
    assert "#8413" in result.answer
    assert "без срока" in result.answer


def test_bitrix_specialist_searches_my_open_tasks():
    tools = FakeResolverTools(
        task_result=ToolResult(
            status="ok",
            tool="task_search",
            data={
                "mode": "list",
                "tasks": [
                    {
                        "id": "101",
                        "title": "Проверить камеру",
                        "status": "3",
                        "responsibleId": "9",
                    }
                ],
                "total": 1,
            },
        )
    )

    result = asyncio.run(
        _bitrix_specialist(tools=tools).handle(
            AgentTask(task_id="t1", request="Покажи мои открытые задачи", user={"id": "9"})
        )
    )

    assert result.status == "completed"
    assert tools.task_search_calls[0]["mode"] == "list"
    assert tools.task_search_calls[0]["filter"] == {"!STATUS": 5, "RESPONSIBLE_ID": 9}
    assert "Проверить камеру" in result.answer


def test_bitrix_specialist_marks_write_for_approval():
    result = asyncio.run(
        _bitrix_specialist().handle(
            AgentTask(task_id="t1", request="Создай задачу на меня проверить IP-камеру завтра", user={"id": "9"})
        )
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
        _bitrix_specialist().handle(
            AgentTask(task_id="t1", request="Создай задачу проверить IP-камеру")
        )
    )

    assert result.status == "needs_clarification"
    assert result.actions_requiring_approval == []
    assert "ответственный" in result.answer


def test_bitrix_specialist_task_create_uses_explicit_ids():
    result = asyncio.run(
        _bitrix_specialist().handle(
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
            )
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
            )
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
            )
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


class FakeResolverTools:
    def __init__(
        self,
        *,
        users: dict[str, list[dict]] | None = None,
        projects: dict[str, list[dict]] | None = None,
        task_result: ToolResult | None = None,
    ) -> None:
        self.users = users or {}
        self.projects = projects or {}
        self.task_result = task_result or ToolResult(status="not_configured", tool="task_search")
        self.task_search_calls: list[dict] = []

    def definitions(self) -> list:
        return []

    async def resolve_user(self, query: str, *, limit: int = 5) -> ToolResult:
        return _fake_resolution("resolve_user", query, self.users.get(query, []))

    async def resolve_project(self, query: str, *, limit: int = 5) -> ToolResult:
        return _fake_resolution("resolve_project", query, self.projects.get(query, []))

    def portal_search_contract(self, args: dict) -> ToolResult:
        return ToolResult(status="not_configured", tool="portal_search", data={"args": args})

    async def task_search(self, args: dict) -> ToolResult:
        self.task_search_calls.append(args)
        return self.task_result


def _fake_resolution(tool: str, query: str, candidates: list[dict]) -> ToolResult:
    if not candidates:
        return ToolResult(status="not_found", tool=tool, data={"query": query, "candidates": []})
    if len(candidates) == 1:
        return ToolResult(status="ok", tool=tool, data={"query": query, "candidate": candidates[0], "candidates": candidates})
    return ToolResult(status="ambiguous", tool=tool, data={"query": query, "candidates": candidates})

