import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from tests.fakes import FakeEmbeddingProvider


def _bitrix_specialist() -> Bitrix24Specialist:
    manifest = get_agent_manifest("bitrix24")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    return Bitrix24Specialist(manifest, retriever=retriever)


def test_bitrix_specialist_selects_task_skill():
    result = asyncio.run(
        _bitrix_specialist().handle(
            AgentTask(task_id="t1", request="Найди просроченные задачи в Битриксе")
        )
    )

    assert result.status == "completed"
    assert result.handoff_to == []
    assert result.actions_taken[0].details["skills"] == ["tasks_search"]


def test_bitrix_specialist_marks_write_for_approval():
    result = asyncio.run(
        _bitrix_specialist().handle(
            AgentTask(task_id="t1", request="Создай задачу в Битриксе")
        )
    )

    assert result.status == "needs_human"
    assert result.actions_requiring_approval
    assert result.actions_requiring_approval[0].details["method"] == "tasks.task.add"


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

