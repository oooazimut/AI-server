import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever


def test_hybrid_retrieval_finds_bitrix_knowledge():
    manifest = get_agent_manifest("bitrix24")

    hits = HybridKnowledgeRetriever().search(
        manifest,
        "tasks.task.add TITLE RESPONSIBLE_ID создание задачи",
        limit=5,
    )

    assert hits
    assert hits[0].score > 0
    assert {hit.chunk.topic for hit in hits} & {"bitrix_rest", "tasks_create_edit"}


def test_bitrix_specialist_includes_retrieval_hits():
    manifest = get_agent_manifest("bitrix24")
    result = asyncio.run(
        Bitrix24Specialist(manifest).handle(
            AgentTask(task_id="t1", request="Как создать задачу в Битриксе с ответственным?")
        )
    )

    retrieval_hits = result.actions_taken[0].details["retrieval_hits"]
    assert retrieval_hits
    assert retrieval_hits[0]["score"] > 0
