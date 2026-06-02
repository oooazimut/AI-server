from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query

from .knowledge import MarkdownKnowledgeBase
from .models import AgentTask, AgentTestRequest, UserContext
from .retrieval import HybridKnowledgeRetriever
from .orchestrator import suggest_agents
from .orchestrators.internal import InternalOrchestrator
from .registry import get_agent_manifest, load_agent_manifests, summarize_agents
from .skills import SkillStore


app = FastAPI(title="AI Server", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    manifests = load_agent_manifests()
    return {
        "status": "ok",
        "architecture": "orchestrator_plus_modular_specialists",
        "agent_count": len(manifests),
        "agents": [agent.id for agent in manifests],
    }


@app.get("/agents")
def agents():
    manifests = load_agent_manifests()
    return summarize_agents(manifests)


@app.get("/agents/{agent_id}")
def agent_detail(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return manifest


@app.get("/agents/{agent_id}/skills")
def agent_skills(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return SkillStore().list_skills(manifest)


@app.get("/agents/{agent_id}/knowledge/topics")
def agent_knowledge_topics(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return MarkdownKnowledgeBase().list_topics(manifest)


@app.get("/agents/{agent_id}/knowledge/search")
def agent_knowledge_search(
    agent_id: str,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
    topic: str | None = None,
):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return HybridKnowledgeRetriever().search(manifest, q, limit=limit, topic=topic)


@app.get("/route-preview")
def route_preview(q: str = Query(..., min_length=1)):
    manifests = load_agent_manifests()
    matches = suggest_agents(q, manifests)
    return summarize_agents(matches)


@app.post("/orchestrator/test")
async def orchestrator_test(body: AgentTestRequest):
    manifests = load_agent_manifests()
    task = AgentTask(
        task_id=str(uuid4()),
        source="local_test",
        user=UserContext(id=body.user_id, channel=body.channel, raw={"dialog_id": body.dialog_id}),
        request=body.text,
    )
    return await InternalOrchestrator(manifests).handle(task)

