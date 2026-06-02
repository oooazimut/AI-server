from fastapi import FastAPI, Query

from .orchestrator import suggest_agents
from .registry import load_agent_manifests, summarize_agents


app = FastAPI(title="AI Server", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents")
def agents():
    manifests = load_agent_manifests()
    return summarize_agents(manifests)


@app.get("/route-preview")
def route_preview(q: str = Query(..., min_length=1)):
    manifests = load_agent_manifests()
    matches = suggest_agents(q, manifests)
    return summarize_agents(matches)
