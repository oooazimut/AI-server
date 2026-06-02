from typing import Literal

from pydantic import BaseModel, Field


AgentKind = Literal["orchestrator", "operator", "specialist"]


class AgentManifest(BaseModel):
    id: str
    name: str
    kind: AgentKind
    description: str
    entrypoint: str | None = None
    channels: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    data_scopes: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    approval_required: list[str] = Field(default_factory=list)


class AgentSummary(BaseModel):
    id: str
    name: str
    kind: AgentKind
    capabilities: list[str]
    tools: list[str]
