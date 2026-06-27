from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    INVALID_TOOL_CALL = "invalid_tool_call"
    NOT_FOUND = "not_found"
    NOT_AVAILABLE = "not_available"
    AMBIGUOUS = "ambiguous"
    DENIED = "denied"
    CONTRACT_VIOLATION = "contract_violation"
    READY = "ready"
    DRY_RUN = "dry_run"


AgentKind = Literal["orchestrator", "operator", "specialist"]
AgentAutomationKind = Literal[
    "channel_adapter",
    "event_worker",
    "scheduled_worker",
    "data_pipeline",
    "business_workflow",
]
AgentAutomationTrigger = Literal[
    "webhook",
    "queue",
    "schedule",
    "polling",
    "manual",
    "message",
]
AgentResultStatus = Literal[
    "completed",
    "needs_clarification",
    "needs_human",
    "failed",
]


class AgentAutomationManifest(BaseModel):
    id: str
    name: str
    kind: AgentAutomationKind
    trigger: AgentAutomationTrigger
    description: str
    owner_agent_id: str = ""
    version: str = "0.1.0"
    source_project: str = ""
    source_modules: list[str] = Field(default_factory=list)
    entrypoint: str | None = None
    schedule_hint: str | None = None
    status_endpoint: str | None = None
    enabled_by_default: bool = False
    uses_llm: bool = False
    requires_oauth_actor: bool = False
    human_approval_required: bool = False
    dependencies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    data_scopes: list[str] = Field(default_factory=list)
    state_paths: list[str] = Field(default_factory=list)
    emits: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AgentManifest(BaseModel):
    id: str
    name: str
    kind: AgentKind
    description: str
    version: str = "0.1.0"
    handoff_description: str = ""
    entrypoint: str | None = None
    instructions_file: str | None = None
    skills_path: str | None = None
    knowledge_path: str | None = None
    automations_path: str | None = None
    channels: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    automations: list[AgentAutomationManifest] = Field(default_factory=list)
    data_scopes: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    approval_required: list[str] = Field(default_factory=list)
    audience: str = "employee"


class AgentSummary(BaseModel):
    id: str
    name: str
    kind: AgentKind
    capabilities: list[str]
    tools: list[str]
    automations: list[str] = Field(default_factory=list)
    handoff_description: str = ""


class AutomationSummary(BaseModel):
    id: str
    name: str
    kind: AgentAutomationKind
    trigger: AgentAutomationTrigger
    owner_agent_id: str
    enabled_by_default: bool
    uses_llm: bool = False
    requires_oauth_actor: bool = False
    human_approval_required: bool = False


class UserContext(BaseModel):
    id: str | None = None
    role: str | None = None
    channel: str | None = None
    display_name: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskContext(TypedDict, total=False):
    # Маршрутизация ответа через канал
    channel_id: str
    recipient_id: str
    # Диалог
    dialog_key: str
    dialog_id: str
    dialog_history: list[dict[str, Any]]
    # Вложения и голос
    transcriptions: list[Any]
    attachment_errors: list[str]
    # Bitrix-специфика
    bitrix_event_type: str
    bitrix_current_user_profile: dict[str, Any]
    permission_policy_context: list[Any]
    # Планировщик и события
    event: str
    request_date: str
    reminder_count: int


class AgentTask(BaseModel):
    task_id: str
    source: str = "internal_orchestrator"
    user: UserContext = Field(default_factory=UserContext)
    request: str
    files: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)  # see TaskContext for documented keys
    allowed_actions: list[str] = Field(default_factory=list)
    required_output_format: str = "structured_result"


class ActionRecord(BaseModel):
    name: str
    status: str = "planned"
    details: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    type: str
    title: str
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelUsageRecord(BaseModel):
    agent_id: str
    provider: str
    model: str
    status: str = "used"
    role: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    notes: list[str] = Field(default_factory=list)


class ScheduledTask(BaseModel):
    job_id: str
    agent_id: str
    trigger: dict[str, Any] = Field(default_factory=dict)
    task: AgentTask | None = None
    description: str = ""
    cancel: bool = False


class AgentResult(BaseModel):
    status: AgentResultStatus
    agent_id: str
    answer: str
    artifacts: list[Artifact] = Field(default_factory=list)
    actions_taken: list[ActionRecord] = Field(default_factory=list)
    actions_requiring_approval: list[ActionRecord] = Field(default_factory=list)
    model_usage: list[ModelUsageRecord] = Field(default_factory=list)
    handoff_to: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    logs: list[str] = Field(default_factory=list)
    scheduled_tasks: list[ScheduledTask] = Field(default_factory=list)


class AgentTestRequest(BaseModel):
    text: str
    user_id: str | None = None
    channel: str = "local_test"
    dialog_id: str = "test"


class LearningFeedbackRequest(BaseModel):
    event_id: str
    rating: int | None = Field(default=None, ge=-1, le=1)
    corrected_answer: str = ""
    comment: str = ""
    tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    channel: str = "manual"


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class ToolResult(BaseModel):
    status: ToolStatus
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PolicyDecision(BaseModel):
    decision: Literal["allow", "confirm", "deny"]
    reason: str = ""
