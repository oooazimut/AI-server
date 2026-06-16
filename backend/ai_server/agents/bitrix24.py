from __future__ import annotations

from ai_server.agents.bitrix_llm import (
    BitrixAgentLLM,
    BitrixLLMService,
    BitrixLLMToolCall,
    llm_failure_result,
)
from ai_server.agents.bitrix_task_closure import (
    TASK_CLOSURE_PENDING_METHOD,
    BitrixTaskClosureDraft,
    build_task_closure_draft_from_args,
)
from ai_server.agents.bitrix_task_create import (
    BitrixTaskCreateDraft,
    BitrixTaskCreateResolution,
    build_task_create_draft_from_args,
)
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.bitrix import BitrixToolset
from ai_server.utils import optional_int, unique


class Bitrix24Specialist:
    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        tools: BitrixToolset | None = None,
        llm: BitrixAgentLLM | None = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base or MarkdownKnowledgeBase()
        self.skill_store = skill_store or SkillStore()
        self.retriever = retriever or HybridKnowledgeRetriever(knowledge_base=self.knowledge_base)
        self.tools = tools or BitrixToolset()
        self.llm = llm or BitrixLLMService()

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        bitrix_tools: BitrixToolset | None = None,
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_llm: BitrixAgentLLM | None = None,
        **_: object,
    ) -> Bitrix24Specialist:
        return cls(manifest, retriever=bitrix_retriever, tools=bitrix_tools, llm=bitrix_llm)

    async def handle(self, task: AgentTask) -> AgentResult:
        available_skills = self.skill_store.list_skills(self.manifest)
        permission_context = await self._load_permission_context(task)
        task = task.model_copy(update={"context": {**task.context, **permission_context}})
        retrieval_hits = self.retriever.search(self.manifest, task.request, limit=5)
        actions_taken = [
            ActionRecord(
                name="load_bitrix_specialist_context",
                status="completed",
                details={
                    "available_skills": [
                        {"id": skill.id, "title": skill.title, "preview": skill.preview} for skill in available_skills
                    ],
                    "retrieval_topics": unique([hit.chunk.topic for hit in retrieval_hits]),
                    "retrieval_hits": [
                        {
                            "topic": hit.chunk.topic,
                            "section": hit.chunk.section,
                            "score": hit.score,
                            "keyword_score": hit.keyword_score,
                            "vector_score": hit.vector_score,
                            "embedding_provider": hit.embedding_provider,
                        }
                        for hit in retrieval_hits
                    ],
                    "bitrix_current_user_profile_status": _tool_context_status(
                        permission_context.get("bitrix_current_user_profile")
                    ),
                    "permission_policy_topics": [
                        item.get("topic")
                        for item in permission_context.get("permission_policy_context", [])
                        if isinstance(item, dict)
                    ],
                },
            )
        ]

        try:
            decision_result = await self.llm.decide(
                manifest=self.manifest,
                task=task,
                retrieval_hits=retrieval_hits,
                tool_definitions=self.tool_definitions(),
            )
        except Exception as exc:
            failure = llm_failure_result(f"{type(exc).__name__}: {exc}", agent_id=self.manifest.id)
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name="bitrix_llm_decision",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                model_usage=[failure.model_usage],
                confidence=0.0,
                logs=_logs(),
            )

        decision = decision_result.decision
        actions_taken.append(
            ActionRecord(
                name="bitrix_llm_decision",
                status=decision.status,
                details={
                    "tool_calls": [
                        {"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls
                    ],
                    "confidence": decision.confidence,
                },
            )
        )

        tool_results: list[ToolResult] = []
        approval_actions: list[ActionRecord] = []
        for tool_call in decision.tool_calls:
            result, action, approvals = await self._execute_tool_call(tool_call, task)
            if result is not None:
                tool_results.append(result)
            if action is not None:
                actions_taken.append(action)
            approval_actions.extend(approvals)

        try:
            final_result = await self.llm.compose(
                manifest=self.manifest,
                task=task,
                decision=decision,
                tool_results=tool_results,
                approval_actions=[action.model_dump() for action in approval_actions],
            )
        except Exception as exc:
            failure = llm_failure_result(f"{type(exc).__name__}: {exc}", agent_id=self.manifest.id)
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name="bitrix_llm_final_answer",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                actions_requiring_approval=approval_actions,
                model_usage=[decision_result.model_usage, failure.model_usage],
                confidence=0.0,
                logs=_logs(),
            )

        actions_taken.append(
            ActionRecord(
                name="bitrix_llm_final_answer",
                status=final_result.status,
                details={},
            )
        )
        status = "needs_human" if approval_actions else final_result.status

        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=final_result.answer,
            actions_taken=actions_taken,
            actions_requiring_approval=approval_actions,
            model_usage=[decision_result.model_usage, final_result.model_usage],
            confidence=decision.confidence,
            logs=_logs(),
        )

    def tool_definitions(self) -> list[dict]:
        return [
            *[definition.model_dump() for definition in self.tools.definitions()],
            {
                "name": "task_create_draft",
                "description": (
                    "Prepare a Bitrix task creation draft from fields already understood by the LLM. "
                    "The LLM must call this only after it has checked that the contract is complete. "
                    "The backend only rejects malformed calls, resolves explicit Bitrix lookup args, applies policy, "
                    "and requires confirmation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "responsible_id": {"type": "integer"},
                        "responsible_query": {"type": "string"},
                        "responsible_self": {"type": "boolean"},
                        "group_id": {"type": "integer"},
                        "project_query": {"type": "string"},
                        "deadline_iso": {"type": "string"},
                        "no_deadline": {"type": "boolean"},
                    },
                    "required": ["title"],
                    "allOf": [
                        {
                            "anyOf": [
                                {"required": ["responsible_id"]},
                                {"required": ["responsible_query"]},
                                {"required": ["responsible_self"]},
                            ]
                        },
                        {
                            "anyOf": [
                                {"required": ["deadline_iso"]},
                                {"required": ["no_deadline"]},
                            ]
                        },
                    ],
                },
            },
            {
                "name": "task_closure",
                "description": (
                    "Prepare a Bitrix task closure action from fields already understood by the LLM. "
                    "The LLM must call this only after it has checked that result_text and task_id/task_query are present. "
                    "Execution is delayed until the chat user confirms the pending write action."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "task_query": {"type": "string"},
                        "result_text": {"type": "string"},
                    },
                    "required": ["result_text"],
                    "anyOf": [
                        {"required": ["task_id"]},
                        {"required": ["task_query"]},
                    ],
                },
            },
        ]

    async def _task_create_draft(self, task: AgentTask, args: dict) -> BitrixTaskCreateDraft:
        draft = build_task_create_draft_from_args(task, args)
        resolution = await self._resolve_task_create_draft(draft)
        if resolution is None:
            return draft
        return build_task_create_draft_from_args(task, args, resolution=resolution)

    async def _task_closure_draft(self, task: AgentTask, args: dict) -> BitrixTaskClosureDraft:
        return build_task_closure_draft_from_args(task, args)

    async def _execute_tool_call(
        self,
        tool_call: BitrixLLMToolCall,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name == "none":
            return None, None, []
        if tool_call.name == "current_user_profile":
            profile_tool = getattr(self.tools, "current_user_profile", None)
            if profile_tool is None:
                result = ToolResult(
                    status="not_configured",
                    tool="current_user_profile",
                    error="current_user_profile tool is not bound",
                )
            else:
                result = await profile_tool(tool_call.args)
            return (
                result,
                ActionRecord(name="bitrix_current_user_profile", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "portal_search":
            result = self.tools.portal_search_contract(tool_call.args)
            return (
                result,
                ActionRecord(name="portal_search", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "task_create_draft":
            draft = await self._task_create_draft(task, tool_call.args)
            result = _tool_result_from_task_create_draft(draft)
            action = ActionRecord(
                name="bitrix_task_create_draft",
                status=result.status,
                details=draft.as_action_details(),
            )
            return result, action, _approval_actions_from_task_create_draft(draft, self.manifest)
        if tool_call.name == "task_closure":
            draft = await self._task_closure_draft(task, tool_call.args)
            result = _tool_result_from_task_closure_draft(draft)
            action = ActionRecord(
                name="bitrix_task_closure_draft",
                status=result.status,
                details=draft.as_action_details(),
            )
            return result, action, _approval_actions_from_task_closure_draft(draft, self.manifest)
        if tool_call.name == "bitrix_api":
            result = await self.tools.bitrix_api(tool_call.args)
            return (
                result,
                ActionRecord(name="bitrix_api", status=result.status, details=result.model_dump()),
                [],
            )
        result = ToolResult(
            status="invalid_tool_call",
            tool=tool_call.name,
            error=f"unknown Bitrix LLM tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []

    async def _load_permission_context(self, task: AgentTask) -> dict:
        profile_result = await self._current_user_profile_result(task)
        policy_hits = self.retriever.search(
            self.manifest,
            _permission_policy_query(task.request, profile_result),
            limit=3,
            topic="user_permissions",
        )
        return {
            "bitrix_current_user_profile": profile_result.model_dump(),
            "permission_policy_context": [
                {
                    "topic": hit.chunk.topic,
                    "section": hit.chunk.section,
                    "score": hit.score,
                    "text": hit.chunk.text[:1200],
                }
                for hit in policy_hits
            ],
        }

    async def _current_user_profile_result(self, task: AgentTask) -> ToolResult:
        user_id = optional_int(task.user.id)
        tool = getattr(self.tools, "current_user_profile", None)
        if user_id is None or tool is None:
            return ToolResult(
                status="not_available",
                tool="current_user_profile",
                error="current Bitrix user id is not available",
            )
        try:
            return await tool({"user_id": user_id})
        except Exception as exc:
            return ToolResult(
                status="error",
                tool="current_user_profile",
                error=f"{type(exc).__name__}: {exc}",
                data={"user_id": user_id},
            )

    async def _resolve_task_create_draft(
        self,
        draft: BitrixTaskCreateDraft,
    ) -> BitrixTaskCreateResolution | None:
        responsible_id = None
        responsible_note = ""
        group_id = None
        group_note = ""
        notes: list[str] = []

        fields = _draft_fields(draft)
        if draft.responsible_query and "RESPONSIBLE_ID" not in fields:
            result = await self.tools.resolve_user(draft.responsible_query)
            if result.status == "ok":
                candidate = result.data.get("candidate") if isinstance(result.data, dict) else None
                if isinstance(candidate, dict):
                    responsible_id = optional_int(candidate.get("id"))
                    responsible_note = (
                        f"Ответственный найден в Bitrix: {candidate.get('label') or draft.responsible_query}."
                    )
            elif result.status == "ambiguous":
                notes.append(_format_ambiguous_note("Найдено несколько сотрудников", result.data.get("candidates")))
            elif result.status == "not_found":
                notes.append(f"Не нашёл сотрудника в Bitrix по запросу `{draft.responsible_query}`.")
            else:
                notes.append(f"Не смог проверить сотрудника через Bitrix: {result.status}.")

        if draft.project_query and "GROUP_ID" not in fields:
            result = await self.tools.resolve_project(draft.project_query)
            if result.status == "ok":
                candidate = result.data.get("candidate") if isinstance(result.data, dict) else None
                if isinstance(candidate, dict):
                    group_id = optional_int(candidate.get("id"))
                    group_note = f"Проект найден в Bitrix: {candidate.get('label') or draft.project_query}."
            elif result.status == "ambiguous":
                notes.append(_format_ambiguous_note("Найдено несколько проектов", result.data.get("candidates")))
            elif result.status == "not_found":
                notes.append(f"Не нашёл проект в Bitrix по запросу `{draft.project_query}`.")
            else:
                notes.append(f"Не смог проверить проект через Bitrix: {result.status}.")

        if responsible_id is None and group_id is None and not notes:
            return None
        return BitrixTaskCreateResolution(
            responsible_id=responsible_id,
            responsible_note=responsible_note,
            group_id=group_id,
            group_note=group_note,
            notes=notes,
        )


def _format_ambiguous_note(prefix: str, candidates: object) -> str:
    if not isinstance(candidates, list):
        return prefix + "; нужно уточнение."
    labels = []
    for candidate in candidates[:5]:
        if isinstance(candidate, dict):
            label = candidate.get("label") or candidate.get("id")
            if label:
                labels.append(str(label))
    if not labels:
        return prefix + "; нужно уточнение."
    return prefix + ": " + "; ".join(labels) + ". Уточни нужный вариант."


def _permission_policy_query(request: str, profile_result: ToolResult) -> str:
    parts = [request, "права разрешения роли администратор монтажники отдел создание закрытие задачи"]
    profile = profile_result.data.get("profile") if isinstance(profile_result.data, dict) else None
    if isinstance(profile, dict):
        for key in ("work_position", "user_type", "label"):
            value = str(profile.get(key) or "").strip()
            if value:
                parts.append(value)
        department_ids = profile.get("department_ids")
        if isinstance(department_ids, list):
            parts.extend(f"department_id:{item}" for item in department_ids)
        if profile.get("is_admin") is True:
            parts.append("администратор admin")
    return " ".join(parts)


def _tool_context_status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status") or "")
    return ""


def _tool_result_from_task_create_draft(draft: BitrixTaskCreateDraft) -> ToolResult:
    return ToolResult(
        status="ready" if draft.is_ready else "contract_violation",
        tool="task_create_draft",
        data=draft.as_action_details(),
        error=None if draft.is_ready else "LLM called task_create_draft with arguments outside the tool contract.",
    )


def _approval_actions_from_task_create_draft(
    draft: BitrixTaskCreateDraft, manifest: AgentManifest
) -> list[ActionRecord]:
    if not draft.is_ready:
        return []
    needs_approval = "bitrix_write" in manifest.approval_required
    return [
        ActionRecord(
            name="bitrix_api",
            status="approval_required" if needs_approval else "allow",
            details={
                "method": draft.method,
                "params": draft.params,
                "policy": {
                    "decision": "confirm" if needs_approval else "allow",
                    "reason": "manifest.approval_required",
                },
                "summary": draft.summary,
                "specialist_id": manifest.id,
            },
        )
    ]


def _tool_result_from_task_closure_draft(draft: BitrixTaskClosureDraft) -> ToolResult:
    return ToolResult(
        status="ready" if draft.is_ready else "contract_violation",
        tool="task_closure",
        data=draft.as_action_details(),
        error=None if draft.is_ready else "LLM called task_closure with arguments outside the tool contract.",
    )


def _approval_actions_from_task_closure_draft(
    draft: BitrixTaskClosureDraft, manifest: AgentManifest
) -> list[ActionRecord]:
    if not draft.is_ready:
        return []
    needs_approval = "close_task" in manifest.approval_required
    return [
        ActionRecord(
            name="bitrix_task_closure",
            status="approval_required" if needs_approval else "allow",
            details={
                "method": TASK_CLOSURE_PENDING_METHOD,
                "params": draft.params,
                "policy": {
                    "decision": "confirm" if needs_approval else "allow",
                    "reason": "manifest.approval_required",
                },
                "summary": draft.summary,
                "specialist_id": manifest.id,
            },
        )
    ]


def _logs() -> list[str]:
    return [
        "Bitrix24 specialist is an LLM subagent; backend tools only execute and validate selected tool calls.",
        "Knowledge context is selected through hybrid retrieval over the agent package.",
    ]


def _draft_fields(draft: BitrixTaskCreateDraft) -> dict:
    fields = draft.params.get("fields")
    return fields if isinstance(fields, dict) else {}
