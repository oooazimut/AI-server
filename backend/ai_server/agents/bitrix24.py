from __future__ import annotations

from ai_server.agents.bitrix_task_create import (
    BitrixTaskCreateDraft,
    BitrixTaskCreateResolution,
    build_task_create_draft_from_args,
)
from ai_server.agents.bitrix_llm import (
    BitrixAgentLLM,
    BitrixLLMService,
    BitrixLLMToolCall,
    llm_failure_result,
)
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.bitrix import BitrixToolset
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy


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

    async def handle(self, task: AgentTask) -> AgentResult:
        text = task.request.casefold()
        selected_skills = self._select_skills(text)
        selected_topics = self._select_knowledge_topics(text)
        retrieval_hits = self.retriever.search(self.manifest, task.request, limit=3)
        actions_taken = [
            ActionRecord(
                name="load_bitrix_specialist_context",
                status="completed",
                details={
                    "skills": selected_skills,
                    "knowledge_topics": selected_topics,
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
            failure = llm_failure_result(f"{type(exc).__name__}: {exc}")
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
                        {"name": call.name, "args": call.args, "summary": call.summary}
                        for call in decision.tool_calls
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
                task=task,
                decision=decision,
                tool_results=tool_results,
                approval_actions=[action.model_dump() for action in approval_actions],
            )
        except Exception as exc:
            failure = llm_failure_result(f"{type(exc).__name__}: {exc}")
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
                    "The backend validates fields, resolves Bitrix IDs, applies policies, and requires confirmation."
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
                },
            },
        ]

    async def _task_create_draft(self, task: AgentTask, args: dict) -> BitrixTaskCreateDraft:
        draft = build_task_create_draft_from_args(task, args)
        resolution = await self._resolve_task_create_draft(draft)
        if resolution is None:
            return draft
        return build_task_create_draft_from_args(task, args, resolution=resolution)

    async def _execute_tool_call(
        self,
        tool_call: BitrixLLMToolCall,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name == "none":
            return None, None, []
        if tool_call.name == "task_search":
            result = await self.tools.task_search(tool_call.args)
            return (
                result,
                ActionRecord(name="bitrix_task_search", status=result.status, details=result.model_dump()),
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
            return result, action, _approval_actions_from_task_create_draft(draft)
        result = ToolResult(
            status="invalid_tool_call",
            tool=tool_call.name,
            error=f"unknown Bitrix LLM tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []

    async def _resolve_task_create_draft(
        self,
        draft: BitrixTaskCreateDraft,
    ) -> BitrixTaskCreateResolution | None:
        responsible_id = None
        responsible_note = ""
        group_id = None
        group_note = ""
        notes: list[str] = []

        if "responsible_id" in draft.missing_fields and draft.responsible_query:
            result = await self.tools.resolve_user(draft.responsible_query)
            if result.status == "ok":
                candidate = result.data.get("candidate") if isinstance(result.data, dict) else None
                if isinstance(candidate, dict):
                    responsible_id = _optional_int(candidate.get("id"))
                    responsible_note = f"Ответственный найден в Bitrix: {candidate.get('label') or draft.responsible_query}."
            elif result.status == "ambiguous":
                notes.append(_format_ambiguous_note("Найдено несколько сотрудников", result.data.get("candidates")))
            elif result.status == "not_found":
                notes.append(f"Не нашёл сотрудника в Bitrix по запросу `{draft.responsible_query}`.")
            else:
                notes.append(f"Не смог проверить сотрудника через Bitrix: {result.status}.")

        if "group_id" in draft.missing_fields and draft.project_query:
            result = await self.tools.resolve_project(draft.project_query)
            if result.status == "ok":
                candidate = result.data.get("candidate") if isinstance(result.data, dict) else None
                if isinstance(candidate, dict):
                    group_id = _optional_int(candidate.get("id"))
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

    def _select_skills(self, text: str) -> list[str]:
        selected: list[str] = []
        if any(marker in text for marker in ("задач", "заявк", "исполнитель", "срок", "дедлайн")):
            selected.append("tasks_search")
        if any(marker in text for marker in ("создай", "создать", "измени", "изменить", "поставь задачу")):
            selected.append("tasks_create_edit")
        if any(marker in text for marker in ("документ", "файл", "смет", "диск", "вложен", "договор", "портал")):
            selected.append("portal_document_search")
        if any(marker in text for marker in ("crm", "сделк", "лид", "проект", "групп")):
            selected.append("projects_crm")
        if any(marker in text for marker in ("удали", "закрой", "заверши", "обнови", "поменяй", "добавь")):
            selected.append("safe_bitrix_write")
        return _unique(selected)

    def _select_knowledge_topics(self, text: str) -> list[str]:
        topics: list[str] = []
        if any(marker in text for marker in ("задач", "заявк", "срок", "исполнитель")):
            topics.append("tasks_search")
        if any(marker in text for marker in ("создай", "создать", "измени", "изменить")):
            topics.append("tasks_create_edit")
        if any(marker in text for marker in ("контроль", "результат", "закрой", "заверши")):
            topics.append("tasks_quality")
        if any(marker in text for marker in ("документ", "файл", "смет", "диск", "договор", "портал")):
            topics.append("documents")
        if "bitrix" in text or "битрикс" in text or "rest" in text:
            topics.append("bitrix_rest")
        if any(marker in text for marker in ("crm", "сделк", "лид", "проект", "групп")):
            topics.append("projects_crm")
        return _unique(topics)


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


def _tool_result_from_task_create_draft(draft: BitrixTaskCreateDraft) -> ToolResult:
    return ToolResult(
        status="ready" if draft.is_ready else "contract_violation",
        tool="task_create_draft",
        data=draft.as_action_details(),
        error=None if draft.is_ready else "LLM called task_create_draft without enough validated fields.",
    )


def _approval_actions_from_task_create_draft(draft: BitrixTaskCreateDraft) -> list[ActionRecord]:
    if not draft.is_ready:
        return []
    decision = decide_bitrix_method_policy(draft.method)
    return [
        ActionRecord(
            name="bitrix_api",
            status="approval_required" if decision.decision == "confirm" else decision.decision,
            details={
                "method": draft.method,
                "params": draft.params,
                "policy": decision.model_dump(),
                "summary": draft.summary,
            },
        )
    ]


def _logs() -> list[str]:
    return [
        "Bitrix24 specialist is an LLM subagent; backend tools only execute and validate selected tool calls.",
        "Knowledge context is selected through hybrid retrieval over the agent package.",
    ]


def _optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


