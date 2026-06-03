from __future__ import annotations

from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask
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
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base or MarkdownKnowledgeBase()
        self.skill_store = skill_store or SkillStore()
        self.retriever = retriever or HybridKnowledgeRetriever(knowledge_base=self.knowledge_base)
        self.tools = tools or BitrixToolset()

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
        portal_search_action = self._portal_search_action(task.request, selected_skills)
        if portal_search_action is not None:
            actions_taken.append(portal_search_action)

        approval_actions = self._approval_actions(text)
        status = "needs_human" if approval_actions else "completed"
        answer = self._answer(
            selected_skills,
            selected_topics,
            retrieval_hits,
            bool(approval_actions),
            portal_search_action=portal_search_action,
        )

        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=answer,
            actions_taken=actions_taken,
            actions_requiring_approval=approval_actions,
            confidence=0.74 if selected_skills else 0.48,
            logs=[
                "Bitrix24 specialist was separated from the old autonomous BitrixAIAgent runtime.",
                "Knowledge context is selected through hybrid retrieval over the agent package.",
            ],
        )

    def tool_definitions(self) -> list[dict]:
        return [definition.model_dump() for definition in self.tools.definitions()]

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

    def _approval_actions(self, text: str) -> list[ActionRecord]:
        if not any(marker in text for marker in ("создай", "создать", "измени", "изменить", "закрой", "удали", "добавь")):
            return []

        method = "tasks.task.add" if "задач" in text or "заявк" in text else "bitrix.write"
        decision = decide_bitrix_method_policy(method)
        return [
            ActionRecord(
                name="bitrix_api",
                status="approval_required" if decision.decision == "confirm" else decision.decision,
                details={
                    "method": method,
                    "policy": decision.model_dump(),
                    "summary": "Подготовить черновик изменения в Битрикс24; выполнение только после подтверждения.",
                },
            )
        ]

    def _portal_search_action(self, request: str, selected_skills: list[str]) -> ActionRecord | None:
        if "portal_document_search" not in selected_skills:
            return None
        result = self.tools.portal_search_contract(
            {
                "query": request,
                "scope": "documents",
                "limit": 5,
            }
        )
        return ActionRecord(
            name="portal_search",
            status=result.status,
            details=result.model_dump(),
        )

    def _answer(
        self,
        selected_skills: list[str],
        selected_topics: list[str],
        retrieval_hits: list,
        needs_approval: bool,
        *,
        portal_search_action: ActionRecord | None = None,
    ) -> str:
        if not selected_skills:
            return (
                "Я Битрикс24-специалист. Вижу запрос, но пока не уверен, какой Bitrix-сценарий нужен. "
                "Могу работать с задачами, заявками, проектами, CRM, документами и поиском по порталу."
            )

        parts = ["Битрикс24-специалист принял задачу."]
        parts.append("Подходящие skills: " + ", ".join(selected_skills) + ".")
        if selected_topics:
            parts.append("Подключаемые знания: " + ", ".join(selected_topics) + ".")
        if retrieval_hits:
            labels = [f"{hit.chunk.topic}/{hit.chunk.section}" for hit in retrieval_hits[:2]]
            parts.append("Hybrid RAG подобрал фрагменты: " + "; ".join(labels) + ".")
        if portal_search_action is not None:
            tool_data = portal_search_action.details.get("data") if isinstance(portal_search_action.details, dict) else {}
            results = tool_data.get("results") if isinstance(tool_data, dict) else None
            if portal_search_action.status == "ok" and isinstance(results, list):
                parts.append(f"Локальный индекс портала вернул результатов: {len(results)}.")
            elif portal_search_action.status == "not_configured":
                parts.append("Локальный индекс портала пока не подключён; его нужно перенести из `var/search_index.sqlite` или построить заново.")
        if needs_approval:
            parts.append("Запрос похож на изменение в Bitrix24, поэтому выполнение должно идти через черновик и подтверждение.")
        else:
            parts.append("Запрос похож на read-only сценарий; следующим шагом можно подключать живой Bitrix API или локальный индекс портала.")
        return " ".join(parts)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


