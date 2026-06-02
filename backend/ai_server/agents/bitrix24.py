from __future__ import annotations

from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask
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
        tools: BitrixToolset | None = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base or MarkdownKnowledgeBase()
        self.skill_store = skill_store or SkillStore()
        self.tools = tools or BitrixToolset()

    async def handle(self, task: AgentTask) -> AgentResult:
        text = task.request.casefold()
        selected_skills = self._select_skills(text)
        selected_topics = self._select_knowledge_topics(text)
        actions_taken = [
            ActionRecord(
                name="load_bitrix_specialist_context",
                status="completed",
                details={"skills": selected_skills, "knowledge_topics": selected_topics},
            )
        ]

        approval_actions = self._approval_actions(text)
        status = "needs_human" if approval_actions else "completed"
        answer = self._answer(selected_skills, selected_topics, bool(approval_actions))

        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=answer,
            actions_taken=actions_taken,
            actions_requiring_approval=approval_actions,
            confidence=0.72 if selected_skills else 0.48,
            logs=[
                "Bitrix24 specialist was separated from the old autonomous BitrixAIAgent runtime.",
                "Current MVP uses deterministic routing; LLM tool loop will be added behind the same contract.",
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
        if any(marker in text for marker in ("документ", "файл", "смет", "диск", "вложен")):
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
        if any(marker in text for marker in ("документ", "файл", "смет", "диск")):
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

    def _answer(self, selected_skills: list[str], selected_topics: list[str], needs_approval: bool) -> str:
        if not selected_skills:
            return (
                "Я Битрикс24-специалист. Вижу запрос, но пока не уверен, какой Bitrix-сценарий нужен. "
                "Могу работать с задачами, заявками, проектами, CRM, документами и поиском по порталу."
            )

        parts = ["Битрикс24-специалист принял задачу."]
        parts.append("Подходящие skills: " + ", ".join(selected_skills) + ".")
        if selected_topics:
            parts.append("Подключаемые знания: " + ", ".join(selected_topics) + ".")
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
