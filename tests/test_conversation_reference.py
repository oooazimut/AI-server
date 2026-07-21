import asyncio

from ai_server.models import AgentTask
from ai_server.orchestrators.conversation_reference import resolve_conversation_reference
from ai_server.orchestrators.internal import _append_conversation_reference
from tests.fakes import FakeOrchestratorStore


def _task(request: str) -> AgentTask:
    return AgentTask(
        task_id="t",
        request=request,
        context={"dialog_key": "chat:4321:user:1", "base_dialog_key": "chat:4321:user:1"},
    )


def test_new_branch_uses_visible_number_and_explicit_continuation_reuses_it():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)
        assert first.error == ""
        assert first.task.context["conversation_number"] == 101

        next_page = await resolve_conversation_reference(_task("101 следующая страница"), store)
        assert next_page.error == ""
        assert next_page.task.request == "следующая страница"
        assert next_page.task.context["conversation_number"] == 101
        assert next_page.task.context["dialog_key"] == first.task.context["dialog_key"]

    asyncio.run(run())


def test_short_continuation_needs_no_number_only_when_one_branch_is_active():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)
        automatic = await resolve_conversation_reference(_task("Покажи следующую страницу"), store)
        assert automatic.error == ""
        assert automatic.task.context["dialog_key"] == first.task.context["dialog_key"]

        await resolve_conversation_reference(_task("Покажи склад Карасева"), store)
        ambiguous = await resolve_conversation_reference(_task("Покажи следующую страницу"), store)
        assert "несколько активных" in ambiguous.error

    asyncio.run(run())


def test_footer_places_number_last_and_rewrites_draft_confirmation_phrase():
    task = _task("create").model_copy(update={"context": {**_task("create").context, "conversation_number": 101}})
    rendered = _append_conversation_reference(
        "Черновик. Для подтверждения отправьте фразу: «да, подтверждаю создание задачи».", task
    )
    assert "«101, да, подтверждаю создание задачи»" in rendered
    assert rendered.endswith("Диалог №101. Для продолжения: «101, …»")
