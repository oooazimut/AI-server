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


def test_short_continuation_without_number_is_restricted_even_with_one_branch():
    async def run():
        store = FakeOrchestratorStore()
        await resolve_conversation_reference(_task("Покажи склад Борисова"), store)
        restricted = await resolve_conversation_reference(_task("Покажи следующую страницу"), store)
        assert "укажите номер диалога" in restricted.error.casefold()
        assert restricted.task.context["conversation_reference_dispatch_allowed"] is False

    asyncio.run(run())


def test_continuation_accepts_plain_visible_number_at_the_end():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)

        continued = await resolve_conversation_reference(_task("Покажи следующую страницу 101"), store)

        assert continued.error == ""
        assert continued.task.request == "Покажи следующую страницу"
        assert continued.task.context["dialog_key"] == first.task.context["dialog_key"]
        assert continued.task.context["conversation_reference_explicit"] is True

    asyncio.run(run())


def test_two_explicit_numbers_load_their_own_dialog_branches():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)
        second = await resolve_conversation_reference(_task("Покажи склад Карасева"), store)

        continued = await resolve_conversation_reference(_task("следующая 102"), store)

        assert first.task.context["conversation_number"] == 101
        assert second.task.context["conversation_number"] == 102
        assert continued.task.context["dialog_key"] == second.task.context["dialog_key"]
        assert continued.task.context["dialog_key"] != first.task.context["dialog_key"]

    asyncio.run(run())


def test_composite_dialog_keeps_one_root_and_never_requires_a_part_number():
    async def run():
        store = FakeOrchestratorStore()
        root = await resolve_conversation_reference(_task("Покажи склады Борисова и Карасева"), store)
        continued = await resolve_conversation_reference(_task("Покажи следующую страницу склада Борисова 101"), store)

        assert continued.error == ""
        assert continued.task.context["dialog_key"] == root.task.context["dialog_key"]
        assert "conversation_part" not in continued.task.context

    asyncio.run(run())


def test_legacy_part_reference_is_rejected_before_any_specialist_dispatch():
    async def run():
        store = FakeOrchestratorStore()
        await resolve_conversation_reference(_task("Покажи склад Борисова"), store)

        rejected = await resolve_conversation_reference(_task("101 часть 1 покажи остатки"), store)

        assert "Части ответа больше не используются" in rejected.error
        assert rejected.task.context["conversation_reference_dispatch_allowed"] is False

    asyncio.run(run())


def test_footer_places_number_last_and_rewrites_draft_confirmation_phrase():
    task = _task("create").model_copy(update={"context": {**_task("create").context, "conversation_number": 101}})
    rendered = _append_conversation_reference(
        "Черновик. Для подтверждения отправьте фразу: «да, подтверждаю создание задачи».", task
    )
    assert "«101 подтвердить»" in rendered
    assert rendered.endswith("Диалог №101. Для продолжения: «101 следующая»")
