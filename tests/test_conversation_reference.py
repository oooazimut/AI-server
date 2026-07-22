import asyncio
from datetime import datetime, timedelta

from ai_server.models import AgentTask
from ai_server.orchestrators import conversation_reference as conversation_reference_module
from ai_server.orchestrators.conversation_reference import resolve_conversation_reference
from ai_server.orchestrators.internal import _append_conversation_reference
from ai_server.utils import MOSCOW_TZ
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


def test_new_write_reuses_the_active_draft_branch_but_read_keeps_a_new_branch():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("создай задачу проверить договор"), store)
        base_key = "chat:4321:user:1"
        await store.set_kv(base_key, "conversation_reference_active_draft_branch", first.task.context["dialog_key"])
        await store.set_kv(base_key, "conversation_reference_active_draft_number", "101")

        revision = await resolve_conversation_reference(_task("создай задачу проверить счёт"), store)
        read = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)

        assert revision.task.context["dialog_key"] == first.task.context["dialog_key"]
        assert revision.task.context["conversation_number"] == 101
        assert revision.task.context["conversation_reference_reused_active_draft"] is True
        assert read.task.context["conversation_number"] == 102
        assert read.task.context["dialog_key"] != first.task.context["dialog_key"]

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


def test_footer_uses_only_numbered_confirmation_hint_for_draft():
    task = _task("create").model_copy(update={"context": {**_task("create").context, "conversation_number": 101}})
    rendered = _append_conversation_reference(
        "Черновик. Для подтверждения отправьте фразу: «да, подтверждаю создание задачи».", task
    )
    assert rendered == "Черновик.\n\nДиалог №101. Для подтверждения: «101 подтвердить»"
    assert "следующая" not in rendered
    assert "отправьте фразу" not in rendered


def test_footer_uses_numbered_continuation_only_when_answer_has_next_page():
    task = _task("warehouse").model_copy(update={"context": {**_task("warehouse").context, "conversation_number": 102}})
    rendered = _append_conversation_reference(
        "Показаны позиции 1-50 из 244. Остальные позиции есть; можно запросить следующие.", task
    )
    assert rendered.endswith("Диалог №102. Для продолжения: «102 следующая»")
    assert "подтвердить" not in rendered


def test_footer_uses_only_dialog_number_for_final_answer():
    task = _task("warehouse").model_copy(update={"context": {**_task("warehouse").context, "conversation_number": 103}})
    rendered = _append_conversation_reference("Показаны все 10 позиций с положительным остатком.", task)
    assert rendered.endswith("Диалог №103.")
    assert "Для продолжения" not in rendered
    assert "Для подтверждения" not in rendered


def test_visible_numbers_increment_during_day_and_restart_from_101_next_day(monkeypatch):
    async def run():
        store = FakeOrchestratorStore()
        current = datetime(2026, 7, 21, 9, 0, tzinfo=MOSCOW_TZ)
        monkeypatch.setattr(conversation_reference_module, "_now", lambda: current)

        first = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)
        second = await resolve_conversation_reference(_task("Покажи склад Карасева"), store)

        assert first.task.context["conversation_number"] == 101
        assert second.task.context["conversation_number"] == 102
        assert first.task.context["conversation_day"] == "20260721"

        current = datetime(2026, 7, 22, 9, 0, tzinfo=MOSCOW_TZ)
        next_day = await resolve_conversation_reference(_task("Покажи склад Ивашина"), store)

        assert next_day.task.context["conversation_number"] == 101
        assert next_day.task.context["conversation_day"] == "20260722"
        assert next_day.task.context["dialog_key"] != first.task.context["dialog_key"]

    asyncio.run(run())


def test_read_branch_does_not_hide_numbered_active_draft_confirmation():
    async def run():
        store = FakeOrchestratorStore()
        draft = await resolve_conversation_reference(_task("Напомни завтра позвонить Борисову"), store)
        base_key = "chat:4321:user:1"
        await store.set_kv(base_key, "conversation_reference_active_draft_branch", draft.task.context["dialog_key"])
        await store.set_kv(base_key, "conversation_reference_active_draft_number", "101")

        warehouse = await resolve_conversation_reference(_task("Покажи склад Карасева"), store)
        confirmation = await resolve_conversation_reference(_task("101 подтвердить"), store)

        assert warehouse.task.context["conversation_number"] == 102
        assert warehouse.task.context["dialog_key"] != draft.task.context["dialog_key"]
        assert confirmation.error == ""
        assert confirmation.task.request == "подтвердить"
        assert confirmation.task.context["conversation_number"] == 101
        assert confirmation.task.context["dialog_key"] == draft.task.context["dialog_key"]
        assert confirmation.task.context["conversation_reference_explicit"] is True

    asyncio.run(run())


def test_numbered_active_draft_cancel_reuses_branch_and_strips_number():
    async def run():
        store = FakeOrchestratorStore()
        draft = await resolve_conversation_reference(_task("Создай задачу проверить договор"), store)

        cancellation = await resolve_conversation_reference(_task("101 отменить"), store)

        assert cancellation.error == ""
        assert cancellation.task.request == "отменить"
        assert cancellation.task.context["conversation_number"] == 101
        assert cancellation.task.context["dialog_key"] == draft.task.context["dialog_key"]
        assert cancellation.task.context["conversation_reference_explicit"] is True

    asyncio.run(run())


def test_numbered_continuation_expires_after_fifteen_minutes(monkeypatch):
    async def run():
        store = FakeOrchestratorStore()
        current = datetime(2026, 7, 21, 9, 0, tzinfo=MOSCOW_TZ)
        monkeypatch.setattr(conversation_reference_module, "_now", lambda: current)
        await resolve_conversation_reference(_task("Покажи склад Борисова"), store)

        current += timedelta(minutes=16)
        expired = await resolve_conversation_reference(_task("101 следующая страница"), store)

        assert "не активен" in expired.error.casefold()
        assert expired.task.context["conversation_reference_dispatch_allowed"] is False

    asyncio.run(run())
