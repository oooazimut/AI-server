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


def test_new_unnumbered_write_ignores_legacy_active_draft_pointer_and_gets_new_branch():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("создай задачу проверить договор"), store)
        base_key = "chat:4321:user:1"
        await store.set_kv(base_key, "conversation_reference_active_draft_branch", first.task.context["dialog_key"])
        await store.set_kv(base_key, "conversation_reference_active_draft_number", "101")

        revision = await resolve_conversation_reference(_task("создай задачу проверить счёт"), store)
        read = await resolve_conversation_reference(_task("Покажи склад Борисова"), store)

        assert revision.task.context["dialog_key"] != first.task.context["dialog_key"]
        assert revision.task.context["conversation_number"] == 102
        assert "conversation_reference_reused_active_draft" not in revision.task.context
        assert read.task.context["conversation_number"] == 103
        assert read.task.context["dialog_key"] != first.task.context["dialog_key"]

    asyncio.run(run())


def test_task_reminder_calendar_project_and_close_drafts_each_get_their_own_number():
    async def run():
        store = FakeOrchestratorStore()
        requests = [
            "Создай задачу проверить договор",
            "Напомни завтра позвонить Борисову",
            "Добавь встречу в календарь завтра в 10:00",
            "Создай проект Монтаж",
            "Закрой задачу 555",
        ]

        branches = [await resolve_conversation_reference(_task(request), store) for request in requests]

        assert [item.task.context["conversation_number"] for item in branches] == [101, 102, 103, 104, 105]
        assert len({item.task.context["dialog_key"] for item in branches}) == len(requests)

    asyncio.run(run())


def test_numbering_is_source_agnostic_for_warehouse_disk_tasks_and_composite_search():
    async def run():
        store = FakeOrchestratorStore()
        requests = [
            "Покажи склад Борисова",
            "Найди на Диске Bitrix документ Инструкция",
            "Покажи мои активные задачи",
            "Покажи склад Борисова и найди проект Монтаж",
        ]

        branches = [await resolve_conversation_reference(_task(request), store) for request in requests]

        assert [item.task.context["conversation_number"] for item in branches] == [101, 102, 103, 104]
        assert len({item.task.context["dialog_key"] for item in branches}) == len(requests)

    asyncio.run(run())


def test_every_unnumbered_user_turn_gets_a_new_number_without_intent_classification():
    async def run():
        store = FakeOrchestratorStore()

        first = await resolve_conversation_reference(_task("Спасибо"), store)
        second = await resolve_conversation_reference(_task("Есть ещё один вопрос"), store)

        assert first.task.context["conversation_number"] == 101
        assert second.task.context["conversation_number"] == 102
        assert first.task.context["dialog_key"] != second.task.context["dialog_key"]

    asyncio.run(run())


def test_internal_task_without_user_chat_key_is_not_numbered():
    task = AgentTask(task_id="internal", request="quality_control", context={"bitrix_event_type": "ONTASKUPDATE"})

    resolved = asyncio.run(resolve_conversation_reference(task, FakeOrchestratorStore()))

    assert "conversation_number" not in resolved.task.context
    assert resolved.task.context == task.context


def test_three_parallel_draft_numbers_keep_independent_confirmation_and_cancel_routes():
    async def run():
        store = FakeOrchestratorStore()
        first = await resolve_conversation_reference(_task("Создай задачу проверить договор"), store)
        second = await resolve_conversation_reference(_task("Напомни завтра позвонить Борисову"), store)
        third = await resolve_conversation_reference(_task("Закрой задачу 555"), store)

        confirm_second = await resolve_conversation_reference(_task("102 подтвердить"), store)
        cancel_first = await resolve_conversation_reference(_task("101 отменить"), store)
        change_third = await resolve_conversation_reference(_task("103 изменить результат"), store)

        assert confirm_second.task.context["dialog_key"] == second.task.context["dialog_key"]
        assert cancel_first.task.context["dialog_key"] == first.task.context["dialog_key"]
        assert change_third.task.context["dialog_key"] == third.task.context["dialog_key"]
        assert confirm_second.task.request == "подтвердить"
        assert cancel_first.task.request == "отменить"
        assert change_third.task.request == "изменить результат"

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


def test_yesterday_number_is_inactive_after_midnight_before_new_day_starts(monkeypatch):
    async def run():
        store = FakeOrchestratorStore()
        current = datetime(2026, 7, 21, 23, 59, tzinfo=MOSCOW_TZ)
        monkeypatch.setattr(conversation_reference_module, "_now", lambda: current)
        yesterday = await resolve_conversation_reference(_task("Создай задачу проверить договор"), store)

        current = datetime(2026, 7, 22, 0, 1, tzinfo=MOSCOW_TZ)
        rejected = await resolve_conversation_reference(_task("101 подтвердить"), store)
        today = await resolve_conversation_reference(_task("Создай задачу проверить счёт"), store)

        assert rejected.task.context["conversation_reference_dispatch_allowed"] is False
        assert today.task.context["conversation_number"] == 101
        assert today.task.context["dialog_key"] != yesterday.task.context["dialog_key"]

    asyncio.run(run())


def test_invalid_or_foreign_day_counter_starts_current_day_from_101(monkeypatch):
    async def run():
        invalid_store = FakeOrchestratorStore()
        foreign_day_store = FakeOrchestratorStore()
        base_key = "chat:4321:user:1"
        current = datetime(2026, 7, 22, 9, 0, tzinfo=MOSCOW_TZ)
        monkeypatch.setattr(conversation_reference_module, "_now", lambda: current)

        await invalid_store.set_kv(base_key, "conversation_reference_counter", "broken")
        first = await resolve_conversation_reference(_task("Покажи склад Борисова"), invalid_store)
        await foreign_day_store.set_kv(base_key, "conversation_reference_counter", "20260721:999")
        second = await resolve_conversation_reference(_task("Покажи склад Карасева"), foreign_day_store)

        assert first.task.context["conversation_number"] == 101
        assert second.task.context["conversation_number"] == 101

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
