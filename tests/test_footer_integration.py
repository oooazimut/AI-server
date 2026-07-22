from ai_server.models import AgentTask
from ai_server.orchestrators.internal import _append_conversation_reference
from ai_server.technical_footer import append_footer

TECHNICAL_FOOTER = "оркестр → Bitrix, 5177/299 ток., DeepSeek OK, $12.34"


def _task(number: int) -> AgentTask:
    return AgentTask(
        task_id="footer-integration",
        request="test",
        context={"conversation_number": number},
    )


def _render(message: str, number: int, technical_footer: str = TECHNICAL_FOOTER) -> str:
    return _append_conversation_reference(
        append_footer(message, technical_footer),
        _task(number),
    )


def test_technical_footer_precedes_single_final_dialog_reference():
    rendered = _render("Показаны все 10 позиций с положительным остатком.", 101)

    assert rendered.count("Тех:") == 1
    assert rendered.count("Диалог №101") == 1
    assert rendered.index("Тех:") < rendered.index("Диалог №101")
    assert rendered.endswith("Диалог №101.")


def test_technical_footer_and_draft_keep_only_short_numbered_confirmation():
    rendered = _render(
        "Черновик. Для подтверждения отправьте фразу: «да, подтверждаю создание задачи».",
        102,
    )

    assert rendered.count("Тех:") == 1
    assert rendered.count("Диалог №102") == 1
    assert "Для подтверждения отправьте фразу" not in rendered
    assert rendered.count("«102 подтвердить»") == 1
    assert rendered.index("Тех:") < rendered.index("Диалог №102")
    assert rendered.endswith("Диалог №102. Для подтверждения: «102 подтвердить»")


def test_technical_footer_and_pagination_keep_only_one_numbered_continuation():
    rendered = _render(
        "Показаны позиции 1-50 из 244. Остальные позиции есть; можно запросить следующие.",
        103,
    )

    assert rendered.count("Тех:") == 1
    assert rendered.count("Диалог №103") == 1
    assert rendered.count("«103 следующая»") == 1
    assert rendered.index("Тех:") < rendered.index("Диалог №103")
    assert rendered.endswith("Диалог №103. Для продолжения: «103 следующая»")


def test_active_draft_conflict_keeps_only_numbered_confirm_and_cancel_commands():
    rendered = _render(
        "Новый запрос не запущен.\n\n"
        "Активный черновик — создание задачи: «Проверить договор».\n\n"
        "Для управления активным черновиком: подтвердить или отменить.",
        104,
    )

    assert rendered.count("Тех:") == 1
    assert rendered.count("Диалог №104") == 1
    assert "Для управления активным черновиком" not in rendered
    assert rendered.count("«104 подтвердить»") == 1
    assert rendered.count("«104 отменить»") == 1
    assert rendered.index("Тех:") < rendered.index("Диалог №104")
    assert rendered.endswith(
        "Диалог №104. Для подтверждения: «104 подтвердить». Для отмены: «104 отменить»"
    )


def test_hidden_technical_footer_does_not_remove_dialog_reference():
    rendered = _render("Готово.", 105, technical_footer="")

    assert "Тех:" not in rendered
    assert rendered.count("Диалог №105") == 1
    assert rendered == "Готово.\n\nДиалог №105."
