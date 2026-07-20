from ai_server.agents.bitrix24.draft_confirmation import matches_draft_confirmation


def test_voice_confirmation_ignores_comma_and_case():
    assert matches_draft_confirmation(
        "Да подтверждаю создание задачи", {"_draft_type": "task_create"}
    )


def test_voice_confirmation_allows_one_small_typo_but_not_bare_yes():
    draft = {"_draft_type": "calendar_event"}
    assert matches_draft_confirmation("да подтверждаю создание записи в календае", draft)
    assert not matches_draft_confirmation("да", draft)
    assert not matches_draft_confirmation("да подтверждаю создание задачи", draft)
