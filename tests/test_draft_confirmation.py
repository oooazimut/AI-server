from ai_server.agents.bitrix24.draft_confirmation import matches_draft_confirmation


def test_voice_confirmation_ignores_comma_and_case():
    assert matches_draft_confirmation(
        "Да подтверждаю создание задачи", {"_draft_type": "task_create"}
    )


def test_voice_confirmation_allows_one_small_typo_but_not_bare_yes():
    draft = {"_draft_type": "calendar_event"}
    assert matches_draft_confirmation("да подтверждаю создание записи в календае", draft)
    assert matches_draft_confirmation("подтвердить", draft, allow_short_command=True)
    assert not matches_draft_confirmation("подтвердить", draft)
    assert not matches_draft_confirmation("да", draft)
    assert not matches_draft_confirmation("да подтверждаю создание задачи", draft)


def test_voice_confirmation_requires_the_action_type_word():
    assert not matches_draft_confirmation("да подтверждаю создание", {"_draft_type": "task_create"})
    assert not matches_draft_confirmation("подтверждаю создание записи", {"_draft_type": "calendar_event"})
    assert matches_draft_confirmation("подтверждаю создание записи в календае", {"_draft_type": "calendar_event"})
