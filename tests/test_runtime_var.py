from pathlib import Path

from ai_server.runtime import runtime_paths
from ai_server.runtime_migration import (
    EXCLUDED_RUNTIME_ITEMS,
    build_var_migration_plan,
    migrate_var,
    selected_migration_items,
)


def test_runtime_paths_use_var_root():
    paths = runtime_paths(Path("var"))

    assert paths.search_index_db == Path("var") / "search_index.sqlite"
    assert paths.bitrix_oauth_db == Path("var") / "bitrix_oauth.sqlite"
    assert paths.dialog_state_db == Path("var") / "dialog_state.sqlite"
    assert paths.attachments_dir == Path("var") / "attachments"


def test_cutover_profile_includes_sensitive_continuity_state():
    items = {item.path: item for item in selected_migration_items("cutover")}

    assert items["bitrix_oauth.sqlite"].sensitive is True
    assert items["bitrix_oauth.sqlite"].required_for_cutover is True
    assert items["dialog_state.sqlite"].sensitive is True
    assert items["dialog_state.sqlite"].required_for_cutover is True
    assert "attachments" in items
    assert "document_drafts" in items


def test_migration_plan_marks_existing_and_missing_items(tmp_path):
    source = tmp_path / "old" / "var"
    target = tmp_path / "new" / "var"
    source.mkdir(parents=True)
    (source / "bitrix_oauth.sqlite").write_text("token-db", encoding="utf-8")
    (source / "dialog_state.sqlite").write_text("dialog-db", encoding="utf-8")

    plan = build_var_migration_plan(source, target, profile="cutover")
    by_path = {item["path"]: item for item in plan}

    assert by_path["bitrix_oauth.sqlite"]["exists"] is True
    assert by_path["bitrix_oauth.sqlite"]["action"] == "copy"
    assert by_path["dialog_state.sqlite"]["exists"] is True
    assert by_path["search_index.sqlite"]["exists"] is False
    assert by_path["search_index.sqlite"]["action"] == "missing"


def test_transient_old_runtime_items_are_excluded():
    assert "search_indexer.lock" in EXCLUDED_RUNTIME_ITEMS
    assert "tmp*" in EXCLUDED_RUNTIME_ITEMS


def test_migrate_var_preserves_gitkeep_for_runtime_directories(tmp_path):
    source = tmp_path / "old" / "var"
    target = tmp_path / "new" / "var"
    (source / "attachments").mkdir(parents=True)
    (source / "attachments" / "voice.m4a").write_text("audio", encoding="utf-8")
    (target / "attachments").mkdir(parents=True)
    (target / "attachments" / ".gitkeep").write_text("", encoding="utf-8")

    result = migrate_var(source, target, profile="attachments", execute=True)

    assert result[0]["result"] == "copied"
    assert (target / "attachments" / "voice.m4a").exists()
    assert (target / "attachments" / ".gitkeep").exists()
    assert (target / "legacy" / "backups").exists()
