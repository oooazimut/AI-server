import logging

from ai_server.agents.specialist_llm_shared import load_instructions
from ai_server.models import AgentManifest


def _manifest_with_instructions(instructions_file: str | None) -> AgentManifest:
    return AgentManifest(
        id="test_agent",
        name="Test Agent",
        kind="specialist",
        description="Test",
        instructions_file=instructions_file,
    )


def test_no_instructions_file_returns_empty():
    manifest = _manifest_with_instructions(None)
    assert load_instructions(manifest) == ""


def test_existing_instructions_file_returns_content(tmp_path):
    instructions_path = tmp_path / "instructions.txt"
    instructions_path.write_text("  Custom instructions here.  ", encoding="utf-8")
    manifest = _manifest_with_instructions(str(instructions_path))
    assert load_instructions(manifest) == "Custom instructions here."


def test_missing_instructions_file_returns_empty(tmp_path):
    nonexistent = tmp_path / "does_not_exist.txt"
    manifest = _manifest_with_instructions(str(nonexistent))
    assert load_instructions(manifest) == ""


def test_missing_instructions_file_logs_warning(tmp_path, caplog):
    nonexistent = tmp_path / "missing.txt"
    manifest = _manifest_with_instructions(str(nonexistent))
    with caplog.at_level(logging.WARNING):
        load_instructions(manifest)
    assert any("missing.txt" in record.message for record in caplog.records)


def test_instructions_stripped_of_whitespace(tmp_path):
    path = tmp_path / "instr.txt"
    path.write_text("\n\n  Leading and trailing spaces.  \n\n", encoding="utf-8")
    manifest = _manifest_with_instructions(str(path))
    assert load_instructions(manifest) == "Leading and trailing spaces."


def test_empty_instructions_file_string_returns_empty():
    manifest = _manifest_with_instructions("")
    assert load_instructions(manifest) == ""
