import logging

import pytest

from ai_server.agents.bitrix_llm import _load_instructions as bitrix_load_instructions
from ai_server.agents.logistics_llm import _load_instructions as logistics_load_instructions
from ai_server.agents.pto_llm import _load_instructions as pto_load_instructions
from ai_server.models import AgentManifest

_LOADERS = [
    ("bitrix_llm", bitrix_load_instructions),
    ("pto_llm", pto_load_instructions),
    ("logistics_llm", logistics_load_instructions),
]


def _manifest_with_instructions(instructions_file: str | None) -> AgentManifest:
    return AgentManifest(
        id="test_agent",
        name="Test Agent",
        kind="specialist",
        description="Test",
        instructions_file=instructions_file,
    )


@pytest.mark.parametrize("name,loader", _LOADERS)
def test_no_instructions_file_returns_empty(name, loader):
    manifest = _manifest_with_instructions(None)
    assert loader(manifest) == ""


@pytest.mark.parametrize("name,loader", _LOADERS)
def test_existing_instructions_file_returns_content(name, loader, tmp_path):
    instructions_path = tmp_path / "instructions.txt"
    instructions_path.write_text("  Custom instructions here.  ", encoding="utf-8")
    manifest = _manifest_with_instructions(str(instructions_path))
    result = loader(manifest)
    assert result == "Custom instructions here."


@pytest.mark.parametrize("name,loader", _LOADERS)
def test_missing_instructions_file_returns_empty(name, loader, tmp_path):
    nonexistent = tmp_path / "does_not_exist.txt"
    manifest = _manifest_with_instructions(str(nonexistent))
    result = loader(manifest)
    assert result == ""


@pytest.mark.parametrize("name,loader", _LOADERS)
def test_missing_instructions_file_logs_warning(name, loader, tmp_path, caplog):
    nonexistent = tmp_path / "missing.txt"
    manifest = _manifest_with_instructions(str(nonexistent))
    with caplog.at_level(logging.WARNING):
        loader(manifest)
    assert any("missing.txt" in record.message for record in caplog.records)


@pytest.mark.parametrize("name,loader", _LOADERS)
def test_instructions_stripped_of_whitespace(name, loader, tmp_path):
    path = tmp_path / "instr.txt"
    path.write_text("\n\n  Leading and trailing spaces.  \n\n", encoding="utf-8")
    manifest = _manifest_with_instructions(str(path))
    assert loader(manifest) == "Leading and trailing spaces."


@pytest.mark.parametrize("name,loader", _LOADERS)
def test_empty_instructions_file_string_returns_empty(name, loader):
    manifest = _manifest_with_instructions("")
    assert loader(manifest) == ""
