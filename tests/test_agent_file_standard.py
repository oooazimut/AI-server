from pathlib import Path

import yaml

from ai_server.registry import PROJECT_ROOT, load_agent_manifests


def test_all_agent_packages_follow_rule_and_skill_file_standard():
    for manifest in load_agent_manifests():
        agent_dir = PROJECT_ROOT / "agents" / manifest.id

        assert (agent_dir / "manifest.yaml").exists(), manifest.id
        assert (agent_dir / "instructions.md").exists(), manifest.id
        assert (agent_dir / "rule_index.yaml").exists(), manifest.id
        assert (agent_dir / "skill_index.yaml").exists(), manifest.id
        assert (agent_dir / "knowledge").is_dir(), manifest.id
        assert (agent_dir / "skills").is_dir(), manifest.id


def test_agent_rule_and_skill_indexes_reference_existing_files():
    for manifest in load_agent_manifests():
        agent_dir = PROJECT_ROOT / "agents" / manifest.id
        _assert_index_files_exist(agent_dir / "rule_index.yaml", "rules")
        _assert_index_files_exist(agent_dir / "skill_index.yaml", "skills")


def _assert_index_files_exist(index_path: Path, section: str) -> None:
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    entries = payload.get(section) or []

    for entry in entries:
        file_name = str(entry.get("file") or "").strip()
        assert file_name, f"{index_path}: missing file in {entry.get('id')}"
        assert (index_path.parent / file_name).exists(), f"{index_path}: missing {file_name}"
