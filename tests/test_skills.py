from pathlib import Path

from ai_server.models import AgentManifest
from ai_server.skills import SkillStore


def _manifest(skills_dir: Path) -> AgentManifest:
    return AgentManifest(
        id="test_agent",
        name="Test Agent",
        kind="specialist",
        description="Test",
        skills_path=str(skills_dir),
    )


def _write_skill(directory: Path, filename: str, content: str) -> None:
    (directory / filename).write_text(content, encoding="utf-8")


# list_skills
def test_list_skills_empty_directory(tmp_path):
    store = SkillStore()
    assert store.list_skills(_manifest(tmp_path)) == []


def test_list_skills_nonexistent_directory():
    store = SkillStore()
    manifest = AgentManifest(
        id="test_agent",
        name="Test",
        kind="specialist",
        description="Test",
        skills_path="/nonexistent/skills/path",
    )
    assert store.list_skills(manifest) == []


def test_list_skills_returns_skills(tmp_path):
    _write_skill(tmp_path, "task_search.md", "# Поиск задач\nКак искать задачи.")
    _write_skill(tmp_path, "safe_write.md", "# Безопасная запись\nКак писать в Bitrix.")
    store = SkillStore()
    skills = store.list_skills(_manifest(tmp_path))
    ids = [s.id for s in skills]
    assert "task_search" in ids
    assert "safe_write" in ids


def test_list_skills_extracts_title(tmp_path):
    _write_skill(tmp_path, "task_search.md", "# Поиск задач\nКак искать задачи.")
    store = SkillStore()
    skills = store.list_skills(_manifest(tmp_path))
    assert skills[0].title == "Поиск задач"


def test_list_skills_fallback_title_from_stem(tmp_path):
    _write_skill(tmp_path, "task_search.md", "Нет заголовка #.")
    store = SkillStore()
    skills = store.list_skills(_manifest(tmp_path))
    assert skills[0].title == "task search"


def test_list_skills_has_no_content(tmp_path):
    _write_skill(tmp_path, "task_search.md", "# Задачи\nТекст навыка.")
    store = SkillStore()
    skills = store.list_skills(_manifest(tmp_path))
    assert skills[0].content is None


def test_list_skills_has_preview(tmp_path):
    _write_skill(tmp_path, "task_search.md", "# Задачи\nТекст навыка.")
    store = SkillStore()
    skills = store.list_skills(_manifest(tmp_path))
    assert skills[0].preview


# read_skill
def test_read_skill_found(tmp_path):
    _write_skill(tmp_path, "task_search.md", "# Поиск задач\nПодробное описание навыка.")
    store = SkillStore()
    skill = store.read_skill(_manifest(tmp_path), "task_search")
    assert skill is not None
    assert skill.id == "task_search"
    assert skill.content is not None
    assert "Подробное описание" in skill.content


def test_read_skill_not_found(tmp_path):
    store = SkillStore()
    assert store.read_skill(_manifest(tmp_path), "nonexistent") is None


def test_read_skill_has_title(tmp_path):
    _write_skill(tmp_path, "safe_write.md", "# Безопасная запись\nТекст.")
    store = SkillStore()
    skill = store.read_skill(_manifest(tmp_path), "safe_write")
    assert skill is not None
    assert skill.title == "Безопасная запись"


def test_read_skill_preview_truncated(tmp_path):
    long_content = "# Навык\n" + "Текст. " * 100
    _write_skill(tmp_path, "long.md", long_content)
    store = SkillStore()
    skill = store.read_skill(_manifest(tmp_path), "long")
    assert skill is not None
    assert len(skill.preview) <= 243
