from pathlib import Path

from ai_server.knowledge import (
    MarkdownKnowledgeBase,
    _find_section,
    _parse_sections,
    _preview,
    _search_sections,
)
from ai_server.models import AgentManifest


def _manifest(tmp_path: Path) -> AgentManifest:
    return AgentManifest(
        id="test_agent",
        name="Test Agent",
        kind="specialist",
        description="Test",
        knowledge_path=str(tmp_path),
    )


def _write_topic(directory: Path, filename: str, content: str) -> Path:
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


# list_topics
def test_list_topics_empty_directory(tmp_path):
    kb = MarkdownKnowledgeBase()
    manifest = _manifest(tmp_path)
    assert kb.list_topics(manifest) == []


def test_list_topics_nonexistent_directory():
    kb = MarkdownKnowledgeBase()
    manifest = AgentManifest(
        id="test_agent",
        name="Test",
        kind="specialist",
        description="Test",
        knowledge_path="/nonexistent/path/that/does/not/exist",
    )
    assert kb.list_topics(manifest) == []


def test_list_topics_returns_topics(tmp_path):
    _write_topic(tmp_path, "vacations.md", "# Отпуска\nПолитика отпусков.")
    _write_topic(tmp_path, "equipment.md", "# Оборудование\nПеречень оборудования.")
    kb = MarkdownKnowledgeBase()
    topics = kb.list_topics(_manifest(tmp_path))
    names = [t.name for t in topics]
    assert "vacations" in names
    assert "equipment" in names


def test_list_topics_extracts_title(tmp_path):
    _write_topic(tmp_path, "rules.md", "# Правила работы\nТекст правил.")
    kb = MarkdownKnowledgeBase()
    topics = kb.list_topics(_manifest(tmp_path))
    assert topics[0].title == "Правила работы"


def test_list_topics_fallback_title_from_stem(tmp_path):
    _write_topic(tmp_path, "work_rules.md", "Текст без заголовка.")
    kb = MarkdownKnowledgeBase()
    topics = kb.list_topics(_manifest(tmp_path))
    assert topics[0].title == "work rules"


# execute — list
def test_execute_list_action(tmp_path):
    _write_topic(tmp_path, "policy.md", "# Политика\nТекст.")
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "list"})
    assert result["status"] == "ok"
    assert any(t["name"] == "policy" for t in result["topics"])


def test_execute_unknown_action(tmp_path):
    _write_topic(tmp_path, "policy.md", "# Политика\nТекст.")
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "fly"})
    assert result["status"] == "error"


def test_execute_unknown_topic(tmp_path):
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "read", "topic": "nosuchone"})
    assert result["status"] == "not_found"


# execute — read
_SAMPLE_DOC = "# Договоры\n\n## Общие положения\nТекст раздела один.\n\n## Порядок согласования\nТекст раздела два."


def test_execute_read_returns_content(tmp_path):
    _write_topic(tmp_path, "contracts.md", _SAMPLE_DOC)
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "read", "topic": "contracts"})
    assert result["status"] == "ok"
    assert "Договоры" in result["content"]


# execute — outline
def test_execute_outline_returns_sections(tmp_path):
    _write_topic(tmp_path, "contracts.md", _SAMPLE_DOC)
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "outline", "topic": "contracts"})
    assert result["status"] == "ok"
    section_slugs = [s["section"] for s in result["sections"]]
    assert any("положения" in slug for slug in section_slugs)


# execute — read_section
def test_execute_read_section_found(tmp_path):
    _write_topic(tmp_path, "contracts.md", _SAMPLE_DOC)
    kb = MarkdownKnowledgeBase()
    result = kb.execute(
        _manifest(tmp_path), {"action": "read_section", "topic": "contracts", "section": "Общие положения"}
    )
    assert result["status"] == "ok"
    assert "Текст раздела один" in result["content"]


def test_execute_read_section_not_found(tmp_path):
    _write_topic(tmp_path, "contracts.md", _SAMPLE_DOC)
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "read_section", "topic": "contracts", "section": "Нет такого"})
    assert result["status"] == "not_found"
    assert "available_sections" in result


# execute — search
def test_execute_search_finds_relevant_section(tmp_path):
    _write_topic(tmp_path, "contracts.md", _SAMPLE_DOC)
    kb = MarkdownKnowledgeBase()
    result = kb.execute(_manifest(tmp_path), {"action": "search", "topic": "contracts", "query": "порядок"})
    assert result["status"] == "ok"
    assert len(result["matches"]) > 0
    assert any("порядок" in m["section"] for m in result["matches"])


# _parse_sections
def test_parse_sections_no_headers():
    sections = _parse_sections("Просто текст без разделов.")
    assert len(sections) == 1
    assert sections[0].slug == "document"


def test_parse_sections_multiple_headers():
    content = "# Документ\n\n## Раздел A\nТекст А.\n\n## Раздел Б\nТекст Б."
    sections = _parse_sections(content)
    assert len(sections) == 2
    assert any(s.slug == "раздел-a" or "раздел" in s.slug for s in sections)


def test_parse_sections_skips_toc():
    content = "## Оглавление\nПункты.\n\n## Содержание раздела\nТекст."
    sections = _parse_sections(content)
    # "Оглавление" section should be skipped, "Содержание раздела" kept
    assert not any(s.slug == "оглавление" for s in sections)


# _find_section
def test_find_section_by_title():
    sections = _parse_sections("## Общие положения\nТекст.")
    result = _find_section(sections, "Общие положения")
    assert result is not None
    assert "положен" in result.title.lower()


def test_find_section_partial_match():
    sections = _parse_sections("## Порядок согласования\nТекст.")
    result = _find_section(sections, "согласован")
    assert result is not None


def test_find_section_not_found():
    sections = _parse_sections("## Первый раздел\nТекст.")
    assert _find_section(sections, "Несуществующий") is None


# _preview
def test_preview_short_text_unchanged():
    text = "Короткий текст."
    assert _preview(text) == text


def test_preview_long_text_truncated():
    text = "A" * 300
    result = _preview(text)
    assert len(result) <= 223
    assert result.endswith("...")


# _search_sections
def test_search_sections_scores_title_higher():
    sections = _parse_sections(
        "## Согласование договоров\nМинимальный текст.\n\n## Другой раздел\nСогласование упомянуто здесь много раз согласование согласование."
    )
    results = _search_sections(sections, "согласование договор", limit=5)
    assert results[0].title == "Согласование договоров"
