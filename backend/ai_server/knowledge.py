import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import AgentManifest
from .registry import agent_package_path, resolve_project_path


class KnowledgeTopic(BaseModel):
    name: str
    title: str
    description: str = ""
    path: str


class KnowledgeSection(BaseModel):
    slug: str
    title: str
    content: str


class MarkdownKnowledgeBase:
    def list_topics(self, manifest: AgentManifest) -> list[KnowledgeTopic]:
        directory = self._topics_path(manifest)
        if not directory.exists():
            return []
        return [self._topic_from_file(path) for path in sorted(directory.glob("*.md"))]

    def execute(self, manifest: AgentManifest, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "list").strip().lower()
        topics = {topic.name: topic for topic in self.list_topics(manifest)}

        if action == "list":
            return {"status": "ok", "topics": [topic.model_dump() for topic in topics.values()]}
        if action not in {"read", "outline", "search", "read_section"}:
            return {"status": "error", "error": f"unknown knowledge action: {action}"}

        topic_name = str(args.get("topic") or "").strip().lower()
        topic = topics.get(topic_name)
        if topic is None:
            return {"status": "not_found", "available_topics": sorted(topics)}

        path = Path(topic.path)
        content = path.read_text(encoding="utf-8").strip()
        sections = _parse_sections(content)

        if action == "outline":
            return {
                "status": "ok",
                "topic": topic.name,
                "title": topic.title,
                "sections": [
                    {"section": section.slug, "title": section.title, "preview": _preview(section.content)}
                    for section in sections
                ],
            }

        if action == "search":
            query = str(args.get("query") or "").strip()
            limit = max(1, min(int(args.get("limit") or 5), 10))
            matches = _search_sections(sections, query, limit=limit)
            return {
                "status": "ok",
                "topic": topic.name,
                "title": topic.title,
                "query": query,
                "matches": [
                    {
                        "section": section.slug,
                        "title": section.title,
                        "preview": _preview(section.content, max_chars=500),
                    }
                    for section in matches
                ],
            }

        if action == "read_section":
            section_name = str(args.get("section") or "").strip()
            section = _find_section(sections, section_name)
            if section is None:
                return {
                    "status": "not_found",
                    "available_sections": [{"section": item.slug, "title": item.title} for item in sections],
                }
            return {
                "status": "ok",
                "topic": topic.name,
                "title": topic.title,
                "section": section.slug,
                "section_title": section.title,
                "content": section.content,
            }

        return {"status": "ok", "topic": topic.name, "title": topic.title, "content": content}

    def _topics_path(self, manifest: AgentManifest) -> Path:
        configured = resolve_project_path(manifest.knowledge_path)
        return configured or agent_package_path(manifest.id) / "knowledge" / "topics"

    def _topic_from_file(self, path: Path) -> KnowledgeTopic:
        content = path.read_text(encoding="utf-8", errors="ignore")
        title = _title_from_markdown(content) or path.stem.replace("_", " ")
        description = _first_paragraph(content)
        return KnowledgeTopic(name=path.stem, title=title, description=description, path=str(path))


def _parse_sections(content: str) -> list[KnowledgeSection]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", content))
    if not matches:
        return [KnowledgeSection(slug="document", title="Документ", content=content)]

    sections: list[KnowledgeSection] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        if _normalize(title) in {"оглавление", "содержание"}:
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        sections.append(KnowledgeSection(slug=_slugify(title), title=title, content=content[start:end].strip()))
    return sections


def _find_section(sections: list[KnowledgeSection], section_name: str) -> KnowledgeSection | None:
    target = _normalize(section_name)
    if not target:
        return None
    for section in sections:
        if target in {_normalize(section.slug), _normalize(section.title)}:
            return section
    for section in sections:
        if target in _normalize(section.title) or target in _normalize(section.slug):
            return section
    return None


def _search_sections(sections: list[KnowledgeSection], query: str, *, limit: int) -> list[KnowledgeSection]:
    terms = [term for term in _normalize(query).split() if len(term) >= 3]
    if not terms:
        return sections[:limit]

    scored: list[tuple[int, KnowledgeSection]] = []
    for section in sections:
        title = _normalize(section.title)
        text = _normalize(section.content)
        score = 0
        for term in terms:
            if term in title:
                score += 5
            score += text.count(term)
        if score:
            scored.append((score, section))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [section for _, section in scored[:limit]]


def _title_from_markdown(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _first_paragraph(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if lines:
                break
            continue
        lines.append(stripped)
    return _preview(" ".join(lines), max_chars=220)


def _preview(content: str, *, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _slugify(value: str) -> str:
    normalized = _normalize(value)
    slug = re.sub(r"[^0-9a-zа-яё]+", "-", normalized, flags=re.IGNORECASE).strip("-")
    return slug or "section"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
