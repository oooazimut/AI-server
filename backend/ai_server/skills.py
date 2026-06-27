from pathlib import Path

from pydantic import BaseModel

from .models import AgentManifest
from .registry import agent_package_path, resolve_project_path


class Skill(BaseModel):
    id: str
    title: str
    path: str
    preview: str
    content: str | None = None


class SkillStore:
    def list_skills(self, manifest: AgentManifest) -> list[Skill]:
        directory = self._skills_path(manifest)
        if not directory.exists():
            return []
        return [self._read_skill(path, include_content=False) for path in sorted(directory.glob("*.md"))]

    def list_skills_with_content(self, manifest: AgentManifest) -> list[Skill]:
        directory = self._skills_path(manifest)
        if not directory.exists():
            return []
        return [self._read_skill(path, include_content=True) for path in sorted(directory.glob("*.md"))]

    def read_skill(self, manifest: AgentManifest, skill_id: str) -> Skill | None:
        directory = self._skills_path(manifest)
        path = directory / f"{skill_id}.md"
        if not path.exists():
            return None
        return self._read_skill(path, include_content=True)

    def _skills_path(self, manifest: AgentManifest) -> Path:
        configured = resolve_project_path(manifest.skills_path)
        return configured or agent_package_path(manifest.id) / "skills"

    def _read_skill(self, path: Path, *, include_content: bool) -> Skill:
        content = path.read_text(encoding="utf-8").strip()
        title = _title_from_markdown(content) or path.stem.replace("_", " ")
        preview = _preview(content)
        return Skill(
            id=path.stem,
            title=title,
            path=str(path),
            preview=preview,
            content=content if include_content else None,
        )


def _title_from_markdown(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _preview(content: str, *, max_chars: int = 240) -> str:
    text = " ".join(content.split())
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."
