from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_server.registry import PROJECT_ROOT


DEFAULT_RESULT_TEMPLATES_PATH = PROJECT_ROOT / "config" / "result_templates.example.json"


def load_result_templates(path: Path | str | None = None) -> dict[str, Any]:
    selected = Path(path) if path is not None else DEFAULT_RESULT_TEMPLATES_PATH
    if not selected.exists():
        return {"templates": [], "bindings": []}
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"templates": [], "bindings": []}
    if not isinstance(payload, dict):
        return {"templates": [], "bindings": []}
    templates = payload.get("templates") if isinstance(payload.get("templates"), list) else []
    bindings = payload.get("bindings") if isinstance(payload.get("bindings"), list) else []
    return {"templates": templates, "bindings": bindings}


def active_result_templates_context() -> dict[str, Any]:
    catalog = load_result_templates()
    templates = [
        template
        for template in catalog["templates"]
        if isinstance(template, dict) and str(template.get("status") or "active") == "active"
    ]
    return {"templates": templates, "bindings": catalog["bindings"]}
