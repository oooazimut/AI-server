from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from ai_server.models import AgentManifest
from ai_server.registry import resolve_project_path
from ai_server.skills import SkillStore

CAPABILITY_REGISTRY_SCHEMA = "specialist.capabilities.v1"


def build_capability_registry(
    manifest: AgentManifest,
    tool_definitions: list[dict[str, Any]],
    *,
    structured_tool_names: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Build a fresh machine-readable specialist contract.

    The registry is intentionally rebuilt from the live specialist instance and
    the skill files on disk.  The orchestrator therefore never owns a stale,
    manually copied list of specialist tools.
    """

    structured = set(structured_tool_names or ())
    tools: list[dict[str, Any]] = []
    for raw in sorted(tool_definitions, key=lambda item: str(item.get("name") or "")):
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
        tool_contract = {
            "id": name,
            "description": str(raw.get("description") or ""),
            "parameters": parameters,
            "structured_command": name in structured,
        }
        tool_contract["version"] = _contract_hash(tool_contract)
        tools.append(tool_contract)

    skills = [
        {
            "id": skill.id,
            "title": skill.title,
            "content": str(skill.content or ""),
        }
        for skill in SkillStore().list_skills_with_content(manifest)
    ]
    contracts: list[dict[str, Any]] = []
    contracts_path = resolve_project_path(manifest.contracts_path)
    if contracts_path is not None and contracts_path.exists():
        for path in sorted(contracts_path.glob("*.yaml")):
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                contracts.append({"id": path.stem, "content": value})
    payload = {
        "schema_version": CAPABILITY_REGISTRY_SCHEMA,
        "specialist_id": manifest.id,
        "specialist_version": manifest.version,
        "description": str(manifest.handoff_description or manifest.description or manifest.name),
        "tools": tools,
        "skills": skills,
        "contracts": contracts,
        "allowed_actions": sorted(manifest.allowed_actions),
        "approval_required": sorted(manifest.approval_required),
    }
    return {**payload, "registry_version": _contract_hash(payload)}


def validate_tool_arguments(parameters: dict[str, Any], arguments: object) -> list[str]:
    """Validate the JSON-schema subset used by project tool definitions.

    Tool schemas are deliberately small.  Keeping validation local avoids a
    runtime dependency while still failing closed on missing, unknown, wrongly
    typed, out-of-range, or enum-constrained arguments.
    """

    errors: list[str] = []
    _validate_schema(arguments, parameters or {"type": "object"}, path="arguments", errors=errors, strict=True)
    return errors


def registry_tool(registry: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    for item in registry.get("tools") or []:
        if isinstance(item, dict) and str(item.get("id") or "") == tool_name:
            return item
    return None


def _contract_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_schema(
    value: object,
    schema: dict[str, Any],
    *,
    path: str,
    errors: list[str],
    strict: bool = False,
) -> None:
    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        errors.append(f"{path}: expected {expected}")
        return

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value is not in enum")
        return

    if isinstance(value, dict):
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: required")

        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if strict and properties:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key}: unknown argument")
        for key, item in value.items():
            child = properties.get(key)
            if isinstance(child, dict):
                _validate_schema(item, child, path=f"{path}.{key}", errors=errors)

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], path=f"{path}[{index}]", errors=errors)

    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: below minimum {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: above maximum {maximum}")

    for child in schema.get("allOf") or []:
        if isinstance(child, dict):
            _validate_schema(value, child, path=path, errors=errors)

    any_of = [child for child in schema.get("anyOf") or [] if isinstance(child, dict)]
    if any_of and not any(_schema_matches(value, child, path=path) for child in any_of):
        errors.append(f"{path}: no anyOf contract matched")

    one_of = [child for child in schema.get("oneOf") or [] if isinstance(child, dict)]
    if one_of:
        matches = sum(1 for child in one_of if _schema_matches(value, child, path=path))
        if matches != 1:
            errors.append(f"{path}: expected exactly one oneOf contract")


def _schema_matches(value: object, schema: dict[str, Any], *, path: str) -> bool:
    candidate_errors: list[str] = []
    _validate_schema(value, schema, path=path, errors=candidate_errors)
    return not candidate_errors


def _matches_type(value: object, expected: object) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, item) for item in expected)
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(str(expected), True)
