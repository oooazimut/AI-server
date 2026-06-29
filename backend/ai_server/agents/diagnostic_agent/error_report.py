from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any


class ErrorReportService:
    """Read-only error report builder owned by Diagnostic Agent."""

    def __init__(self, learning_recorder: Any) -> None:
        self._learning_recorder = learning_recorder

    def build(self, *, since_hours: int = 24, limit: int = 200) -> dict[str, Any]:
        since_hours = max(1, min(int(since_hours or 24), 24 * 30))
        limit = max(1, min(int(limit or 200), 1000))
        since_at = datetime.now(UTC) - timedelta(hours=since_hours)
        events = [
            event
            for event in self._learning_recorder.latest(limit=limit)
            if _created_at(event) is None or _created_at(event) >= since_at
        ]
        incidents = [event for event in events if event.get("event_type") == "incident"]
        reports = [event for event in events if event.get("event_type") == "diagnostic_report"]
        reports_by_incident = _reports_by_incident(reports)

        groups: dict[str, dict[str, Any]] = {}
        for incident in incidents:
            keys = _incident_group_keys(incident)
            for key in keys:
                group = groups.setdefault(
                    key,
                    {
                        "key": key,
                        "title": _group_title(key),
                        "count": 0,
                        "incident_ids": [],
                        "examples": [],
                        "diagnostic_reports": [],
                    },
                )
                group["count"] += 1
                group["incident_ids"].append(incident.get("id"))
                if len(group["examples"]) < 3:
                    group["examples"].append(_incident_example(incident))

        for group in groups.values():
            summaries = _linked_report_summaries(group["incident_ids"], reports_by_incident)
            group["diagnostic_reports"] = summaries
            first = summaries[0] if summaries else {}
            if first.get("where_to_fix"):
                group["where_to_fix"] = first["where_to_fix"]
            if first.get("fix_proposal"):
                group["fix_proposal"] = first["fix_proposal"]
            if first.get("regression_test"):
                group["regression_test"] = first["regression_test"]

        ordered = sorted(groups.values(), key=lambda item: (-int(item["count"]), str(item["key"])))
        by_agent = Counter(
            str((event.get("metadata") or {}).get("target_agent_id") or event.get("agent_id") or "unknown")
            for event in incidents
        )
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "since_hours": since_hours,
            "limit": limit,
            "total_incidents": len(incidents),
            "total_diagnostic_reports": len(reports),
            "by_target_agent": dict(by_agent),
            "groups": ordered,
        }

    def markdown(self, *, since_hours: int = 24, limit: int = 200, max_groups: int = 5) -> str:
        return format_error_report_markdown(
            self.build(since_hours=since_hours, limit=limit),
            max_groups=max_groups,
        )


def format_error_report_markdown(report: dict[str, Any], *, max_groups: int = 5) -> str:
    lines = [
        f"Отчет Диагноста по ошибкам за {report.get('since_hours')} ч.",
        "",
        f"Incidents: {report.get('total_incidents', 0)}",
        f"Diagnostic reports: {report.get('total_diagnostic_reports', 0)}",
    ]
    groups = report.get("groups") if isinstance(report.get("groups"), list) else []
    if not groups:
        return "\n".join([*lines, "", "Повторяющихся ошибок за период не найдено."])

    lines.append("")
    for index, group in enumerate(groups[: max(1, max_groups)], start=1):
        lines.append(f"{index}. {group.get('title') or group.get('key')} - {group.get('count')} случаев")
        example = (group.get("examples") or [{}])[0]
        if example.get("request"):
            lines.append(f"   Пример: {example['request']}")
        if example.get("feedback_comment"):
            lines.append(f"   Feedback: {example['feedback_comment']}")
        if group.get("where_to_fix"):
            lines.append(f"   Где править: {group['where_to_fix']}")
        if group.get("fix_proposal"):
            lines.append(f"   Что исправить: {group['fix_proposal']}")
        if group.get("regression_test"):
            lines.append(f"   Regression test: {group['regression_test']}")
        reports = group.get("diagnostic_reports") or []
        if reports:
            lines.append(f"   diagnostic_report: {reports[0].get('id')}")
        lines.append("")
    return "\n".join(lines).strip()


def _created_at(event: dict[str, Any]) -> datetime | None:
    raw = str(event.get("created_at") or "")
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _incident_group_keys(event: dict[str, Any]) -> list[str]:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    trace = metadata.get("diagnostic_trace") if isinstance(metadata.get("diagnostic_trace"), dict) else {}
    keys: list[str] = []
    for prefix, value in (
        ("incident_reason", metadata.get("reason")),
        ("target_agent", metadata.get("target_agent_id")),
        ("target_status", metadata.get("target_status")),
    ):
        if value:
            keys.append(f"{prefix}:{value}")
    for agent_id in _strings(trace.get("called_agents")):
        keys.append(f"target_agent:{agent_id}")
    for rule in _list_of_dicts(trace.get("loaded_rules")):
        if rule.get("id"):
            keys.append(f"loaded_rule:{rule['id']}")
    for skill in _list_of_dicts(trace.get("loaded_skills")):
        if skill.get("id"):
            keys.append(f"loaded_skill:{skill['id']}")
    for tool_call in _list_of_dicts(trace.get("tool_calls")):
        name = str(tool_call.get("name") or "")
        if name and name != "none":
            keys.append(f"tool_call:{name}")
    for tag in _strings(metadata.get("tags")):
        keys.append(f"tag:{tag}")
    return sorted(set(keys)) or ["ungrouped"]


def _incident_example(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    return {
        "id": event.get("id"),
        "target_event_id": metadata.get("target_event_id"),
        "feedback_event_id": metadata.get("feedback_event_id"),
        "reason": metadata.get("reason"),
        "request": _short(event.get("request")),
        "response": _short(event.get("response")),
        "feedback_comment": _short(metadata.get("comment")),
        "rating": metadata.get("rating"),
        "rating_scale": metadata.get("rating_scale"),
    }


def _reports_by_incident(reports: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        incident_ids = _strings(metadata.get("incident_event_ids"))
        incident_id = str(metadata.get("incident_event_id") or "")
        if incident_id:
            incident_ids.append(incident_id)
        for item in sorted(set(incident_ids)):
            result.setdefault(item, []).append(report)
    return result


def _linked_report_summaries(
    incident_ids: list[Any],
    reports_by_incident: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for incident_id in _strings(incident_ids):
        for report in reports_by_incident.get(incident_id, []):
            report_id = str(report.get("id") or "")
            if not report_id or report_id in seen:
                continue
            seen.add(report_id)
            summaries.append(_diagnostic_report_summary(report))
            if len(summaries) >= 3:
                return summaries
    return summaries


def _diagnostic_report_summary(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    sections = _diagnostic_report_sections(str(event.get("response") or ""))
    summary = {
        "id": event.get("id"),
        "target_event_id": metadata.get("target_event_id"),
        "feedback_event_ids": _strings(metadata.get("feedback_event_ids")),
        "incident_event_ids": _strings(metadata.get("incident_event_ids")),
        "status": event.get("status"),
    }
    summary.update(sections)
    if not sections and event.get("response"):
        summary["excerpt"] = _short(event.get("response"), max_chars=800)
    return summary


def _diagnostic_report_sections(response: str) -> dict[str, str]:
    markers = {
        "problem": ("что пошло не так", "problem", "what went wrong"),
        "where_to_fix": ("где", "where"),
        "fix_proposal": ("что исправить", "fix", "proposal"),
        "regression_test": ("regression", "регрес", "test", "тест"),
    }
    result: dict[str, str] = {}
    blocks = [block.strip() for block in response.split("\n\n") if block.strip()]
    for block in blocks:
        normalized = block.casefold()
        body = block
        if normalized.startswith("**") and ":**" in block:
            title, body = block.split(":**", 1)
            normalized = title.strip("* ").casefold()
        for key, values in markers.items():
            if key not in result and any(marker in normalized for marker in values):
                result[key] = _short(body.strip())
                break
    return result


def _group_title(key: str) -> str:
    prefix, _, value = key.partition(":")
    titles = {
        "incident_reason": "Причина feedback",
        "target_agent": "Агент",
        "target_status": "Статус ответа",
        "loaded_rule": "Правило",
        "loaded_skill": "Skill",
        "tool_call": "Tool",
        "tag": "Тег feedback",
    }
    return f"{titles.get(prefix, 'Группа')} `{value}`" if value else key


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, list) else []


def _short(value: Any, *, max_chars: int = 300) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ...[short]"
