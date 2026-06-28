from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class ScenarioCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ScenarioRun:
    id: str
    title: str
    ok: bool
    checks: list[ScenarioCheck] = field(default_factory=list)
    status: str = ""
    answer: str = ""
    handoff_to: list[str] = field(default_factory=list)
    learning_event_id: str = ""
    trace_id: str = ""
    incident_ids: list[str] = field(default_factory=list)
    diagnostic_answer: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "ok": self.ok,
            "status": self.status,
            "answer": self.answer,
            "handoff_to": self.handoff_to,
            "learning_event_id": self.learning_event_id,
            "trace_id": self.trace_id,
            "incident_ids": self.incident_ids,
            "diagnostic_answer": self.diagnostic_answer,
            "error": self.error,
            "checks": [check.__dict__ for check in self.checks],
        }


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.scenarios)
    report = run_scenarios(
        config,
        base_url=args.base_url,
        secret=args.secret,
        limit=args.limit,
        timeout_seconds=args.timeout,
    )
    _print_report(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport saved: {output_path}")
    return 0 if report["ok"] else 1


def run_scenarios(
    config: dict[str, Any],
    *,
    base_url: str = DEFAULT_BASE_URL,
    secret: str = "",
    limit: int = 0,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    scenarios = list(config.get("scenarios") or [])
    if limit > 0:
        scenarios = scenarios[:limit]
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    headers = {"X-Agent-Secret": secret} if secret else {}
    results: list[ScenarioRun] = []

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds, headers=headers) as client:
        for scenario in scenarios:
            results.append(_run_one(client, scenario, defaults))

    return {
        "ok": all(item.ok for item in results),
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "total": len(results),
        "passed": sum(1 for item in results if item.ok),
        "failed": sum(1 for item in results if not item.ok),
        "results": [item.as_dict() for item in results],
    }


def _run_one(client: httpx.Client, scenario: dict[str, Any], defaults: dict[str, Any]) -> ScenarioRun:
    scenario_id = str(scenario.get("id") or "unnamed")
    run = ScenarioRun(id=scenario_id, title=str(scenario.get("title") or scenario_id), ok=False)
    text = str(scenario.get("text") or "").strip()
    if not text:
        run.error = "Scenario text is empty"
        return run

    try:
        response = client.post(
            "/orchestrator/test",
            json={
                "text": text,
                "user_id": scenario.get("user_id", defaults.get("user_id")),
                "channel": scenario.get("channel", defaults.get("channel", "local_test")),
                "dialog_id": scenario.get("dialog_id", defaults.get("dialog_id", "scenario-test")),
            },
        )
        response.raise_for_status()
        result = response.json()
        run.status = str(result.get("status") or "")
        run.answer = str(result.get("answer") or "")
        run.handoff_to = _strings(result.get("handoff_to"))

        learning_event = _find_learning_event(client, request_text=text, response_text=run.answer)
        if learning_event:
            run.learning_event_id = str(learning_event.get("id") or "")
            metadata = learning_event.get("metadata") if isinstance(learning_event.get("metadata"), dict) else {}
            run.trace_id = str(metadata.get("trace_id") or "")

        trace_events = _trace_events(client, run.trace_id) if run.trace_id else []
        run.checks = _evaluate_expected(
            expected=scenario.get("expected") if isinstance(scenario.get("expected"), dict) else {},
            result=result,
            learning_event=learning_event,
            trace_events=trace_events,
        )

        feedback = scenario.get("feedback") if isinstance(scenario.get("feedback"), dict) else {}
        if feedback:
            run.checks.extend(_submit_feedback(client, run, feedback))

        run.ok = bool(run.checks) and all(check.ok for check in run.checks)
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
        run.checks.append(ScenarioCheck(name="scenario_error", ok=False, detail=run.error))
    return run


def _evaluate_expected(
    *,
    expected: dict[str, Any],
    result: dict[str, Any],
    learning_event: dict[str, Any] | None,
    trace_events: list[dict[str, Any]],
) -> list[ScenarioCheck]:
    checks: list[ScenarioCheck] = []
    status = str(result.get("status") or "")
    answer = str(result.get("answer") or "")
    handoff_to = _strings(result.get("handoff_to"))
    action_names = [str(item.get("name") or "") for item in _dicts(result.get("actions_taken"))]

    if expected.get("status"):
        wanted = str(expected["status"])
        checks.append(ScenarioCheck("status", status == wanted, f"actual={status} expected={wanted}"))
    if expected.get("status_any"):
        wanted = _strings(expected["status_any"])
        checks.append(ScenarioCheck("status_any", status in wanted, f"actual={status} expected_any={wanted}"))
    if expected.get("handoff_to_any"):
        wanted = _strings(expected["handoff_to_any"])
        checks.append(
            ScenarioCheck("handoff_to_any", bool(set(handoff_to).intersection(wanted)), f"actual={handoff_to}")
        )
    if expected.get("handoff_to_all"):
        wanted = _strings(expected["handoff_to_all"])
        checks.append(ScenarioCheck("handoff_to_all", all(item in handoff_to for item in wanted), f"actual={handoff_to}"))
    if expected.get("answer_contains_any"):
        wanted = _strings(expected["answer_contains_any"])
        folded = answer.casefold()
        checks.append(
            ScenarioCheck(
                "answer_contains_any",
                any(item.casefold() in folded for item in wanted),
                f"expected_any={wanted}",
            )
        )
    if expected.get("answer_contains_all"):
        wanted = _strings(expected["answer_contains_all"])
        folded = answer.casefold()
        checks.append(
            ScenarioCheck(
                "answer_contains_all",
                all(item.casefold() in folded for item in wanted),
                f"expected_all={wanted}",
            )
        )
    if expected.get("actions_include_any"):
        wanted = _strings(expected["actions_include_any"])
        checks.append(
            ScenarioCheck("actions_include_any", bool(set(action_names).intersection(wanted)), f"actual={action_names}")
        )
    if expected.get("trace_events_any"):
        wanted = _strings(expected["trace_events_any"])
        event_names = [str(item.get("event_name") or "") for item in trace_events]
        checks.append(
            ScenarioCheck("trace_events_any", bool(set(event_names).intersection(wanted)), f"actual={event_names}")
        )
    if expected.get("learning_event") is not False:
        checks.append(
            ScenarioCheck(
                "learning_event_recorded",
                learning_event is not None,
                "found" if learning_event else "not found in /learning/events",
            )
        )
    return checks or [ScenarioCheck("no_expectations", True, "no expected checks configured")]


def _submit_feedback(client: httpx.Client, run: ScenarioRun, feedback: dict[str, Any]) -> list[ScenarioCheck]:
    checks: list[ScenarioCheck] = []
    if not run.learning_event_id:
        return [ScenarioCheck("feedback", False, "learning_event_id is empty")]

    response = client.post(
        "/learning/feedback",
        json={
            "event_id": run.learning_event_id,
            "rating": feedback.get("rating"),
            "rating_scale": feedback.get("rating_scale"),
            "outcome": feedback.get("outcome", ""),
            "corrected_answer": feedback.get("corrected_answer", ""),
            "comment": feedback.get("comment", ""),
            "tags": feedback.get("tags") or [],
            "user_id": feedback.get("user_id"),
            "channel": feedback.get("channel", "scenario_runner"),
        },
    )
    checks.append(ScenarioCheck("feedback_status", response.status_code == 200, f"status_code={response.status_code}"))
    if response.status_code != 200:
        return checks
    payload = response.json()
    checks.append(ScenarioCheck("feedback_recorded", payload.get("recorded") is True, str(payload)))

    incident = payload.get("incident") if isinstance(payload.get("incident"), dict) else {}
    if incident.get("event_id"):
        run.incident_ids.append(str(incident["event_id"]))
        checks.append(ScenarioCheck("incident_created", True, str(incident["event_id"])))
    elif _feedback_expects_incident(feedback):
        checks.append(ScenarioCheck("incident_created", False, "feedback did not create incident"))

    if feedback.get("diagnose"):
        diagnostic = client.post(
            "/learning/diagnose",
            json={"event_id": run.learning_event_id, "comment": feedback.get("comment", "")},
        )
        checks.append(
            ScenarioCheck("diagnose_status", diagnostic.status_code == 200, f"status_code={diagnostic.status_code}")
        )
        if diagnostic.status_code == 200:
            diagnostic_payload = diagnostic.json()
            run.diagnostic_answer = str(diagnostic_payload.get("answer") or "")
            checks.append(
                ScenarioCheck(
                    "diagnostic_answer",
                    bool(run.diagnostic_answer.strip()),
                    run.diagnostic_answer[:240],
                )
            )
    return checks


def _find_learning_event(client: httpx.Client, *, request_text: str, response_text: str) -> dict[str, Any] | None:
    response = client.get("/learning/events", params={"limit": 100})
    if response.status_code != 200:
        return None
    for event in reversed(response.json().get("events") or []):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "agent_result":
            continue
        if str(event.get("request") or "") != request_text:
            continue
        if response_text and str(event.get("response") or "") != response_text:
            continue
        return event
    return None


def _trace_events(client: httpx.Client, trace_id: str) -> list[dict[str, Any]]:
    response = client.get("/learning/traces", params={"trace_id": trace_id, "limit": 500})
    if response.status_code != 200:
        return []
    events = response.json().get("events") or []
    return [event for event in events if isinstance(event, dict)]


def _feedback_expects_incident(feedback: dict[str, Any]) -> bool:
    outcome = str(feedback.get("outcome") or "").strip()
    if outcome:
        return outcome in {"not_done", "not_completed", "failed", "incorrect", "bad_result", "не_выполнено", "ошибка"}
    rating = feedback.get("rating")
    rating_scale = feedback.get("rating_scale")
    return isinstance(rating, int) and isinstance(rating_scale, int) and rating <= max(1, rating_scale // 2)


def _print_report(report: dict[str, Any]) -> None:
    print(f"Scenario run: {report['passed']}/{report['total']} passed")
    for item in report["results"]:
        mark = "OK" if item["ok"] else "FAIL"
        print(f"\n[{mark}] {item['id']} - {item['title']}")
        print(f"status={item['status']} handoff_to={item['handoff_to']}")
        if item.get("learning_event_id"):
            print(f"learning_event_id={item['learning_event_id']} trace_id={item.get('trace_id') or ''}")
        if item.get("incident_ids"):
            print(f"incident_ids={item['incident_ids']}")
        if item.get("diagnostic_answer"):
            print(f"diagnostic={item['diagnostic_answer'][:300]}")
        if item.get("error"):
            print(f"error={item['error']}")
        for check in item["checks"]:
            check_mark = "OK" if check["ok"] else "FAIL"
            print(f"  - {check_mark} {check['name']}: {check['detail']}")


def _load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("Scenario file must contain a YAML object")
    return payload


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI-server work scenarios through the public API.")
    parser.add_argument(
        "--scenarios",
        default="tests/scenarios/work_scenarios.yaml",
        help="Path to YAML scenario file.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("AI_SERVER_SCENARIO_BASE_URL", DEFAULT_BASE_URL),
        help="AI-server base URL.",
    )
    parser.add_argument(
        "--secret",
        default=os.getenv("AI_SERVER_SCENARIO_SECRET", ""),
        help="Webhook/agent secret for protected learning endpoints.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N scenarios.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
