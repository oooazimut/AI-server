import json

import httpx

from scripts.work_scenario_runner import run_scenarios

_RealClient = httpx.Client


def test_work_scenario_runner_passes_basic_scenario(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/orchestrator/test":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "answer": "Нашел инструкцию TL-WR820N",
                    "handoff_to": ["secure_org_data"],
                    "actions_taken": [{"name": "delegate_to_specialist"}],
                },
            )
        if request.url.path == "/learning/events":
            return httpx.Response(
                200,
                json={
                    "events": [
                        {
                            "id": "event-1",
                            "event_type": "agent_result",
                            "request": "Найди инструкцию TL-WR820N",
                            "response": "Нашел инструкцию TL-WR820N",
                            "metadata": {"trace_id": "trace-1"},
                        }
                    ],
                },
            )
        if request.url.path == "/learning/traces":
            return httpx.Response(200, json={"events": [{"event_name": "orchestrator_decision"}]})
        return httpx.Response(404)

    monkeypatch.setattr(
        "scripts.work_scenario_runner.httpx.Client",
        lambda **kwargs: _RealClient(transport=httpx.MockTransport(handler), **kwargs),
    )

    report = run_scenarios(
        {
            "scenarios": [
                {
                    "id": "secure",
                    "text": "Найди инструкцию TL-WR820N",
                    "expected": {
                        "status": "completed",
                        "handoff_to_any": ["secure_org_data"],
                        "answer_contains_any": ["TL-WR820N"],
                        "trace_events_any": ["orchestrator_decision"],
                    },
                }
            ]
        }
    )

    assert report["ok"] is True
    assert report["passed"] == 1
    assert any(request.url.path == "/orchestrator/test" for request in requests)


def test_work_scenario_runner_creates_feedback_incident_and_diagnosis(monkeypatch):
    events = [
        {
            "id": "event-2",
            "event_type": "agent_result",
            "request": "Найди датчик коленвала на складе",
            "response": "Не найдено",
            "metadata": {"trace_id": "trace-2"},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/orchestrator/test":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "answer": "Не найдено",
                    "handoff_to": ["bitrix24"],
                    "actions_taken": [{"name": "delegate_to_specialist"}],
                },
            )
        if request.url.path == "/learning/events":
            return httpx.Response(200, json={"events": events})
        if request.url.path == "/learning/traces":
            return httpx.Response(200, json={"events": []})
        if request.url.path == "/learning/feedback":
            return httpx.Response(
                200,
                json={
                    "recorded": True,
                    "event_id": "feedback-1",
                    "incident": {"recorded": True, "event_id": "incident-1"},
                },
            )
        if request.url.path == "/learning/diagnose":
            assert request.read()
            body = json.loads(request.content)
            assert body["feedback_event_id"] == "feedback-1"
            events.append(
                {
                    "id": "diagnostic-1",
                    "event_type": "diagnostic_report",
                    "metadata": {
                        "target_event_id": "event-2",
                        "feedback_event_ids": ["feedback-1"],
                    },
                }
            )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "answer": "Проблема вероятно в catalog.",
                    "diagnostic_event": {"recorded": True, "event_id": "diagnostic-1"},
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        "scripts.work_scenario_runner.httpx.Client",
        lambda **kwargs: _RealClient(transport=httpx.MockTransport(handler), **kwargs),
    )

    report = run_scenarios(
        {
            "scenarios": [
                {
                    "id": "incident",
                    "text": "Найди датчик коленвала на складе",
                    "expected": {"status": "completed", "handoff_to_any": ["bitrix24"]},
                    "feedback": {
                        "rating": 3,
                        "rating_scale": 10,
                        "outcome": "not_completed",
                        "comment": "товар есть",
                        "tags": ["catalog"],
                        "diagnose": True,
                    },
                }
            ]
        }
    )

    result = report["results"][0]
    assert report["ok"] is True
    assert result["feedback_event_id"] == "feedback-1"
    assert result["incident_ids"] == ["incident-1"]
    assert result["diagnostic_report_id"] == "diagnostic-1"
    assert result["diagnostic_answer"] == "Проблема вероятно в catalog."
