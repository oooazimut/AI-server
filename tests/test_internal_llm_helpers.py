import asyncio
import json

import pytest

from ai_server.models import AgentTask, ToolResult
from ai_server.orchestrators.orchestrator_llm import (
    OrchestratorDecision,
    OrchestratorLLMService,
    _parse_decision,
    _status,
)
from ai_server.registry import get_agent_manifest
from tests.fakes import RecordingLLMClient


def _tool_defs(*names: str) -> list[dict]:
    return [{"name": n, "description": ""} for n in names]


# _parse_decision
def test_parse_decision_valid_tool_call():
    defs = _tool_defs("call_bitrix24", "call_pto")
    data = {
        "status": "completed",
        "answer": "Передаю в битрикс",
        "tool_calls": [{"name": "call_bitrix24", "args": {"request": "создай задачу"}, "summary": ""}],
        "confidence": 0.9,
    }
    decision = _parse_decision(data, defs)
    assert decision.status == "completed"
    assert decision.answer == "Передаю в битрикс"
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "call_bitrix24"
    assert decision.tool_calls[0].args == {"request": "создай задачу"}
    assert decision.confidence == pytest.approx(0.9)


def test_parse_decision_filters_unknown_tools():
    defs = _tool_defs("call_bitrix24")
    data = {
        "status": "completed",
        "answer": "",
        "tool_calls": [
            {"name": "call_bitrix24", "args": {}},
            {"name": "call_unknown", "args": {}},
        ],
        "confidence": 0.8,
    }
    decision = _parse_decision(data, defs)
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "call_bitrix24"


def test_parse_decision_none_tool_always_allowed():
    defs = _tool_defs("call_bitrix24")
    data = {
        "status": "completed",
        "answer": "Отвечу сам",
        "tool_calls": [{"name": "none", "args": {}}],
        "confidence": 0.7,
    }
    decision = _parse_decision(data, defs)
    assert decision.tool_calls[0].name == "none"


def test_parse_decision_empty_tool_calls_defaults_to_none():
    defs = _tool_defs("call_bitrix24")
    data = {"status": "completed", "answer": "Ответ", "tool_calls": [], "confidence": 0.5}
    decision = _parse_decision(data, defs)
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "none"


def test_parse_decision_missing_tool_calls_defaults_to_none():
    defs = _tool_defs("call_bitrix24")
    data = {"status": "completed", "answer": "Ответ", "confidence": 0.5}
    decision = _parse_decision(data, defs)
    assert decision.tool_calls[0].name == "none"


def test_parse_decision_invalid_status_defaults_to_completed():
    defs = _tool_defs("call_bitrix24")
    data = {"status": "nonsense", "answer": "", "tool_calls": [], "confidence": 0.5}
    decision = _parse_decision(data, defs)
    assert decision.status == "completed"


def test_parse_decision_invalid_confidence_defaults_to_half():
    defs = _tool_defs("call_bitrix24")
    data = {"status": "completed", "answer": "", "tool_calls": [], "confidence": "bad"}
    decision = _parse_decision(data, defs)
    assert decision.confidence == pytest.approx(0.5)


def test_parse_decision_multiple_tool_calls():
    defs = _tool_defs("call_bitrix24", "call_pto")
    data = {
        "status": "completed",
        "answer": "",
        "tool_calls": [
            {"name": "call_bitrix24", "args": {"request": "задача"}},
            {"name": "call_pto", "args": {"request": "документ"}},
        ],
        "confidence": 0.85,
    }
    decision = _parse_decision(data, defs)
    assert len(decision.tool_calls) == 2
    assert {tc.name for tc in decision.tool_calls} == {"call_bitrix24", "call_pto"}


# _status
@pytest.mark.parametrize(
    "value,expected",
    [
        ("completed", "completed"),
        ("needs_clarification", "needs_clarification"),
        ("failed", "failed"),
        ("nonsense", "completed"),
        (None, "completed"),
        ("", "completed"),
        ("needs_human", "completed"),  # not a valid orchestrator status
    ],
)
def test_status_normalization(value, expected):
    assert _status(value) == expected


def test_orchestrator_decide_clarifies_ambiguous_admin_panel_without_llm():
    client = RecordingLLMClient('{"status":"completed","answer":"wrong","tool_calls":[{"name":"none","args":{}}]}')
    service = OrchestratorLLMService(client)

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("internal_orchestrator"),
            task=AgentTask(task_id="t1", request="Покажи список операторов"),
            dialog_history=[
                {"role": "assistant", "content": "Настройки контроля закрытия задач:\n- Автозакрытие: 20:00"}
            ],
            retrieval_hits=[],
            tool_definitions=_tool_defs("call_specialist"),
            tool_results=[],
        )
    )

    assert client.calls == []
    assert result.decision.status == "needs_clarification"
    assert result.decision.answer == "Уточните, какую панель показать: закрытие задач или отчет по машинам и людям."
    assert [call.name for call in result.decision.tool_calls] == ["none"]


def test_orchestrator_decide_keeps_explicit_vehicle_admin_panel_for_llm():
    client = RecordingLLMClient('{"status":"completed","answer":"","tool_calls":[{"name":"none","args":{}}]}')
    service = OrchestratorLLMService(client)

    result = asyncio.run(
        service.decide(
            manifest=get_agent_manifest("internal_orchestrator"),
            task=AgentTask(task_id="t1", request="Покажи список операторов отчета по машинам и людям"),
            retrieval_hits=[],
            tool_definitions=_tool_defs("call_specialist"),
            tool_results=[],
        )
    )

    assert client.calls
    assert [call.name for call in result.decision.tool_calls] == ["none"]


def test_orchestrator_compose_passes_through_specialist_answer_without_llm():
    client = RecordingLLMClient(json.dumps({"answer": "wrong rewrite", "status": "completed"}))
    service = OrchestratorLLMService(client)

    result = asyncio.run(
        service.compose(
            manifest=get_agent_manifest("internal_orchestrator"),
            task=AgentTask(task_id="t1", request="Битрикс покажи задачи"),
            decision=OrchestratorDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="call_specialist",
                    data={
                        "specialist": "bitrix24",
                        "answer": "Готовый ответ Bitrix specialist",
                        "status": "completed",
                    },
                )
            ],
        )
    )

    assert result.answer == "Готовый ответ Bitrix specialist"
    assert result.status == "completed"
    assert client.calls == []
