import pytest

from ai_server.models import AgentManifest
from ai_server.orchestrators.internal_llm import (
    _parse_decision,
    _specialist_routing_hints,
    _status,
)


def _fake_manifest(
    agent_id: str,
    *,
    kind: str = "specialist",
    handoff_description: str = "",
) -> AgentManifest:
    return AgentManifest(
        id=agent_id,
        name=agent_id.capitalize(),
        kind=kind,
        description="Test manifest",
        handoff_description=handoff_description,
    )


# _specialist_routing_hints
def test_routing_hints_generated_from_specialist_manifests():
    manifests = [
        _fake_manifest("bitrix24", handoff_description="Bitrix запросы и задачи"),
        _fake_manifest("pto", handoff_description="ПТО документы"),
    ]
    hints = _specialist_routing_hints(manifests)
    assert "bitrix24" in hints
    assert "pto" in hints
    assert "Bitrix запросы" in hints
    assert "ПТО документы" in hints


def test_routing_hints_skips_non_specialists():
    manifests = [
        _fake_manifest("internal_orchestrator", kind="orchestrator", handoff_description="Не должен попасть"),
        _fake_manifest("bitrix24", handoff_description="Bitrix"),
    ]
    hints = _specialist_routing_hints(manifests)
    assert "internal_orchestrator" not in hints
    assert "Не должен попасть" not in hints


def test_routing_hints_skips_empty_handoff_description():
    manifests = [
        _fake_manifest("bitrix24", handoff_description=""),
        _fake_manifest("pto", handoff_description="ПТО"),
    ]
    hints = _specialist_routing_hints(manifests)
    assert "bitrix24" not in hints
    assert "pto" in hints


def test_routing_hints_empty_manifests():
    assert _specialist_routing_hints([]) == ""


def test_routing_hints_no_specialists_with_description():
    manifests = [
        _fake_manifest("bitrix24", handoff_description=""),
        _fake_manifest("pto", handoff_description=""),
    ]
    assert _specialist_routing_hints(manifests) == ""


# _parse_decision
def test_parse_decision_valid():
    manifests = [_fake_manifest("bitrix24"), _fake_manifest("pto")]
    data = {"status": "completed", "answer": "Привет", "handoff_to": ["bitrix24"], "confidence": 0.9}
    decision = _parse_decision(data, manifests)
    assert decision.status == "completed"
    assert decision.answer == "Привет"
    assert decision.handoff_to == ["bitrix24"]
    assert decision.confidence == pytest.approx(0.9)


def test_parse_decision_filters_unknown_specialists():
    manifests = [_fake_manifest("bitrix24")]
    data = {"status": "completed", "answer": "", "handoff_to": ["bitrix24", "unknown_agent"], "confidence": 0.8}
    decision = _parse_decision(data, manifests)
    assert decision.handoff_to == ["bitrix24"]
    assert "unknown_agent" not in decision.handoff_to


def test_parse_decision_deduplicates_handoff_to():
    manifests = [_fake_manifest("bitrix24")]
    data = {"status": "completed", "answer": "", "handoff_to": ["bitrix24", "bitrix24"], "confidence": 0.8}
    decision = _parse_decision(data, manifests)
    assert decision.handoff_to == ["bitrix24"]


def test_parse_decision_invalid_status_defaults_to_completed():
    manifests = [_fake_manifest("bitrix24")]
    data = {"status": "nonsense", "answer": "", "handoff_to": [], "confidence": 0.5}
    decision = _parse_decision(data, manifests)
    assert decision.status == "completed"


def test_parse_decision_invalid_confidence_defaults_to_half():
    manifests = [_fake_manifest("bitrix24")]
    data = {"status": "completed", "answer": "", "handoff_to": [], "confidence": "bad"}
    decision = _parse_decision(data, manifests)
    assert decision.confidence == pytest.approx(0.5)


def test_parse_decision_empty_handoff_list():
    manifests = [_fake_manifest("bitrix24")]
    data = {"status": "needs_clarification", "answer": "Уточни запрос", "handoff_to": [], "confidence": 0.3}
    decision = _parse_decision(data, manifests)
    assert decision.handoff_to == []
    assert decision.answer == "Уточни запрос"


def test_parse_decision_non_list_handoff_ignored():
    manifests = [_fake_manifest("bitrix24")]
    data = {"status": "completed", "answer": "", "handoff_to": "bitrix24", "confidence": 0.8}
    decision = _parse_decision(data, manifests)
    assert decision.handoff_to == []


# _status
@pytest.mark.parametrize("value,expected", [
    ("completed", "completed"),
    ("needs_clarification", "needs_clarification"),
    ("failed", "failed"),
    ("nonsense", "completed"),
    (None, "completed"),
    ("", "completed"),
    ("needs_human", "completed"),  # not a valid orchestrator status
])
def test_status_normalization(value, expected):
    assert _status(value) == expected
