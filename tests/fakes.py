from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

from ai_server.agents.bitrix_llm import (
    BitrixLLMDecision,
    BitrixLLMDecisionResult,
    BitrixLLMFinalResult,
    BitrixLLMToolCall,
)
from ai_server.agents.bitrix_task_closure import TaskClosureDecision, TaskClosureToolCall
from ai_server.agents.pending_control_llm import PendingControlDecision, PendingControlResult
from ai_server.agents.pto_llm import (
    PtoLLMDecision,
    PtoLLMDecisionResult,
    PtoLLMFinalResult,
    PtoLLMToolCall,
)
from ai_server.models import ModelUsageRecord
from ai_server.orchestrators.internal_llm import InternalRouteDecision, InternalRouteResult


class FakeEmbeddingProvider:
    name = "test_embeddings"

    def __init__(self, *, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> dict[int, float]:
        tokens = re.findall(r"[0-9a-zа-яё_\.]{2,}", text.casefold().replace("ё", "е"))
        counts = Counter(tokens)
        vector: dict[int, float] = {}
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] = vector.get(index, 0.0) + (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in vector.items()}


class FakeBitrixLLM:
    def __init__(
        self,
        *,
        tool_calls: list[BitrixLLMToolCall] | None = None,
        decision_status: str = "completed",
        decision_answer: str = "",
        final_status: str = "completed",
        final_answer: str = "Готово.",
        confidence: float = 0.82,
    ) -> None:
        self.tool_calls = tool_calls or [BitrixLLMToolCall(name="none")]
        self.decision_status = decision_status
        self.decision_answer = decision_answer
        self.final_status = final_status
        self.final_answer = final_answer
        self.confidence = confidence
        self.decide_calls = []
        self.compose_calls = []

    async def decide(self, **kwargs):
        self.decide_calls.append(kwargs)
        return BitrixLLMDecisionResult(
            decision=BitrixLLMDecision(
                status=self.decision_status,
                answer=self.decision_answer,
                confidence=self.confidence,
                tool_calls=self.tool_calls,
            ),
            model_usage=_fake_usage(),
        )

    async def compose(self, **kwargs):
        self.compose_calls.append(kwargs)
        return BitrixLLMFinalResult(
            status=self.final_status,
            answer=self.final_answer,
            model_usage=_fake_usage(),
        )


class FakePendingControlLLM:
    def __init__(
        self,
        decision: str,
        *,
        answer: str = "",
        confidence: float = 0.9,
        reasoning: str = "test decision",
    ) -> None:
        self.decision = decision
        self.answer = answer
        self.confidence = confidence
        self.reasoning = reasoning
        self.classify_calls = []

    async def classify(self, **kwargs):
        self.classify_calls.append(kwargs)
        return PendingControlResult(
            decision=PendingControlDecision(
                decision=self.decision,
                answer=self.answer,
                confidence=self.confidence,
                reasoning=self.reasoning,
            ),
            model_usage=_fake_usage(agent_id="bitrix24_pending_control"),
        )


class FakeTaskClosureLLM:
    def __init__(self, decisions: list[TaskClosureDecision | dict] | None = None) -> None:
        self.decisions = decisions or [
            TaskClosureDecision(
                status="completed",
                answer="Готово.",
                tool_calls=[TaskClosureToolCall(name="none")],
                model_usage=_fake_usage(agent_id="bitrix24"),
            )
        ]
        self.decide_calls = []
        self._index = 0

    async def decide(self, **kwargs):
        self.decide_calls.append(kwargs)
        index = min(self._index, len(self.decisions) - 1)
        self._index += 1
        decision = self.decisions[index]
        if isinstance(decision, TaskClosureDecision):
            if decision.model_usage is not None:
                return decision
            return TaskClosureDecision(
                status=decision.status,
                answer=decision.answer,
                tool_calls=decision.tool_calls,
                confidence=decision.confidence,
                raw=decision.raw,
                model_usage=_fake_usage(agent_id="bitrix24"),
            )
        tool_calls = [
            call if isinstance(call, TaskClosureToolCall) else TaskClosureToolCall(**call)
            for call in decision.get("tool_calls", [{"name": "none"}])
        ]
        return TaskClosureDecision(
            status=str(decision.get("status") or "completed"),
            answer=str(decision.get("answer") or ""),
            tool_calls=tool_calls,
            confidence=float(decision.get("confidence") or 0.9),
            raw=decision,
            model_usage=_fake_usage(agent_id="bitrix24"),
        )


class FakePtoLLM:
    def __init__(
        self,
        *,
        tool_calls: list[PtoLLMToolCall] | None = None,
        tool_call_steps: list[list[PtoLLMToolCall]] | None = None,
        decision_status: str = "completed",
        decision_answer: str = "",
        final_status: str = "completed",
        final_answer: str = "Готово.",
        confidence: float = 0.82,
    ) -> None:
        self.tool_calls = tool_calls or [PtoLLMToolCall(name="none")]
        self.tool_call_steps = tool_call_steps
        self.decision_status = decision_status
        self.decision_answer = decision_answer
        self.final_status = final_status
        self.final_answer = final_answer
        self.confidence = confidence
        self.decide_calls = []
        self.compose_calls = []

    async def decide(self, **kwargs):
        self.decide_calls.append(kwargs)
        if self.tool_call_steps is not None:
            index = min(len(self.decide_calls) - 1, len(self.tool_call_steps) - 1)
            tool_calls = self.tool_call_steps[index]
        elif len(self.decide_calls) == 1:
            tool_calls = self.tool_calls
        else:
            tool_calls = [PtoLLMToolCall(name="none")]
        return PtoLLMDecisionResult(
            decision=PtoLLMDecision(
                status=self.decision_status,
                answer=self.decision_answer,
                confidence=self.confidence,
                tool_calls=tool_calls,
            ),
            model_usage=_fake_usage(agent_id="pto"),
        )

    async def compose(self, **kwargs):
        self.compose_calls.append(kwargs)
        return PtoLLMFinalResult(
            status=self.final_status,
            answer=self.final_answer,
            model_usage=_fake_usage(agent_id="pto"),
        )


def _fake_usage(*, agent_id: str = "bitrix24") -> ModelUsageRecord:
    return ModelUsageRecord(
        agent_id=agent_id,
        provider="fake",
        model="fake-bitrix-llm",
        status="used",
    )


class FakeInternalOrchestratorLLM:
    def __init__(
        self,
        *,
        handoff_to: list[str] | None = None,
        status: str = "completed",
        answer: str = "",
        confidence: float = 0.9,
    ) -> None:
        self.handoff_to = handoff_to or []
        self.status = status
        self.answer = answer
        self.confidence = confidence
        self.route_calls = []

    async def route(self, **kwargs):
        self.route_calls.append(kwargs)
        return InternalRouteResult(
            decision=InternalRouteDecision(
                status=self.status,
                answer=self.answer,
                handoff_to=self.handoff_to,
                confidence=self.confidence,
            ),
            model_usage=ModelUsageRecord(
                agent_id="internal_orchestrator",
                provider="fake",
                model="fake-orchestrator-llm",
                status="used",
            ),
        )

