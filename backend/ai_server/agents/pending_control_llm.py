from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ai_server.integrations.bitrix.dialog_state import PendingBitrixAction
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import ModelUsageRecord


PENDING_CONTROL_DECISIONS = {"confirm", "cancel", "new_request", "needs_clarification"}


class PendingControlLLM(Protocol):
    async def classify(
        self,
        *,
        dialog_key: str,
        user_id: int | None,
        user_text: str,
        pending_action: PendingBitrixAction,
    ) -> "PendingControlResult":
        pass


@dataclass(frozen=True)
class PendingControlDecision:
    decision: str
    answer: str = ""
    confidence: float = 0.5
    reasoning: str = ""


@dataclass(frozen=True)
class PendingControlResult:
    decision: PendingControlDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class PendingControlLLMService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def classify(
        self,
        *,
        dialog_key: str,
        user_id: int | None,
        user_text: str,
        pending_action: PendingBitrixAction,
    ) -> PendingControlResult:
        completion = await self.client.complete(
            agent_id="bitrix24_pending_control",
            messages=[
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "dialog_key": dialog_key,
                            "user_id": user_id,
                            "user_text": user_text,
                            "pending_action": _pending_payload(pending_action),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return PendingControlResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def _system_prompt() -> str:
    return (
        "Ты LLM-классификатор управления ожидающим Bitrix-действием. "
        "В диалоге уже подготовлено write-действие, но оно еще не выполнено. "
        "По новой реплике пользователя реши, что он хочет сделать с этим конкретным ожидающим действием. "
        "Учитывай естественную речь: подтверждение может быть выражено не только словом 'да', "
        "а отмена не только словом 'отмена'. "
        "decision=confirm только если пользователь явно разрешает выполнить именно ожидающее действие. "
        "decision=cancel только если пользователь явно отказывается, отменяет или просит не выполнять действие. "
        "decision=new_request если реплика выглядит как новый самостоятельный запрос и не подтверждает/не отменяет pending. "
        "decision=needs_clarification если смысл неоднозначен. "
        "Не выполняй действие и не выдумывай результат. Верни только JSON без markdown: "
        '{"decision":"confirm|cancel|new_request|needs_clarification",'
        '"answer":"короткий вопрос человеку, только если нужна ясность",'
        '"confidence":0.0,"reasoning":"кратко почему"}'
    )


def _pending_payload(action: PendingBitrixAction) -> dict[str, Any]:
    payload = asdict(action)
    payload["params"] = _compact_params(action.params)
    return payload


def _compact_params(params: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(params, ensure_ascii=False, default=str)
    if len(text) <= 3000:
        return params
    return {"_truncated_json": text[:3000]}


def _parse_decision(data: dict[str, Any]) -> PendingControlDecision:
    decision = str(data.get("decision") or "needs_clarification").strip()
    if decision not in PENDING_CONTROL_DECISIONS:
        decision = "needs_clarification"
    return PendingControlDecision(
        decision=decision,
        answer=str(data.get("answer") or "").strip(),
        confidence=_confidence(data.get("confidence")),
        reasoning=str(data.get("reasoning") or "").strip(),
    )


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(number, 0.0), 1.0)
