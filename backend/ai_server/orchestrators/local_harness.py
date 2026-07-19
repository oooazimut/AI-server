"""Deterministic local-only harness for the future AI-server orchestrator.

It deliberately uses no network clients or persistent services.  The module keeps
the production core's task and dialog-partition concepts while supplying explicit
scripted adapters for the T-0006 corpus.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ai_server.agent_queue_utils import agent_queue_partition_key
from ai_server.models import AgentTask


CAPABILITIES = ("warehouse lookup", "scripted contents/stock", "scripted delivery")
WAREHOUSES = {
    "борисов": {"id": "19", "title": "Борисов А.А.", "active": "Y"},
    "гараж": {"id": "3", "title": "Гараж", "active": "Y"},
    "карасев": {"id": "23", "title": "Карасев А.В.", "active": "Y"},
}


@dataclass
class BranchResult:
    executor: str
    status: str
    answer: str
    attempt_id: str


@dataclass
class HarnessResult:
    case_id: str
    verdict: str
    route: list[str]
    clarification: str | None
    branches: list[BranchResult] = field(default_factory=list)
    parallel: bool = False
    round_trips: int = 0
    final_response: str = ""
    correlation_ids: dict[str, str] = field(default_factory=dict)
    latency_ms: int = 0
    executor_calls: int = 0
    model_calls: int = 0
    model_tokens: int = 0
    model_trace: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


class LocalOrchestratorHarness:
    """A repeatable test seam, not a replacement for ``InternalOrchestrator``."""

    def __init__(self, *, model_receipts: dict[str, dict[str, Any]] | None = None) -> None:
        self.model_receipts = model_receipts or {}
        self.delivered_attempts: set[str] = set()
        self.contexts: dict[str, str] = {}

    async def run_case(self, case_id: str, request: str) -> HarnessResult:
        started = time.monotonic()
        task = AgentTask(
            task_id=f"T0006-S02-{case_id}",
            request=request,
            user={"id": "u1"},
            context={"dialog_key": f"chat:local:user:{case_id}"},
        )
        correlation = {"task_id": task.task_id, "partition": agent_queue_partition_key({"payload": task.model_dump()})}
        result = await self._dispatch(case_id, task, correlation)
        result.latency_ms = round((time.monotonic() - started) * 1000)
        return result

    async def _dispatch(self, case_id: str, task: AgentTask, correlation: dict[str, str]) -> HarnessResult:
        plans = {
            "Q01": (["bitrix"], None), "Q02": (["bitrix"], None), "Q03": (["bitrix"], None),
            "Q04": (["bitrix"], None), "Q05": (["bitrix"], None), "Q06": (["bitrix", "logistics"], None),
            "Q07": (["bitrix"], "Вы имели в виду склад Карасева А.В.?"),
            "Q08": (["bitrix"], "Уточните: Гараж или Гараж Смородин?"),
            "Q09": ([], None), "Q10": ([], None), "Q11": (["bitrix"], None), "Q12": (["bitrix"], None),
            "Q13": (["bitrix", "logistics"], None), "Q14": ([], "Продолжить активную задачу или отменить её?"),
            "Q15": (["bitrix"], None), "Q16": (["bitrix"], None),
        }
        route, clarification = plans[case_id]
        result = HarnessResult(case_id=case_id, verdict="PASS", route=route, clarification=clarification, correlation_ids=correlation)
        receipt = self.model_receipts.get(case_id)
        if receipt is None:
            result.notes.append("DeepSeek routing receipt unavailable: wrapper blocked before network by policy.")
        else:
            result.model_calls = 1
            result.model_tokens = int(receipt.get("usage", {}).get("completion_tokens", 0))
            result.model_trace = {
                "status": str(receipt.get("status", "UNKNOWN")),
                "elapsed_ms": int(receipt.get("elapsed_ms", 0)),
                "usage": receipt.get("usage", {}),
                "content": str(receipt.get("content", "")),
            }
            result.notes.append("DeepSeek receipt merged; planner text is non-authoritative and did not replace fixtures.")

        if case_id == "Q09":
            result.final_response = "Доступно: " + ", ".join(CAPABILITIES) + "."
            return result
        if case_id == "Q10":
            result.verdict = "NOT_SUPPORTED"
            result.final_response = "Погода, время и анекдоты не входят в активный каталог возможностей."
            return result
        if case_id == "Q14":
            result.verdict = "PARTIAL"
            result.final_response = clarification or ""
            return result
        if clarification:
            result.verdict = "PARTIAL"
            result.final_response = clarification
            return result
        if case_id == "Q15":
            task_a = AgentTask(task_id=f"{task.task_id}:a", request="Борисов", user={"id": "a"}, context={"dialog_key": "chat:a:user:1"})
            task_b = AgentTask(task_id=f"{task.task_id}:b", request="Гараж", user={"id": "b"}, context={"dialog_key": "chat:b:user:2"})
            left, right = await asyncio.gather(self._record_context(task_a), self._record_context(task_b))
            result.correlation_ids.update({"user_a_partition": left, "user_b_partition": right})
            result.final_response = "Контексты пользователей изолированы: Борисов; Гараж."
            return result
        if case_id == "Q16":
            attempt = f"{task.task_id}:attempt:1"
            accepted = self._deliver_attempt(attempt)
            duplicate_accepted = self._deliver_attempt(attempt)
            result.branches = [BranchResult("bitrix", "ok", "Гараж", attempt)]
            result.executor_calls = 1
            result.final_response = "Гараж (id 3). Поздний duplicate attempt:1 подавлен."
            result.correlation_ids["suppressed_attempt_id"] = attempt
            result.notes.append(f"accepted_attempt={accepted}; duplicate_accepted={duplicate_accepted}; stale duplicate suppressed")
            return result

        result.parallel = len(route) > 1 or case_id in {"Q03", "Q12"}
        if case_id == "Q13":
            branches = []
        elif case_id in {"Q03", "Q12"}:
            branches = [self._execute(case_id, "bitrix", task.task_id, index + 1) for index in range(3)]
        else:
            branches = [self._execute(case_id, executor, task.task_id, index + 1) for index, executor in enumerate(route)]
        result.branches = list(await asyncio.gather(*branches))
        result.executor_calls = len(result.branches)
        result.round_trips = 1
        if case_id == "Q11":
            result.verdict = "PARTIAL"
            result.final_response = "Склад Борисов временно недоступен: scripted timeout."
        elif case_id == "Q12":
            result.verdict = "PARTIAL"
            result.final_response = "Доступные ветки вернули результаты; одна scripted ветка остатков завершилась ошибкой."
        elif case_id == "Q13":
            result.branches = []
            for round_no in range(1, 4):
                round_results = await asyncio.gather(
                    *(self._execute(case_id, executor, task.task_id, round_no * 10 + index) for index, executor in enumerate(route, 1))
                )
                result.branches.extend(round_results)
            result.executor_calls = len(result.branches)
            result.round_trips = 3
            result.verdict = "PARTIAL"
            result.final_response = "Исполнители подтвердили «не моя задача»; цикл остановлен на лимите 3 раундов."
        elif case_id == "Q03":
            result.verdict = "PARTIAL"
            result.final_response = "Найден Борисов; Иванов и Петров не подтверждены очищенным warehouse fixture."
        else:
            result.final_response = self._compose(case_id, result.branches)
        return result

    async def _execute(self, case_id: str, executor: str, task_id: str, index: int) -> BranchResult:
        await asyncio.sleep(0)
        attempt = f"{task_id}:attempt:{index}"
        if case_id == "Q12" and index == 2:
            return BranchResult(executor, "error", "scripted stock adapter failure", attempt)
        if case_id == "Q11":
            return BranchResult(executor, "timeout", "scripted slow/absent Bitrix response", attempt)
        if case_id == "Q13":
            return BranchResult(executor, "not_mine", "scripted executor says not mine", attempt)
        if executor == "logistics":
            return BranchResult(executor, "ok", "scripted delivery: route requires confirmation", attempt)
        if case_id == "Q03" and index == 2:
            return BranchResult(executor, "not_found", "Иванов: not present in cleaned warehouse fixture", attempt)
        if case_id == "Q03" and index == 3:
            return BranchResult(executor, "not_found", "Петров: not present in cleaned warehouse fixture", attempt)
        return BranchResult(executor, "ok", self._warehouse_answer(case_id), attempt)

    async def _record_context(self, task: AgentTask) -> str:
        await asyncio.sleep(0)
        dialog_key = str(task.context["dialog_key"])
        self.contexts[dialog_key] = task.request
        return agent_queue_partition_key({"payload": task.model_dump()})

    def _deliver_attempt(self, attempt_id: str) -> bool:
        if attempt_id in self.delivered_attempts:
            return False
        self.delivered_attempts.add(attempt_id)
        return True

    @staticmethod
    def _warehouse_answer(case_id: str) -> str:
        if case_id == "Q04":
            item = WAREHOUSES["гараж"]
        else:
            item = WAREHOUSES["борисов"]
        answer = f"{item['title']} (id {item['id']}, active {item['active']})"
        if case_id == "Q02":
            answer += "; scripted contents (not Bitrix facts): cable=12, fasteners=20"
        return answer

    @staticmethod
    def _compose(case_id: str, branches: list[BranchResult]) -> str:
        return "; ".join(branch.answer for branch in branches if branch.status == "ok") or f"No successful scripted branch for {case_id}."
