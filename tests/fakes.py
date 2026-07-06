from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Any

from ai_server.agents.bitrix24 import (
    BitrixLLMDecision,
    BitrixLLMDecisionResult,
    BitrixLLMFinalResult,
    BitrixLLMToolCall,
)
from ai_server.agents.logistics import (
    LogisticsLLMDecision,
    LogisticsLLMDecisionResult,
    LogisticsLLMFinalResult,
    LogisticsLLMToolCall,
)
from ai_server.agents.pto import (
    PtoLLMDecision,
    PtoLLMDecisionResult,
    PtoLLMFinalResult,
    PtoLLMToolCall,
)
from ai_server.llm import LLMCompletion
from ai_server.models import ModelUsageRecord
from ai_server.orchestrators.orchestrator_llm import (
    OrchestratorDecision,
    OrchestratorDecisionResult,
    OrchestratorFinalResult,
    OrchestratorToolCall,
)


class RecordingLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return LLMCompletion(
            content=self.content,
            model_usage=ModelUsageRecord(agent_id=kwargs["agent_id"], provider="fake", model="fake"),
            raw={},
        )


@_dataclass
class PendingControlDecision:
    decision: str
    answer: str = ""
    confidence: float = 0.9
    reasoning: str = ""


@_dataclass
class PendingControlResult:
    decision: PendingControlDecision
    model_usage: ModelUsageRecord


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
        tool_call_steps: list[list[BitrixLLMToolCall]] | None = None,
        decision_status: str = "completed",
        decision_answer: str = "",
        final_status: str = "completed",
        final_answer: str = "Готово.",
        confidence: float = 0.82,
    ) -> None:
        self.tool_calls = tool_calls or [BitrixLLMToolCall(name="none")]
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
            tool_calls = [BitrixLLMToolCall(name="none")]
        return BitrixLLMDecisionResult(
            decision=BitrixLLMDecision(
                status=self.decision_status,
                answer=self.decision_answer,
                confidence=self.confidence,
                tool_calls=tool_calls,
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


class FakeLogisticsLLM:
    def __init__(
        self,
        *,
        tool_call_steps: list[list[LogisticsLLMToolCall]] | None = None,
        final_status: str = "completed",
        final_answer: str = "Готово.",
        confidence: float = 0.84,
    ) -> None:
        self.tool_call_steps = tool_call_steps or [[LogisticsLLMToolCall(name="none")]]
        self.final_status = final_status
        self.final_answer = final_answer
        self.confidence = confidence
        self.decide_calls = []
        self.compose_calls = []

    async def decide(self, **kwargs):
        self.decide_calls.append(kwargs)
        index = min(len(self.decide_calls) - 1, len(self.tool_call_steps) - 1)
        return LogisticsLLMDecisionResult(
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                confidence=self.confidence,
                tool_calls=self.tool_call_steps[index],
            ),
            model_usage=_fake_usage(agent_id="logistics"),
        )

    async def compose(self, **kwargs):
        self.compose_calls.append(kwargs)
        return LogisticsLLMFinalResult(
            status=self.final_status,
            answer=self.final_answer,
            model_usage=_fake_usage(agent_id="logistics"),
        )


class FakeVehicleUsageStore:
    """In-memory VehicleUsageStorePort for tests."""

    def __init__(self) -> None:
        self._employees: list[dict[str, Any]] = []
        self._vehicles: list[dict[str, Any]] = [
            {"id": 1, "brand_model": "Авто 1", "registration_number": ""},
            {"id": 2, "brand_model": "Авто 2", "registration_number": ""},
            {"id": 3, "brand_model": "Авто 3", "registration_number": ""},
        ]
        self._requests: list[dict[str, Any]] = []
        self._day_reports: list[dict[str, Any]] = []
        self._operator_ids: set[int] = set()
        self._next_id = 1

    def upsert_employees(self, members: list[Any]) -> None:
        self._employees = [{"display_order": m.order, "full_name": m.name, "user_id": m.user_id} for m in members]

    def staff_roster(self) -> list[dict[str, Any]]:
        return list(self._employees)

    def vehicles(self) -> list[dict[str, Any]]:
        return list(self._vehicles)

    def vehicle_usage_operator_ids(self) -> set[int]:
        return set(self._operator_ids)

    def set_vehicle_usage_operators(self, *, operator_user_ids: list[int], actor_user_id: int | None) -> list[int]:
        self._operator_ids = set(operator_user_ids)
        return sorted(self._operator_ids)

    def context(self, *, request_date: str, user_id: int | None, dialog_id: str) -> dict[str, Any]:
        return {
            "request_date": request_date,
            "staff_roster": self.staff_roster(),
            "vehicles": self.vehicles(),
            "latest_request": self.latest_request(user_id=user_id, dialog_id=dialog_id),
            "day_report": self.get_day_report(report_date=request_date),
        }

    def latest_request(self, *, user_id: int | None, dialog_id: str) -> dict[str, Any] | None:
        if not user_id and not dialog_id:
            return None
        for req in reversed(self._requests):
            if dialog_id and req.get("dialog_id") == dialog_id:
                return req
            if user_id and req.get("user_id") == user_id:
                return req
        return None

    def get_request(self, *, request_date: str, user_id: int | None) -> dict[str, Any] | None:
        for req in reversed(self._requests):
            if req.get("request_date") == request_date and (user_id is None or req.get("user_id") == user_id):
                return req
        return None

    def latest_requests(self, *, limit: int) -> list[dict[str, Any]]:
        return list(reversed(self._requests))[:limit]

    def get_day_report(self, *, report_date: str) -> dict[str, Any]:
        for report in reversed(self._day_reports):
            if report.get("status_date") == report_date:
                return report
        return {
            "report_date": report_date,
            "employee_statuses": [],
            "vehicle_assignments": [],
            "vehicle_drivers": [],
        }

    def create_sent_request(self, data: Any) -> int:
        req_id = self._next_id
        self._next_id += 1
        self._requests.append(
            {
                "id": req_id,
                "request_date": data.request_date,
                "user_id": data.user_id,
                "dialog_id": data.dialog_id,
                "message": data.message,
                "sent_at": data.sent_at,
                "reminder_count": data.reminder_count,
                "status": "sent",
            }
        )
        return req_id

    def mark_escalated(self, *, request_date: str, user_id: int | None, escalated_at: str) -> bool:
        for req in self._requests:
            if req.get("request_date") == request_date:
                req["escalated_at"] = escalated_at
                return True
        return False

    def save_draft(
        self,
        *,
        request_date: str,
        user_id: int | None,
        dialog_id: str,
        response_text: str,
        parsed: dict[str, Any],
        status: str = "pending_confirmation",
    ) -> int:
        req_id = self._next_id
        self._next_id += 1
        self._requests.append(
            {
                "id": req_id,
                "request_date": request_date,
                "user_id": user_id,
                "dialog_id": dialog_id,
                "response_text": response_text,
                "parsed": parsed,
                "status": status,
            }
        )
        return req_id

    def replace_day_report(
        self,
        *,
        status_date: str,
        employee_statuses: list[tuple[int, str, str]],
        vehicle_assignments: list[tuple[int, int | None, str] | tuple[int, int | None, str, str]],
        actor_user_id: int | None = None,
    ) -> None:
        self._day_reports.append(
            {
                "status_date": status_date,
                "employee_statuses": list(employee_statuses),
                "vehicle_assignments": list(vehicle_assignments),
                "actor_user_id": actor_user_id,
            }
        )

    def update_day_report(
        self,
        *,
        report_date: str,
        people: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        actor_user_id: int | None = None,
        change_summary: str = "",
    ) -> dict[str, Any]:
        return {
            "report_date": report_date,
            "employee_updates": len(people),
            "vehicle_updates": len(vehicles),
            "actor_user_id": actor_user_id,
            "change_summary": change_summary,
        }

    def get_employee_period_report(self, *, employee_name: str, date_from: str, date_to: str) -> dict[str, Any]:
        return {
            "subject": "employee",
            "employee_name": employee_name,
            "date_from": date_from,
            "date_to": date_to,
            "days": [
                {
                    "status_date": date_from,
                    "status": "on_car",
                    "vehicle_name": "РђРІС‚Рѕ 2",
                    "notes": "",
                }
            ],
            "summary": {"on_car": 1},
        }

    def get_vehicle_period_report(self, *, vehicle_name: str, date_from: str, date_to: str) -> dict[str, Any]:
        return {
            "subject": "vehicle",
            "vehicle_name": vehicle_name,
            "date_from": date_from,
            "date_to": date_to,
            "days": [
                {
                    "assignment_date": date_from,
                    "status": "in_use",
                    "drivers": ["Р‘РѕСЂРёСЃРѕРІ РђРЅРґСЂРµР№"],
                    "notes": "",
                }
            ],
            "summary": {"in_use": 1},
        }

    def cancel_day_report(
        self,
        *,
        report_date: str,
        user_id: int | None,
        dialog_id: str,
        reason: str,
    ) -> int:
        parsed = {
            "date": report_date,
            "status": "day_off",
            "reason": reason,
        }
        request_id = self.save_draft(
            request_date=report_date,
            user_id=user_id,
            dialog_id=dialog_id,
            response_text=reason,
            parsed=parsed,
            status="cancelled_day_off",
        )
        self.replace_day_report(
            status_date=report_date,
            employee_statuses=[
                (int(row["display_order"]), "day_off", reason)
                for row in self._employees
                if row.get("display_order") is not None
            ],
            vehicle_assignments=[
                (int(row["id"]), None, "not_required", reason) for row in self._vehicles if row.get("id") is not None
            ],
            actor_user_id=user_id,
        )
        return request_id

    def finalize_pending_unknowns(
        self,
        *,
        report_date: str,
        actor_user_id: int | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        note = reason or "Auto-filled missing vehicle usage data as unknown."
        request = None
        for item in reversed(self._requests):
            if item.get("request_date") == report_date and item.get("status") in {
                "pending_clarification",
                "pending_confirmation",
            }:
                request = item
                break
        if request is None:
            return {"status": "skipped", "reason": "no_pending_draft", "report_date": report_date}
        parsed = dict(request.get("parsed") or {})
        existing_people = {
            int(item.get("staff_order")): item
            for item in parsed.get("people", [])
            if isinstance(item, dict) and item.get("staff_order") is not None
        }
        existing_vehicles = {
            int(item.get("vehicle_id")): item
            for item in parsed.get("vehicles", [])
            if isinstance(item, dict) and item.get("vehicle_id") is not None
        }
        people = []
        for employee in self._employees:
            employee_id = int(employee["display_order"])
            known = dict(existing_people.get(employee_id, {}))
            status = str(known.get("status") or "unknown")
            notes = str(known.get("notes") or "")
            if status == "unknown":
                notes = notes or note
            known.update(
                {
                    "staff_order": employee_id,
                    "full_name": employee.get("full_name"),
                    "status": status,
                    "notes": notes,
                }
            )
            people.append(known)
        vehicles = []
        for vehicle in self._vehicles:
            vehicle_id = int(vehicle["id"])
            known = dict(existing_vehicles.get(vehicle_id, {}))
            status = str(known.get("status") or "unknown")
            notes = str(known.get("notes") or "")
            if status == "unknown":
                notes = notes or note
            known.update(
                {
                    "vehicle_id": vehicle_id,
                    "vehicle_name": vehicle.get("brand_model"),
                    "status": status,
                    "drivers": known.get("drivers") if isinstance(known.get("drivers"), list) else [],
                    "notes": notes,
                }
            )
            vehicles.append(known)
        completed = dict(parsed)
        completed.update({"date": report_date, "people": people, "vehicles": vehicles, "auto_completed_unknown": True})
        request_id = self.save_draft(
            request_date=report_date,
            user_id=request.get("user_id"),
            dialog_id=str(request.get("dialog_id") or ""),
            response_text=str(request.get("response_text") or note),
            parsed=completed,
            status="answered",
        )
        self.replace_day_report(
            status_date=report_date,
            employee_statuses=[(int(item["staff_order"]), str(item["status"]), str(item.get("notes") or "")) for item in people],
            vehicle_assignments=[
                (int(item["vehicle_id"]), None, str(item["status"]), str(item.get("notes") or "")) for item in vehicles
            ],
            actor_user_id=actor_user_id or request.get("user_id"),
        )
        return {
            "status": "finalized_unknown",
            "report_date": report_date,
            "request_id": request_id,
            "employee_statuses_saved": len(people),
            "vehicle_assignments_saved": len(vehicles),
        }

    def auto_close_unanswered_day(
        self,
        *,
        report_date: str,
        reason: str,
    ) -> dict[str, Any]:
        request = self.get_request(request_date=report_date, user_id=None)
        if request and request.get("status") in {"answered", "cancelled_day_off", "pending_clarification", "pending_confirmation"}:
            return {
                "status": "skipped",
                "reason": "useful_response_exists",
                "report_date": report_date,
                "request_status": request.get("status"),
            }
        operator_ids = sorted(self._operator_ids)
        user_id = operator_ids[0] if operator_ids else request.get("user_id") if request else None
        request_id = self.cancel_day_report(
            report_date=report_date,
            user_id=user_id,
            dialog_id=str(user_id or ""),
            reason=reason,
        )
        return {"status": "closed_day_off", "report_date": report_date, "request_id": request_id, "user_id": user_id}


class FakePortalSearchIndex:
    """In-memory PortalSearchIndex for tests — implements the Protocol structurally."""

    def __init__(self, *, exists: bool = True) -> None:
        self._exists = exists
        self._items: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure_schema(self) -> None:
        pass

    def upsert_item(
        self,
        *,
        entity_type: str,
        entity_id: object,
        title: str = "",
        body: str = "",
        url: str = "",
        metadata: dict[str, Any] | None = None,
        source_updated_at: str | None = None,
        preserve_content: bool = True,
    ) -> None:
        key = (entity_type, str(entity_id))
        existing = self._items.get(key)
        meta = dict(metadata or {})
        if preserve_content and existing:
            for k, v in existing.get("metadata", {}).items():
                if k.startswith("content_"):
                    meta.setdefault(k, v)
        self._items[key] = {
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "title": title,
            "body": body,
            "url": url,
            "metadata": meta,
            "source_updated_at": source_updated_at,
        }

    def delete_item(self, *, entity_type: str, entity_id: object) -> bool:
        key = (entity_type, str(entity_id))
        if key in self._items:
            del self._items[key]
            return True
        return False

    def delete_stale_items(self, *, entity_types: set[str], seen_before: str) -> int:
        return 0

    def search(self, query: str, *, entity_types: set[str] | None = None, limit: int = 10) -> list[Any]:
        from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult

        query_lower = query.casefold()
        terms = [t for t in query_lower.split() if t]
        results = []
        for item in self._items.values():
            if entity_types and item["entity_type"] not in entity_types:
                continue
            text = f"{item['title']} {item['body']}".casefold()
            score = sum(1 for t in terms if t in text)
            if score > 0:
                results.append(
                    PortalSearchResult(
                        entity_type=item["entity_type"],
                        entity_id=item["entity_id"],
                        title=item["title"],
                        body=item.get("body", ""),
                        url=item.get("url", ""),
                        score=score * 10,
                        metadata=dict(item.get("metadata", {})),
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def stats(self) -> Any:
        from ai_server.integrations.bitrix.portal_search.types import PortalIndexStats

        by_type: dict[str, int] = {}
        content_by_status: dict[str, int] = {}
        for item in self._items.values():
            et = item["entity_type"]
            by_type[et] = by_type.get(et, 0) + 1
            status = item.get("metadata", {}).get("content_index_status")
            if status:
                content_by_status[status] = content_by_status.get(status, 0) + 1
        return PortalIndexStats(
            total_items=len(self._items),
            by_type=by_type,
            content_by_status=content_by_status,
            last_indexed_at=None,
            path=Path("/fake/portal_search"),
            exists=self._exists,
        )

    def get_item(self, *, entity_type: str, entity_id: object) -> Any:
        from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult

        item = self._items.get((entity_type, str(entity_id)))
        if item is None:
            return None
        return PortalSearchResult(
            entity_type=item["entity_type"],
            entity_id=item["entity_id"],
            title=item["title"],
            body=item.get("body", ""),
            url=item.get("url", ""),
            score=0,
            metadata=dict(item.get("metadata", {})),
        )

    def item_snapshot(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None:
        return (
            dict(self._items[(entity_type, str(entity_id))]) if (entity_type, str(entity_id)) in self._items else None
        )

    def disk_delta_folder_candidates(
        self,
        *,
        cursor_type: str | None,
        cursor_id: str | None,
        limit: int,
    ) -> tuple[list[Any], str | None, str | None, bool]:
        candidates = [
            self.get_item(entity_type=et, entity_id=eid)
            for et, eid in self._items
            if et in ("disk_storage", "disk_folder")
        ]
        candidates = [c for c in candidates if c is not None]
        return candidates[:limit], None, None, True

    def children_by_parent_id(self, parent_id: object) -> list[Any]:
        from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult

        parent_str = str(parent_id)
        result = []
        for item in self._items.values():
            meta = item.get("metadata", {})
            if str(meta.get("parent_id", "")) == parent_str:
                result.append(
                    PortalSearchResult(
                        entity_type=item["entity_type"],
                        entity_id=item["entity_id"],
                        title=item["title"],
                        body=item.get("body", ""),
                        url=item.get("url", ""),
                        score=0,
                        metadata=dict(meta),
                    )
                )
        return result

    def content_candidates(self, *, limit: int) -> list[Any]:
        from ai_server.integrations.bitrix.portal_search.types import CONTENT_TERMINAL_STATUSES, PortalSearchResult

        result = []
        for item in self._items.values():
            status = item.get("metadata", {}).get("content_index_status", "")
            if not status or status not in {"indexed"} | CONTENT_TERMINAL_STATUSES:
                result.append(
                    PortalSearchResult(
                        entity_type=item["entity_type"],
                        entity_id=item["entity_id"],
                        title=item["title"],
                        body=item.get("body", ""),
                        url=item.get("url", ""),
                        score=0,
                        metadata=dict(item.get("metadata", {})),
                    )
                )
        return result[:limit]

    def content_readiness(self, *, allowed_extensions: set[str]) -> Any:
        from ai_server.integrations.bitrix.portal_search.types import PortalContentReadiness

        indexed = sum(
            1 for item in self._items.values() if item.get("metadata", {}).get("content_index_status") == "indexed"
        )
        total = len(self._items)
        return PortalContentReadiness(
            total_documents=total,
            supported_documents=total,
            indexed=indexed,
            pending=total - indexed,
            terminal=0,
            unsupported=0,
            indexed_by_extension={},
            pending_by_extension={},
            pending_by_status={},
            terminal_by_status={},
            unsupported_by_extension={},
        )

    def update_item_body_metadata(
        self, *, entity_type: str, entity_id: object, body: str, metadata: dict[str, Any]
    ) -> None:
        key = (entity_type, str(entity_id))
        if key in self._items:
            self._items[key]["body"] = body
            self._items[key]["metadata"] = metadata


class FakeOrchestratorStore:
    """In-memory orchestrator store with KV support for pending_specialist tests."""

    def __init__(self) -> None:
        self._kv: dict[tuple[str, str], str] = {}
        self._turns: dict[str, list[dict[str, str]]] = {}

    def set_pending(self, dialog_key: str, specialist_id: str) -> None:
        self._kv[(dialog_key, "pending_specialist")] = specialist_id

    async def ensure_schema(self) -> None:
        pass

    async def load_turns(self, dialog_key: str, *, limit: int = 20) -> list[dict[str, str]]:
        return self._turns.get(dialog_key, [])[-limit:]

    async def append_turn(self, dialog_key: str, user_text: str, agent_response: str) -> None:
        self._turns.setdefault(dialog_key, []).extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": agent_response},
            ]
        )

    async def get_kv(self, dialog_key: str, field: str) -> str | None:
        return self._kv.get((dialog_key, field))

    async def set_kv(self, dialog_key: str, field: str, value: str) -> None:
        self._kv[(dialog_key, field)] = value

    async def delete_kv(self, dialog_key: str, field: str) -> None:
        self._kv.pop((dialog_key, field), None)


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
        call_specialists: list[str] | None = None,
        status: str = "completed",
        answer: str = "",
        confidence: float = 0.9,
        synthesized_answer: str = "",
    ) -> None:
        self.call_specialists = call_specialists or []
        self.status = status
        self.answer = answer
        self.confidence = confidence
        self.synthesized_answer = synthesized_answer
        self.decide_calls: list[dict] = []
        self.compose_calls: list[dict] = []

    def _make_usage(self) -> ModelUsageRecord:
        return ModelUsageRecord(
            agent_id="internal_orchestrator",
            provider="fake",
            model="fake-orchestrator-llm",
            status="used",
        )

    async def decide(self, *, task, tool_results=None, **kwargs):
        self.decide_calls.append({"task": task, "tool_results": tool_results or [], **kwargs})
        existing = tool_results or []
        if existing or not self.call_specialists:
            # Уже есть результаты или нечего вызывать — завершить цикл
            tool_calls = [OrchestratorToolCall(name="none")]
        else:
            tool_calls = [
                OrchestratorToolCall(
                    name="call_specialist",
                    args={"specialist_id": sid, "request": task.request},
                )
                for sid in self.call_specialists
            ]
        return OrchestratorDecisionResult(
            decision=OrchestratorDecision(
                status=self.status,
                answer=self.answer,
                tool_calls=tool_calls,
                confidence=self.confidence,
            ),
            model_usage=self._make_usage(),
        )

    async def compose(self, *, tool_results=None, **kwargs):
        self.compose_calls.append({"tool_results": tool_results or [], **kwargs})
        successful = [tr for tr in (tool_results or []) if getattr(tr, "status", None) == "ok"]
        if not successful:
            # Нет успешных специалистов — прямой ответ или ошибка
            return OrchestratorFinalResult(
                answer=self.answer or "",
                status=self.status if self.answer else "failed",
                model_usage=self._make_usage(),
            )
        if self.synthesized_answer and len(successful) > 1:
            answer = self.synthesized_answer
        else:
            data = successful[0].data or {}
            answer = data.get("answer", self.answer)
        return OrchestratorFinalResult(
            answer=answer,
            status=self.status,
            model_usage=self._make_usage(),
        )


class FakeTaskDraftStore:
    def __init__(self) -> None:
        self._drafts: dict[str, dict] = {}

    async def save_task_draft(self, dialog_key: str, params: dict[str, Any]) -> None:
        self._drafts[dialog_key] = params

    async def get_task_draft(self, dialog_key: str) -> dict[str, Any] | None:
        return self._drafts.get(dialog_key)

    async def delete_task_draft(self, dialog_key: str) -> None:
        self._drafts.pop(dialog_key, None)


class FakeProposalStore:
    def __init__(self) -> None:
        self._proposals: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    def save_proposal(
        self,
        *,
        task_id: int,
        task_title: str,
        missing_parts: str,
        responsible_id: int | None,
        responsible_dialog_id: str,
        scheduled_for: str,
    ) -> int:
        pid = self._next_id
        self._next_id += 1
        self._proposals[pid] = {
            "id": pid,
            "task_id": task_id,
            "task_title": task_title,
            "missing_parts": missing_parts,
            "responsible_id": responsible_id,
            "responsible_dialog_id": responsible_dialog_id,
            "scheduled_for": scheduled_for,
            "status": "awaiting_response",
        }
        return pid

    def delete_proposal(self, proposal_id: int) -> None:
        self._proposals.pop(proposal_id, None)

    def update_responsible_response(self, proposal_id: int, response_text: str) -> None:
        if proposal_id in self._proposals:
            self._proposals[proposal_id]["responsible_response"] = response_text

    def get_proposals_for_manager(self) -> list[dict[str, Any]]:
        return list(self._proposals.values())

    def get_pending_for_responsible(self, responsible_id: int) -> dict[str, Any] | None:
        return next(
            (p for p in self._proposals.values() if p.get("responsible_id") == responsible_id),
            None,
        )
