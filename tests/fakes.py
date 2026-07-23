from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Any

from ai_server.llm import LLMCompletion
from ai_server.models import ModelUsageRecord


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
                "source": str(getattr(data, "source", "") or ""),
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
        source = ""
        for previous in reversed(self._requests):
            if previous.get("request_date") == request_date and previous.get("user_id") == user_id:
                source = str(previous.get("source") or "")
                break
        self._requests.append(
            {
                "id": req_id,
                "request_date": request_date,
                "user_id": user_id,
                "dialog_id": dialog_id,
                "response_text": response_text,
                "parsed": parsed,
                "status": status,
                "source": source,
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
            employee_statuses=[
                (int(item["staff_order"]), str(item["status"]), str(item.get("notes") or "")) for item in people
            ],
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
        if (
            request
            and request.get("source") == "manual"
            and request.get("status") in {"pending_clarification", "pending_confirmation"}
            and isinstance(request.get("parsed"), dict)
        ):
            return self.finalize_pending_unknowns(
                report_date=report_date,
                actor_user_id=request.get("user_id"),
                reason="Auto-filled missing vehicle usage data as unknown at day close.",
            )
        if request and request.get("status") in {
            "answered",
            "cancelled_day_off",
            "pending_clarification",
            "pending_confirmation",
        }:
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
        self._task_close_processing_state: dict[tuple[str, str], dict[str, Any]] = {}
        self._task_close_control_events: dict[tuple[str, str], dict[str, Any]] = {}
        self._task_close_settings: dict[str, dict[str, Any]] = {}
        self._task_close_controlled_users: set[int] = set()
        self._task_close_controlled_from: dict[int, str] = {}

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

    def get_task_close_processing_state(self, *, task_id: object, state_key: str) -> dict[str, Any] | None:
        state = self._task_close_processing_state.get((str(task_id), state_key))
        return dict(state) if state else None

    def list_task_close_processing_states(
        self,
        *,
        statuses: list[str] | None = None,
        state_key_prefix: str = "",
        responsible_id: int | None = None,
        dialog_key: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        status_set = set(statuses or [])
        rows: list[dict[str, Any]] = []
        for state in self._task_close_processing_state.values():
            payload = dict(state.get("payload") or {})
            if status_set and state.get("status") not in status_set:
                continue
            if state_key_prefix and not str(state.get("state_key") or "").startswith(state_key_prefix):
                continue
            if responsible_id is not None and payload.get("responsible_id") != responsible_id:
                continue
            if dialog_key and payload.get("dialog_key") != dialog_key:
                continue
            rows.append(dict(state))
        rows.sort(key=lambda row: (row.get("created_at") or 0, row.get("task_id") or "", row.get("state_key") or ""))
        return rows[:limit]

    def upsert_task_close_processing_state(
        self,
        *,
        task_id: object,
        state_key: str,
        status: str,
        payload: dict[str, Any] | None = None,
        actor_user_id: int | None = None,
    ) -> None:
        existing = self._task_close_processing_state.get((str(task_id), state_key))
        self._task_close_processing_state[(str(task_id), state_key)] = {
            "task_id": str(task_id),
            "state_key": state_key,
            "status": status,
            "payload": dict(payload or {}),
            "actor_user_id": actor_user_id,
            "created_at": (existing or {}).get("created_at", len(self._task_close_processing_state) + 1),
            "updated_at": len(self._task_close_processing_state) + 1,
        }

    def get_task_close_control_event(self, *, task_id: object, close_event_key: str) -> dict[str, Any] | None:
        event = self._task_close_control_events.get((str(task_id), close_event_key))
        return dict(event) if event else None

    def upsert_task_close_control_event(
        self,
        *,
        task_id: object,
        close_event_key: str,
        decision: str,
        reason: str = "",
        closed_at: str | None = None,
        responsible_id: int | None = None,
        closed_by_user_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not close_event_key or not decision:
            return
        self._task_close_control_events[(str(task_id), close_event_key)] = {
            "task_id": str(task_id),
            "close_event_key": close_event_key,
            "decision": decision,
            "reason": reason,
            "closed_at": closed_at,
            "responsible_id": responsible_id,
            "closed_by_user_id": closed_by_user_id,
            "payload": dict(payload or {}),
        }

    def get_task_close_control_setting(self, key: str) -> dict[str, Any] | None:
        setting = self._task_close_settings.get(key)
        return dict(setting) if setting else None

    def set_task_close_control_setting(self, *, key: str, value: str, updated_by: int | None = None) -> None:
        self._task_close_settings[key] = {"key": key, "value": value, "updated_by": updated_by}

    def task_close_controlled_user_ids(self) -> set[int]:
        return set(self._task_close_controlled_users)

    def upsert_task_close_controlled_user(
        self,
        *,
        user_id: int,
        active: bool = True,
        updated_by: int | None = None,
        controlled_from: str | None = None,
    ) -> None:
        if active:
            self._task_close_controlled_users.add(int(user_id))
            if controlled_from:
                self._task_close_controlled_from.setdefault(int(user_id), controlled_from)
        else:
            self._task_close_controlled_users.discard(int(user_id))

    def get_task_close_controlled_user(self, user_id: int) -> dict[str, Any] | None:
        if int(user_id) not in self._task_close_controlled_users:
            return None
        return {
            "user_id": int(user_id),
            "active": True,
            "controlled_from": self._task_close_controlled_from.get(int(user_id)),
        }


class FakeOrchestratorStore:
    """In-memory orchestrator store with KV support for pending_specialist tests."""

    def __init__(self) -> None:
        self._kv: dict[tuple[str, str], str] = {}
        self._turns: dict[str, list[dict[str, str]]] = {}
        self._replacement_candidates: dict[str, dict[str, str]] = {}

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

    async def save_replacement_candidate(self, dialog_key, *, request_text, draft_id, draft_type, ttl_minutes=15):
        current = self._replacement_candidates.get(dialog_key)
        if current:
            return dict(current)
        value = {
            "request_text": str(request_text),
            "draft_id": str(draft_id),
            "draft_type": str(draft_type),
            "created_at": "fake-now",
            "expires_at": "fake-later",
        }
        self._replacement_candidates[dialog_key] = value
        return dict(value)

    async def get_replacement_candidate(self, dialog_key):
        current = self._replacement_candidates.get(dialog_key)
        return dict(current) if current else None

    async def delete_replacement_candidate(self, dialog_key):
        self._replacement_candidates.pop(dialog_key, None)


def _fake_usage(*, agent_id: str = "bitrix24") -> ModelUsageRecord:
    return ModelUsageRecord(
        agent_id=agent_id,
        provider="fake",
        model="fake-bitrix-executor",
        status="used",
    )


class FakeTaskDraftStore:
    def __init__(self) -> None:
        self._drafts: dict[str, dict] = {}
        self._expired: set[str] = set()
        self._task_close_operators: set[int] = set()
        self._task_close_controlled_users: set[int] = set()
        self._task_close_settings: dict[str, dict[str, Any]] = {}
        self._task_close_revisions: list[dict[str, Any]] = []
        self._draft_sequence = 0
        self._confirming: set[str] = set()
        self._claim_sequence = 0
        self._claim_tokens: dict[str, str] = {}

    async def save_task_draft(self, dialog_key: str, params: dict[str, Any]) -> None:
        if dialog_key in self._confirming:
            raise RuntimeError("ACTIVE_DRAFT_IN_PROGRESS")
        current = self._drafts.get(dialog_key)
        incoming_type = str(params.get("_draft_type") or "task_create")
        if current and str(current.get("_draft_type") or "task_create") != incoming_type:
            raise RuntimeError("ACTIVE_DRAFT_CONFLICT")
        if current:
            draft_id = str(current.get("_draft_id"))
            version = int(current.get("_draft_version") or 1) + 1
            created_at = current.get("_draft_created_at", "fake-created-at")
        else:
            self._draft_sequence += 1
            draft_id = f"fake-draft-{self._draft_sequence}"
            version = 1
            created_at = "fake-created-at"
        self._drafts[dialog_key] = {
            **params,
            "_draft_id": draft_id,
            "_draft_type": incoming_type,
            "_draft_version": version,
            "_draft_created_at": created_at,
            "_draft_expires_at": "fake-expires-at",
        }
        if current and current.get("_original_request"):
            self._drafts[dialog_key]["_original_request"] = current["_original_request"]
        self._expired.discard(dialog_key)

    async def claim_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
    ) -> dict[str, Any] | None:
        draft = self._drafts.get(dialog_key)
        if (
            not draft
            or dialog_key in self._confirming
            or str(draft.get("_draft_id")) != expected_draft_id
            or int(draft.get("_draft_version") or 0) != expected_version
            or str(draft.get("_draft_type")) != expected_type
        ):
            return None
        self._confirming.add(dialog_key)
        self._claim_sequence += 1
        claim_token = f"fake-claim-{self._claim_sequence}"
        self._claim_tokens[dialog_key] = claim_token
        return {**draft, "_draft_claim_token": claim_token}

    async def claim_expired_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
    ) -> dict[str, Any] | None:
        return await self.claim_task_draft(
            dialog_key,
            expected_draft_id=expected_draft_id,
            expected_version=expected_version,
            expected_type=expected_type,
        )

    async def reclaim_stale_finalizing_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        draft = self._drafts.get(dialog_key)
        if (
            not draft
            or dialog_key not in self._confirming
            or str(draft.get("_draft_id")) != expected_draft_id
            or int(draft.get("_draft_version") or 0) != expected_version
            or str(draft.get("_draft_type")) != expected_type
        ):
            return None
        self._claim_sequence += 1
        claim_token = f"fake-claim-{self._claim_sequence}"
        self._claim_tokens[dialog_key] = claim_token
        return {**draft, "_draft_claim_token": claim_token, "_reclaimed_finalizing": True}

    async def renew_task_draft_claim(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        claim_token: str,
        expected_status: str,
    ) -> bool:
        draft = self._drafts.get(dialog_key)
        return bool(
            draft
            and dialog_key in self._confirming
            and str(draft.get("_draft_id")) == draft_id
            and self._claim_tokens.get(dialog_key) == claim_token
            and expected_status in {"confirming", "finalizing"}
        )

    async def resolve_stale_confirming_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        draft = self._drafts.get(dialog_key)
        if (
            not draft
            or dialog_key not in self._confirming
            or str(draft.get("_draft_id")) != expected_draft_id
            or int(draft.get("_draft_version") or 0) != expected_version
            or str(draft.get("_draft_type")) != expected_type
            or not draft.get("_force_stale_claim")
        ):
            return None
        self._confirming.discard(dialog_key)
        self._claim_tokens.pop(dialog_key, None)
        self._drafts.pop(dialog_key, None)
        return {**draft, "_draft_resolution_status": "attention"}

    async def release_task_draft(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        claim_token: str = "",
    ) -> None:
        draft = self._drafts.get(dialog_key)
        token_matches = not claim_token or self._claim_tokens.get(dialog_key) == claim_token
        if draft and str(draft.get("_draft_id")) == draft_id and token_matches:
            self._confirming.discard(dialog_key)
            self._claim_tokens.pop(dialog_key, None)

    async def finalize_task_draft_claim(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        params: dict[str, Any],
        claim_token: str = "",
    ) -> dict[str, Any] | None:
        current = self._drafts.get(dialog_key)
        if (
            not current
            or dialog_key not in self._confirming
            or str(current.get("_draft_id")) != draft_id
            or (claim_token and self._claim_tokens.get(dialog_key) != claim_token)
        ):
            return None
        stored = {
            **params,
            "_draft_id": draft_id,
            "_draft_type": current.get("_draft_type"),
            "_draft_version": int(current.get("_draft_version") or 1) + 1,
            "_draft_created_at": current.get("_draft_created_at"),
            "_draft_expires_at": current.get("_draft_expires_at"),
        }
        self._drafts[dialog_key] = stored
        self._confirming.discard(dialog_key)
        self._claim_tokens.pop(dialog_key, None)
        return stored

    async def get_task_draft_for_finalizer(self, dialog_key: str) -> dict[str, Any] | None:
        draft = self._drafts.get(dialog_key)
        return dict(draft) if isinstance(draft, dict) else None

    async def get_claimed_task_draft(
        self,
        dialog_key: str,
        *,
        expected_type: str,
    ) -> dict[str, Any] | None:
        draft = self._drafts.get(dialog_key)
        if dialog_key not in self._confirming or not isinstance(draft, dict):
            return None
        if str(draft.get("_draft_type")) != expected_type:
            return None
        return {**draft, "_draft_claim_token": self._claim_tokens.get(dialog_key, "")}

    async def get_task_draft(self, dialog_key: str, *, ttl_minutes: int | None = None) -> dict[str, Any] | None:
        if ttl_minutes is not None and dialog_key in self._expired:
            self._drafts.pop(dialog_key, None)
            self._expired.discard(dialog_key)
            return None
        return self._drafts.get(dialog_key)

    async def delete_task_draft(
        self,
        dialog_key: str,
        *,
        status: str = "cancelled",
        expected_draft_id: str = "",
        expected_version: int | None = None,
        expected_claim_token: str = "",
    ) -> None:
        current = self._drafts.get(dialog_key)
        identity_matches = not expected_draft_id or (current and str(current.get("_draft_id")) == expected_draft_id)
        version_matches = expected_version is None or (
            current and int(current.get("_draft_version") or 0) == expected_version
        )
        token_matches = not expected_claim_token or self._claim_tokens.get(dialog_key) == expected_claim_token
        if identity_matches and version_matches and token_matches:
            self._drafts.pop(dialog_key, None)
            self._confirming.discard(dialog_key)
            self._claim_tokens.pop(dialog_key, None)

    async def confirm_admin_change_draft(
        self,
        *,
        dialog_key: str,
        draft_id: str,
        draft_version: int,
        actor_user_id: int,
    ) -> dict[str, Any]:
        draft = self._drafts.get(dialog_key)
        if (
            not isinstance(draft, dict)
            or dialog_key in self._confirming
            or str(draft.get("_draft_id")) != draft_id
            or int(draft.get("_draft_version") or 0) != draft_version
            or str(draft.get("_draft_type")) != "admin_change"
        ):
            return {"status": "not_found"}
        field = str(draft.get("field") or "")
        target_user_id = int(draft.get("target_user_id") or 0)
        if field == "operator":
            current: object = target_user_id in self._task_close_operators
        elif field == "controlled_user":
            current = target_user_id in self._task_close_controlled_users
        elif field == "auto_close_time":
            current = str(self._task_close_settings.get(field, {}).get("value") or "20:00")
        elif field == "control_enabled_from":
            current = str(self._task_close_settings.get(field, {}).get("value") or "")
        else:
            return {"status": "invalid"}
        if current != draft.get("old_value"):
            return {"status": "conflict", "current_value": current, "draft": dict(draft)}

        new_value = draft.get("new_value")
        if field == "operator":
            if bool(new_value):
                self._task_close_operators.add(target_user_id)
            else:
                self._task_close_operators.discard(target_user_id)
        elif field == "controlled_user":
            if bool(new_value):
                self._task_close_controlled_users.add(target_user_id)
            else:
                self._task_close_controlled_users.discard(target_user_id)
        else:
            self._task_close_settings[field] = {
                "key": field,
                "value": str(new_value or ""),
                "updated_by": actor_user_id,
            }
        self._task_close_revisions.append(
            {
                "action": "confirm_admin_change",
                "actor_user_id": actor_user_id,
                "payload": {
                    "draft_id": draft_id,
                    "field": field,
                    "target_user_id": target_user_id or None,
                    "old_value": current,
                    "new_value": new_value,
                },
            }
        )
        self._drafts.pop(dialog_key, None)
        return {"status": "confirmed", "draft": dict(draft)}

    def get_task_close_control_setting(self, key: str) -> dict[str, Any] | None:
        setting = self._task_close_settings.get(key)
        return dict(setting) if setting else None

    def set_task_close_control_setting(self, *, key: str, value: str, updated_by: int | None = None) -> None:
        self._task_close_settings[key] = {"key": key, "value": value, "updated_by": updated_by}
        self._task_close_revisions.append(
            {"action": "set_setting", "actor_user_id": updated_by, "payload": {"key": key, "value": value}}
        )

    def task_close_operator_ids(self) -> set[int]:
        return set(self._task_close_operators)

    def set_task_close_operators(self, *, operator_user_ids: list[int], actor_user_id: int | None) -> list[int]:
        self._task_close_operators = {int(item) for item in operator_user_ids}
        saved = sorted(self._task_close_operators)
        self._task_close_revisions.append(
            {"action": "set_operators", "actor_user_id": actor_user_id, "payload": {"operator_user_ids": saved}}
        )
        return saved

    def upsert_task_close_operator(self, *, user_id: int, active: bool = True, updated_by: int | None = None) -> None:
        if active:
            self._task_close_operators.add(int(user_id))
        else:
            self._task_close_operators.discard(int(user_id))
        self._task_close_revisions.append(
            {
                "action": "set_operator",
                "actor_user_id": updated_by,
                "payload": {"user_id": int(user_id), "active": active},
            }
        )

    def task_close_controlled_user_ids(self) -> set[int]:
        return set(self._task_close_controlled_users)

    def upsert_task_close_controlled_user(
        self, *, user_id: int, active: bool = True, updated_by: int | None = None
    ) -> None:
        if active:
            self._task_close_controlled_users.add(user_id)
        else:
            self._task_close_controlled_users.discard(user_id)
        self._task_close_revisions.append(
            {
                "action": "set_controlled_user",
                "actor_user_id": updated_by,
                "payload": {"user_id": user_id, "active": active},
            }
        )
