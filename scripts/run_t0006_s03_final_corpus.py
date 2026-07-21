"""Materialize the complete S03 corpus from immutable model receipts."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ai_server.orchestrators.local_harness import LocalOrchestratorHarness, sha256_text


def plan_id(case_id: str, request: str) -> str:
    return f"plan-{case_id.lower()}-{sha256_text(request)[:16]}"


def classify(paths: list[Path]) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    plans: dict[str, dict] = {}
    finals: dict[str, dict] = {}
    all_receipts: list[dict] = []
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")).get("results", []):
            all_receipts.append(item)
            content = str(item.get("content") or "")
            if "t0006.plan.v1" in content:
                plans[str(item["case_id"])] = item
            else:
                finals[str(item["case_id"])] = item
    return plans, finals, all_receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--continuations", required=True, type=Path)
    parser.add_argument("--response", action="append", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    cases = json.loads(args.matrix.read_text(encoding="utf-8"))["cases"]
    continuations = json.loads(args.continuations.read_text(encoding="utf-8"))
    plans, finals, all_receipts = classify(args.response)
    harness = LocalOrchestratorHarness()

    async def execute():
        results = []
        continuation_traces = []
        for case in cases:
            case_id, request = str(case["id"]), str(case["request"])
            receipt = plans.get(case_id, {})
            initial = await harness.run_case(case_id, request, receipt.get("content") if receipt.get("status") == "PASS" else None, finals.get(case_id, {}).get("content") if finals.get(case_id, {}).get("status") == "PASS" else None, fixture_mode=str(case.get("fixture_mode", "normal")), plan_id=plan_id(case_id, request))
            continuation = continuations.get(case_id)
            if continuation and initial.verdict == "CLARIFICATION_REQUIRED":
                continuation_id = str(continuation["receipt_case_id"])
                continuation_request = str(continuation["request"])
                continuation_receipt = plans.get(continuation_id, {})
                final_receipt = finals.get(continuation_id, {})
                resumed = await harness.resume(initial.task_id, str(continuation["user_answer"]), continuation_receipt.get("content") if continuation_receipt.get("status") == "PASS" else "", final_receipt.get("content") if final_receipt.get("status") == "PASS" else None, plan_id=plan_id(continuation_id, continuation_request))
                resumed.correlation_ids["initial_task_id"] = initial.task_id
                resumed.notes.append("same-task continuation after successful zero-call clarification")
                continuation_traces.append({"case_id": case_id, "initial": initial, "continuation": resumed, "same_task_id": initial.task_id == resumed.task_id})
                initial = resumed
            results.append(initial)
        return results, continuation_traces

    results, continuation_traces = asyncio.run(execute())
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 3,
        "session_id": "S03",
        "external_changes": False,
        "responses": [str(path) for path in args.response],
        "model_budget": {"calls_used": ledger["calls_used"], "calls_remaining": ledger["calls_remaining"], "receipt_calls": len(all_receipts), "completion_tokens": sum(int(item.get("usage", {}).get("completion_tokens", 0)) for item in all_receipts), "latency_ms": sum(int(item.get("elapsed_ms", 0)) for item in all_receipts)},
        "causal_proof": {"executor_calls": harness.executor_calls, "all_executor_calls_bound": all(all(record[key] for key in ("response_hash", "plan_id", "subtask_id", "attempt_id")) for record in harness.executor_calls)},
        "continuation_traces": continuation_traces,
        "results": [result.__dict__ for result in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.__dict__), encoding="utf-8")
    print(f"T0006_S03_FINAL_CORPUS_COMPLETE cases={len(results)} executor_calls={len(harness.executor_calls)}")


if __name__ == "__main__":
    main()
