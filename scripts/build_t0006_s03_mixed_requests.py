"""Build one bounded S03 batch containing planner repairs and final render choices."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ai_server.orchestrators.local_harness import LocalOrchestratorHarness, final_prompt, planner_prompt, sha256_text


def plan_id(case_id: str, request: str) -> str:
    return f"plan-{case_id.lower()}-{sha256_text(request)[:16]}"


def receipts(paths: list[Path]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")).get("results", []):
            content = str(item.get("content") or "")
            if "t0006.plan.v1" in content:
                output[str(item["case_id"])] = item
    return output


def model_item(case_id: str, system: str, payload: dict) -> dict:
    return {"case_id": case_id, "max_tokens": 4000, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--planner-response", action="append", required=True, type=Path)
    parser.add_argument("--final-case", action="append", default=[])
    parser.add_argument("--repair-case", action="append", default=[])
    parser.add_argument("--repair-reason", action="append", default=[])
    parser.add_argument("--continuations", type=Path)
    parser.add_argument("--continuation-final-case", action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    matrix = {str(case["id"]): case for case in json.loads(args.matrix.read_text(encoding="utf-8"))["cases"]}
    current = receipts(args.planner_response)
    continuations = json.loads(args.continuations.read_text(encoding="utf-8")) if args.continuations else {}
    repair_reasons = dict(value.split("=", 1) for value in args.repair_reason)
    requests: list[dict] = []
    for case_id in args.repair_case:
        case = matrix[case_id]
        request = str(case["request"])
        payload = planner_prompt(plan_id=plan_id(case_id, request), request=request, dialog_history=[], fixture_mode=str(case.get("fixture_mode", "normal")))
        payload["repair"] = {"previous_rejection": repair_reasons[case_id], "instruction": "Return one valid required_response only. Preserve the requested state semantics and do not use empty input, free prose, or prohibited fields."}
        requests.append(model_item(case_id, "Return only one strict JSON object. Do not use markdown or tool calls. Obey required_response exactly.", payload))
    harness = LocalOrchestratorHarness()

    async def add_finals() -> None:
        for case_id in args.final_case:
            case = matrix[case_id]
            request = str(case["request"])
            receipt = current.get(case_id, {})
            if receipt.get("status") != "PASS":
                raise RuntimeError(f"PLANNER_RECEIPT_UNAVAILABLE:{case_id}")
            result = await harness.run_case(case_id, request, str(receipt.get("content") or ""), fixture_mode=str(case.get("fixture_mode", "normal")), plan_id=plan_id(case_id, request))
            if not result.executor_calls or result.plan_validation.get("status") != "ACCEPT":
                raise RuntimeError(f"FINAL_SOURCE_NOT_EXECUTABLE:{case_id}")
            payload = final_prompt(plan_id=plan_id(case_id, request), response_hash=result.correlation_ids["response_hash"], request=request, results=result.branches)
            requests.append(model_item(case_id, "Return only one strict JSON object. Do not use markdown or free text. Select an ordering containing every supplied subtask id exactly once.", payload))
        for parent_case_id in args.continuation_final_case:
            case = matrix[parent_case_id]
            continuation = continuations[parent_case_id]
            initial_receipt = current.get(parent_case_id, {})
            continuation_case_id = str(continuation["receipt_case_id"])
            continuation_receipt = current.get(continuation_case_id, {})
            if initial_receipt.get("status") != "PASS" or continuation_receipt.get("status") != "PASS":
                raise RuntimeError(f"CONTINUATION_RECEIPT_UNAVAILABLE:{parent_case_id}")
            initial = await harness.run_case(parent_case_id, str(case["request"]), str(initial_receipt.get("content") or ""), fixture_mode=str(case.get("fixture_mode", "normal")), plan_id=plan_id(parent_case_id, str(case["request"])))
            if initial.verdict != "CLARIFICATION_REQUIRED" or initial.executor_calls:
                raise RuntimeError(f"CONTINUATION_INITIAL_INVALID:{parent_case_id}")
            request = str(continuation["request"])
            resumed = await harness.resume(initial.task_id, str(continuation["user_answer"]), str(continuation_receipt.get("content") or ""), plan_id=plan_id(continuation_case_id, request))
            if not resumed.executor_calls or resumed.task_id != initial.task_id:
                raise RuntimeError(f"CONTINUATION_RESUME_INVALID:{parent_case_id}")
            payload = final_prompt(plan_id=plan_id(continuation_case_id, request), response_hash=resumed.correlation_ids["response_hash"], request=request, results=resumed.branches)
            requests.append(model_item(continuation_case_id, "Return only one strict JSON object. Do not use markdown or free text. Select an ordering containing every supplied subtask id exactly once.", payload))
    asyncio.run(add_finals())
    if not 1 <= len(requests) <= 8:
        raise RuntimeError(f"BATCH_SIZE_INVALID:{len(requests)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema_version": 1, "task_id": "T-0006", "session_id": "S03", "requests": requests}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"T0006_S03_MIXED_REQUESTS cases={len(requests)}")


if __name__ == "__main__":
    main()
