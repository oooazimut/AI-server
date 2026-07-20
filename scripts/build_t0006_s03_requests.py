"""Build bounded DeepSeek planner or final-synthesis request batches for S03."""
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


def load_receipts(paths: list[Path]) -> dict[str, dict]:
    data = {}
    for path in paths:
        for result in json.loads(path.read_text(encoding="utf-8")).get("results", []):
            data[str(result["case_id"])] = result
    return data


def item(case_id: str, system: str, payload: dict) -> dict:
    return {"case_id": case_id, "max_tokens": 4000, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("planner", "final"))
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--planner-response", action="append", default=[], type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--repair-reason", action="append", default=[])
    parser.add_argument("--continuations", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.matrix.read_text(encoding="utf-8"))["cases"]
    selected = [case for case in cases if str(case["id"]) in set(args.case_id)] if args.case_id else cases[(args.batch - 1) * 8:args.batch * 8]
    if not selected:
        raise SystemExit("EMPTY_BATCH")
    continuations = json.loads(args.continuations.read_text(encoding="utf-8")) if args.continuations else {}
    if args.phase == "planner":
        repairs = dict(value.split("=", 1) for value in args.repair_reason)
        requests = []
        for case in selected:
            case_id = str(case["id"])
            continuation = continuations.get(case_id, {})
            request_text = str(continuation.get("request") or case["request"])
            output_case_id = str(continuation.get("receipt_case_id") or case_id)
            payload = planner_prompt(plan_id=plan_id(output_case_id, request_text), request=request_text, dialog_history=continuation.get("dialog_history") or [], fixture_mode=str(case.get("fixture_mode", "normal")), clarification_resolved=bool(continuation.get("clarification_resolved")))
            if continuation:
                payload["continuation"] = {"same_task": True, "clarification_resolved": bool(continuation.get("clarification_resolved"))}
            if case_id in repairs:
                payload["repair"] = {"previous_rejection": repairs[case_id], "instruction": "Correct the plan and return the required_response schema exactly. Do not retain invalid fields or dispatch when hard_constraints require clarification."}
            requests.append(item(output_case_id, "Return only one strict JSON object. Do not use markdown or tool calls. Obey the required_response schema exactly; unknown fields or a prohibited capability will be rejected before any executor runs.", payload))
    else:
        planners = load_receipts(args.planner_response)
        harness = LocalOrchestratorHarness()
        async def build_final():
            requests = []
            for case in selected:
                case_id, request = str(case["id"]), str(case["request"])
                receipt = planners.get(case_id, {})
                if receipt.get("status") != "PASS":
                    continue
                result = await harness.run_case(case_id, request, str(receipt.get("content") or ""), fixture_mode=str(case.get("fixture_mode", "normal")), plan_id=plan_id(case_id, request))
                if result.executor_calls:
                    requests.append(item(case_id, "Return only one strict JSON object. Do not use markdown. Include every executed subtask id and use only executor facts supplied in the payload.", final_prompt(plan_id=plan_id(case_id, request), response_hash=sha256_text(str(receipt.get("content") or "")), request=request, results=result.branches)))
            return requests
        requests = asyncio.run(build_final())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema_version": 1, "task_id": "T-0006", "session_id": "S03", "requests": requests}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"T0006_S03_{args.phase.upper()}_REQUESTS cases={len(requests)}")


if __name__ == "__main__":
    main()
