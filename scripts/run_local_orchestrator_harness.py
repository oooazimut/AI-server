"""Materialize an S03 corpus from DeepSeek receipts and deterministic fixtures."""
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


def receipts(paths: list[Path]) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")).get("results", []):
            values[str(item["case_id"])] = item
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--planner-response", action="append", default=[], type=Path)
    parser.add_argument("--final-response", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    planners, finals = receipts(args.planner_response), receipts(args.final_response)
    harness = LocalOrchestratorHarness()

    async def execute():
        output = []
        for item in matrix["cases"]:
            case_id, request = str(item["id"]), str(item["request"])
            planner = planners.get(case_id, {})
            composer = finals.get(case_id, {})
            result = await harness.run_case(
                case_id, request, planner.get("content") if planner.get("status") == "PASS" else None,
                composer.get("content") if composer.get("status") == "PASS" else None,
                fixture_mode=str(item.get("fixture_mode", "normal")), plan_id=plan_id(case_id, request),
            )
            result.model_tokens = sum(int(receipt.get("usage", {}).get("completion_tokens", 0)) for receipt in (planner, composer))
            output.append(result.__dict__)
        return output

    results = asyncio.run(execute())
    all_receipts = [*planners.values(), *finals.values()]
    payload = {
        "schema_version": 2, "session_id": "S03", "external_changes": False,
        "planner_receipts": [str(path) for path in args.planner_response], "final_receipts": [str(path) for path in args.final_response],
        "model_usage": {"calls": len(all_receipts), "completion_tokens": sum(int(item.get("usage", {}).get("completion_tokens", 0)) for item in all_receipts), "latency_ms": sum(int(item.get("elapsed_ms", 0)) for item in all_receipts)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda item: item.__dict__), encoding="utf-8")
    print(f"T0006_S03_LOCAL_HARNESS_COMPLETE cases={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
