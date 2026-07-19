"""Run the approved T-0006 S02 matrix using local scripted adapters only."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ai_server.orchestrators.local_harness import LocalOrchestratorHarness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-response", action="append", default=[], type=Path)
    parser.add_argument("--model-budget-ledger", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    model_receipts = {}
    for response_path in args.model_response:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        for receipt in response.get("results", []):
            model_receipts[str(receipt["case_id"])] = receipt
    harness = LocalOrchestratorHarness(model_receipts=model_receipts)

    async def execute():
        return [await harness.run_case(item["id"], item["request"]) for item in matrix["cases"]]

    results = asyncio.run(execute())
    corpus_completion_tokens = sum(int(item.get("usage", {}).get("completion_tokens", 0)) for item in model_receipts.values())
    corpus_total_tokens = sum(int(item.get("usage", {}).get("total_tokens", 0)) for item in model_receipts.values())
    budget = json.loads(args.model_budget_ledger.read_text(encoding="utf-8")) if args.model_budget_ledger else {}
    auth_calls = int((budget.get("case_calls") or {}).get("__auth__", 0))
    payload = {
        "schema_version": 1,
        "session_id": "S02",
        "external_changes": False,
        "fixture_origin": "approved staging receipt; id/title/active only; all contents/stock/logistics/failure values are scripted",
        "model_receipts": [str(path) for path in args.model_response],
        "model_budget": {
            "calls_limit": budget.get("calls_limit"),
            "calls_used": budget.get("calls_used"),
            "calls_remaining": budget.get("calls_remaining"),
            "corpus_case_calls": len(model_receipts),
            "corpus_completion_tokens": corpus_completion_tokens,
            "corpus_total_tokens": corpus_total_tokens,
            "non_corpus_records": [
                {
                    "case_id": "__auth__",
                    "batch": "pre-stage-auth",
                    "calls": auth_calls,
                    "status": "PRE_STAGE_AUTH_PROBE",
                    "completion_tokens": None,
                    "total_tokens": None,
                    "note": "Ledgered budget call outside the 16-case corpus; its token receipt is not retained here.",
                }
            ] if auth_calls else [],
        },
        "results": [result.__dict__ for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.__dict__), encoding="utf-8")
    print(f"T0006_S02_LOCAL_HARNESS_COMPLETE cases={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
