from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ai_server.diagnostics import run_diagnostic_via_orchestrator  # noqa: E402
from ai_server.learning import LearningEventRecorder  # noqa: E402
from ai_server.models import AgentTask, UserContext  # noqa: E402
from ai_server.registry import load_agent_manifests  # noqa: E402
from ai_server.tracing import TraceRecorder  # noqa: E402


async def _run(args: argparse.Namespace) -> int:
    recorder = LearningEventRecorder()
    trace_recorder = TraceRecorder(enabled=False)
    task = AgentTask(
        task_id=str(uuid4()),
        source="codex_error_report_cli",
        user=UserContext(channel="diagnostics"),
        request=f"Сформируй отчет Диагноста по ошибкам за {args.since_hours} ч.",
        context={
            "error_report_request": {
                "since_hours": args.since_hours,
                "limit": args.limit,
                "max_groups": args.max_groups,
            }
        },
    )
    result = await run_diagnostic_via_orchestrator(
        manifests=load_agent_manifests(),
        task=task,
        trace_recorder=trace_recorder,
        learning_recorder=recorder,
    )
    report = {}
    for artifact in result.artifacts:
        if artifact.type == "diagnostic_error_report":
            report = artifact.metadata.get("report") if isinstance(artifact.metadata, dict) else {}
            break

    output = json.dumps(report, ensure_ascii=False, indent=2) if args.format == "json" else result.answer
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if result.status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Diagnostic Agent error report through orchestrator.")
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-groups", type=int, default=5)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default="")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
