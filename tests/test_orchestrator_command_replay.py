import json
from datetime import datetime
from pathlib import Path

from ai_server.orchestrators.bitrix_semantics import normalize_command_arguments
from ai_server.utils import MOSCOW_TZ


def test_human_request_replay_matrix_produces_stable_executor_commands():
    cases = json.loads(
        (Path(__file__).parent / "data" / "orchestrator_command_replay.json").read_text(encoding="utf-8")
    )
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)

    for case in cases:
        actual = normalize_command_arguments(case["tool"], case["request"], case["input"], now=now)
        for key, expected in case["expected"].items():
            assert actual.get(key) == expected, f"{case['request']}: {key}"
