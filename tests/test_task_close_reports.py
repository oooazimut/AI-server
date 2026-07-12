from __future__ import annotations

import subprocess
import sys


def test_task_close_reports_imports_without_portal_search_cycle():
    code = (
        "from ai_server.integrations.bitrix.task_close_reports import "
        "canonical_task_close_report_file_name; "
        "assert canonical_task_close_report_file_name('AI-close-139-unconfirmed.txt') == 'AI-close-139.txt'"
    )

    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
