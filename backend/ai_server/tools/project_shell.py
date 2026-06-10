from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_server.settings import Settings, get_settings

READ_ONLY_PREFIXES = (
    "dir",
    "ls",
    "gci",
    "get-childitem",
    "type ",
    "cat ",
    "gc ",
    "get-content",
    "rg",
    "select-string",
    "git status",
    "git diff",
    "git show",
    "git log",
)

BLOCKED_PATTERNS = (
    r"\.env\b",
    r"\$env:",
    r"env:",
    r"openai_api_key",
    r"bitrix_bot_token",
    r"bitrix_rest_webhook_url",
    r"secret_key",
    r"invoke-webrequest",
    r"invoke-restmethod",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bssh\b",
    r"\bscp\b",
    r"\bftp\b",
    r"\bcmd\s*/c\b",
    r"\bpowershell\b",
    r"\bpwsh\b",
    r"start-process",
    r"stop-process",
    r"taskkill",
    r"restart-computer",
    r"stop-computer",
    r"shutdown",
    r"format\b",
    r"diskpart",
    r"set-executionpolicy",
    r"\breg\b",
    r"\bicacls\b",
    r"\btakeown\b",
    r"add-type",
    r"new-object\s+net\.webclient",
    r"\[io\.",
    r"system\.io",
    r"reflection",
    r"\.\.",
)


@dataclass(frozen=True)
class ShellValidation:
    ok: bool
    reason: str = ""


class ProjectShellService:
    def __init__(self, settings: Settings | None = None, root: Path | None = None) -> None:
        self._settings = settings or get_settings()
        self.root = (root or Path.cwd()).resolve()

    async def run(self, command: str) -> dict[str, Any]:
        validation = self.validate(command)
        if not validation.ok:
            return {"status": "denied", "reason": validation.reason}

        settings = self._settings
        process = await asyncio.create_subprocess_exec(
            settings.agent_shell_executable,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
            cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=settings.agent_shell_timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return {
                "status": "error",
                "error": f"command timed out after {settings.agent_shell_timeout_seconds}s",
                "command": command,
            }

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return {
            "status": "executed",
            "command": command,
            "cwd": str(self.root),
            "returncode": process.returncode,
            "stdout": _truncate(stdout, settings.agent_shell_max_output_chars),
            "stderr": _truncate(stderr, settings.agent_shell_max_output_chars),
        }

    def validate(self, command: str) -> ShellValidation:
        settings = self._settings
        cleaned = command.strip()
        if not settings.agent_shell_enabled:
            return ShellValidation(False, "Project shell выключен (AGENT_SHELL_ENABLED).")
        if not cleaned:
            return ShellValidation(False, "Команда пустая.")
        if len(cleaned) > settings.agent_shell_max_command_chars:
            return ShellValidation(False, "Команда слишком длинная.")

        lowered = cleaned.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                return ShellValidation(False, f"Команда заблокирована политикой project_shell: {pattern}")

        for raw_path in re.findall(r"[A-Za-z]:\\[^\s'\"`]+", cleaned):
            try:
                path = Path(raw_path).resolve()
            except OSError:
                return ShellValidation(False, f"Не смог проверить путь: {raw_path}")
            if not _inside_root(path, self.root):
                return ShellValidation(False, f"Путь вне проекта запрещён: {raw_path}")

        return ShellValidation(True)

    def is_read_only(self, command: str) -> bool:
        lowered = command.strip().lower()
        if any(marker in lowered for marker in ("|", ">", ";", "&&", "||")):
            return False
        return any(lowered == prefix or lowered.startswith(prefix + " ") for prefix in READ_ONLY_PREFIXES)


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... output truncated ..."
