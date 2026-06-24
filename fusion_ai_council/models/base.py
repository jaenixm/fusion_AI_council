from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fusion_ai_council.config import RuntimeConfig
from fusion_ai_council.types import ModelResult


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    output: str


class ModelClient(Protocol):
    key: str
    label: str
    model: str

    def answer(self, prompt: str, runtime: RuntimeConfig) -> ModelResult:
        ...


def is_error_response(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("[ERROR]"):
        return True

    return is_timeout_response(stripped)


def is_timeout_response(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False

    lower_text = stripped.lower()
    if lower_text.startswith("[error] command timed out"):
        return True
    if lower_text.endswith("error: timed out waiting for response"):
        return True

    last_line = lines[-1].lower()
    return last_line.startswith("error: timed out waiting for response")


def run_command(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    input_text: str | None = None,
) -> str:
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(cwd),
        )
    except FileNotFoundError:
        return f"[ERROR] Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out."

    if result.returncode != 0:
        return (
            "[ERROR] Command failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def clean_cli_output(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    skip_prefixes = (
        "OpenAI Codex",
        "--------",
        "workdir:",
        "model:",
        "provider:",
        "approval:",
        "sandbox:",
        "reasoning effort:",
        "reasoning summaries:",
        "session id:",
        "tokens used",
        "WARNING: proceeding, even though we could not create PATH aliases:",
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if stripped in {"user", "codex"}:
            continue
        if any(stripped.startswith(prefix) for prefix in skip_prefixes):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()
