from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from fusion_ai_council.config import ModelConfig, RuntimeConfig
from fusion_ai_council.models.base import clean_cli_output, is_error_response, run_command
from fusion_ai_council.types import ModelResult


@dataclass(frozen=True)
class CodexModelClient:
    key: str
    label: str
    model: str
    reasoning_effort: str

    @classmethod
    def from_config(cls, config: ModelConfig) -> "CodexModelClient":
        return cls(
            key=config.key,
            label=config.label,
            model=config.model,
            reasoning_effort=config.reasoning_effort or "xhigh",
        )

    def command(self, runtime: RuntimeConfig, output_path: Path | None = None) -> list[str]:
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--cd",
            str(runtime.project_dir),
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
        ]
        if output_path is not None:
            cmd.extend(["--output-last-message", str(output_path)])
        cmd.append("-")
        return cmd

    def answer(self, prompt: str, runtime: RuntimeConfig) -> ModelResult:
        with tempfile.TemporaryDirectory(prefix="fusion_codex_") as tmpdir:
            output_path = Path(tmpdir) / "last_message.md"
            output = run_command(
                self.command(runtime, output_path=output_path),
                cwd=runtime.project_dir,
                timeout=runtime.timeout_seconds,
                input_text=prompt,
            )
            if not is_error_response(output) and output_path.exists():
                content = output_path.read_text(encoding="utf-8").strip()
            else:
                content = clean_cli_output(output)

        return ModelResult(
            key=self.key,
            label=self.label,
            model=self.model,
            file_name="",
            ok=not is_error_response(content),
            content=content,
        )

