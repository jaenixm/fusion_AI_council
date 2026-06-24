from __future__ import annotations

from dataclasses import dataclass

from fusion_ai_council.config import ModelConfig, RuntimeConfig
from fusion_ai_council.models.base import clean_cli_output, is_error_response, run_command
from fusion_ai_council.types import ModelResult


@dataclass(frozen=True)
class ClaudeModelClient:
    key: str
    label: str
    model: str
    reasoning_effort: str

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ClaudeModelClient":
        return cls(
            key=config.key,
            label=config.label,
            model=config.model,
            reasoning_effort=config.reasoning_effort or "xhigh",
        )

    def command(self) -> list[str]:
        return [
            "claude",
            "--print",
            "--model",
            self.model,
            "--effort",
            self.reasoning_effort,
            "--input-format",
            "text",
        ]

    def answer(self, prompt: str, runtime: RuntimeConfig) -> ModelResult:
        output = run_command(
            self.command(),
            cwd=runtime.project_dir,
            timeout=runtime.timeout_seconds,
            input_text=prompt,
        )
        content = clean_cli_output(output) if is_error_response(output) else output.strip()
        return ModelResult(
            key=self.key,
            label=self.label,
            model=self.model,
            file_name="",
            ok=not is_error_response(content),
            content=content,
        )
