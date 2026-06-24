from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from fusion_ai_council.config import ModelConfig, RuntimeConfig
from fusion_ai_council.models.base import is_error_response, is_timeout_response, run_command
from fusion_ai_council.types import ModelResult


FILE_URL_RE = re.compile(r"file://[^\s)\]>]+")
ARTIFACT_EXTENSIONS = {".md", ".markdown", ".txt"}
MAX_ARTIFACT_CHARS = 200_000


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def allowed_artifact_roots(project_dir: Path) -> list[Path]:
    roots = [project_dir.resolve()]
    gemini_brain = Path.home() / ".gemini" / "antigravity-cli" / "brain"
    if gemini_brain.exists():
        roots.append(gemini_brain.resolve())
    return roots


def artifact_path_from_url(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc:
        return None
    if not parsed.path:
        return None
    return Path(unquote(parsed.path)).expanduser()


def read_allowed_artifact(path: Path, allowed_roots: list[Path]) -> tuple[Path, str] | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None

    if not resolved.is_file() or resolved.suffix.lower() not in ARTIFACT_EXTENSIONS:
        return None
    if not any(is_relative_to(resolved, root) for root in allowed_roots):
        return None

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None

    if len(content) > MAX_ARTIFACT_CHARS:
        content = (
            content[:MAX_ARTIFACT_CHARS]
            + f"\n\n[TRUNCATED referenced artifact at {MAX_ARTIFACT_CHARS} characters]"
        )
    return resolved, content


def inline_referenced_artifacts(text: str, project_dir: Path) -> str:
    urls = list(dict.fromkeys(FILE_URL_RE.findall(text)))
    if not urls:
        return text

    allowed_roots = allowed_artifact_roots(project_dir)
    sections = []
    for url in urls:
        path = artifact_path_from_url(url)
        if path is None:
            continue
        artifact = read_allowed_artifact(path, allowed_roots)
        if artifact is None:
            continue
        resolved, content = artifact
        sections.append(
            "\n".join(
                [
                    f"## Included referenced artifact: {resolved}",
                    "",
                    content.strip(),
                ]
            )
        )

    if not sections:
        return text

    return "\n\n".join(
        [
            text.rstrip(),
            "---",
            "Referenced local artifact content included for synthesis:",
            *sections,
        ]
    )


def build_timeout_retry_prompt(original_prompt: str, partial_output: str) -> str:
    return f"""
The previous non-interactive model attempt timed out before returning a complete final answer.

Use the original task and context below. The partial transcript may contain useful notes or checks,
but it is not a final answer. Produce the complete final answer inline now.

Do not narrate internal tool use, subagent status, or process steps. Do not save the answer only in
an external artifact. Return the full substantive answer in this response.

Original task and context:
{original_prompt}

Partial timed-out transcript:
{partial_output}
""".strip()


@dataclass(frozen=True)
class AgyModelClient:
    key: str
    label: str
    model: str

    @classmethod
    def from_config(cls, config: ModelConfig) -> "AgyModelClient":
        return cls(key=config.key, label=config.label, model=config.model)

    def command(self, prompt: str, print_timeout_seconds: int = 1800) -> list[str]:
        return [
            "agy",
            "--model",
            self.model,
            "--print-timeout",
            f"{print_timeout_seconds}s",
            "-p",
            prompt,
        ]

    def run_prompt(self, prompt: str, runtime: RuntimeConfig) -> str:
        content = run_command(
            self.command(prompt, runtime.timeout_seconds),
            cwd=runtime.project_dir,
            timeout=runtime.timeout_seconds + 60,
        )
        return inline_referenced_artifacts(content, runtime.project_dir)

    def answer(self, prompt: str, runtime: RuntimeConfig) -> ModelResult:
        content = self.run_prompt(prompt, runtime)
        if is_timeout_response(content):
            retry_content = self.run_prompt(
                build_timeout_retry_prompt(prompt, content),
                runtime,
            )
            if is_error_response(retry_content):
                content = (
                    "[ERROR] agy timed out and the continuation retry did not produce "
                    "a complete answer.\n\n"
                    f"Initial timed-out output:\n{content}\n\n"
                    f"Continuation retry output:\n{retry_content}"
                )
            else:
                content = retry_content

        return ModelResult(
            key=self.key,
            label=self.label,
            model=self.model,
            file_name="",
            ok=not is_error_response(content),
            content=content,
        )
