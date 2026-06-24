from __future__ import annotations

import datetime
import json
from dataclasses import asdict
from pathlib import Path

from fusion_ai_council.config import RuntimeConfig
from fusion_ai_council.types import ContextBundle, ModelResult


INITIAL_FILE_NAMES = {
    "gemini": "01_gemini.md",
    "claude": "02_claude.md",
    "codex": "03_codex_initial.md",
}


def create_run_dir(runs_dir: Path) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_context_bundle(run_dir: Path, context_bundle: ContextBundle) -> str | None:
    if not context_bundle.text:
        return None
    file_name = "00_context_bundle.md"
    (run_dir / file_name).write_text(context_bundle.text, encoding="utf-8")
    return file_name


def write_model_result(run_dir: Path, index: int, result: ModelResult) -> ModelResult:
    file_name = INITIAL_FILE_NAMES.get(result.key, f"{index:02d}_{result.key}.md")
    (run_dir / file_name).write_text(result.content, encoding="utf-8")
    return ModelResult(
        key=result.key,
        label=result.label,
        model=result.model,
        file_name=file_name,
        ok=result.ok,
        content=result.content,
    )


def write_final_outputs(run_dir: Path, raw_judge_output: str, final: str) -> None:
    (run_dir / "04_judge_raw.md").write_text(raw_judge_output, encoding="utf-8")
    (run_dir / "04_final.md").write_text(final, encoding="utf-8")


def write_metadata(
    run_dir: Path,
    runtime: RuntimeConfig,
    context_bundle: ContextBundle,
    context_bundle_file: str | None,
    initial_responses: list[ModelResult],
    judge_responses: list[ModelResult],
    judge_backend: str | None,
    judge_model: str | None,
    raw_judge_output: str,
    final: str,
) -> None:
    metadata = {
        "question": runtime.question,
        "mode": runtime.mode,
        "project_dir": str(runtime.project_dir),
        "context_bundle_file": context_bundle_file,
        "context_sources": [asdict(source) for source in context_bundle.sources],
        "context_budget_chars": runtime.context_budget_chars,
        "max_context_file_chars": runtime.max_context_file_chars,
        "follow_links": runtime.follow_links,
        "auto_context": runtime.auto_context,
        "judge_backend": judge_backend,
        "judge_model": judge_model,
        "judge_raw_file": "04_judge_raw.md",
        "final_answer_extracted": final != raw_judge_output,
        "judge_response_order": [
            {
                "candidate": index,
                "key": response.key,
                "label": response.label,
                "model": response.model,
                "file_name": response.file_name,
            }
            for index, response in enumerate(judge_responses, start=1)
        ],
        "final_ok": not final.lstrip().startswith("[ERROR]"),
        "run_dir": str(run_dir),
        "models": {
            key: asdict(model)
            for key, model in runtime.models.items()
        },
        "responses": [asdict(response) for response in initial_responses],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
