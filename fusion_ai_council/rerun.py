from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from fusion_ai_council.config import RuntimeConfig, build_runtime_config
from fusion_ai_council.context.modes import compose_model_prompt
from fusion_ai_council.judge.parse import extract_final_answer
from fusion_ai_council.judge.prompt import build_judge_prompt
from fusion_ai_council.models import build_model_client
from fusion_ai_council.models.base import is_error_response
from fusion_ai_council.storage import (
    INITIAL_FILE_NAMES,
    write_final_outputs,
    write_metadata,
    write_model_result,
)
from fusion_ai_council.types import ContextBundle, ContextSource, ModelResult


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def context_bundle_from_run(run_dir: Path, metadata: dict[str, Any]) -> ContextBundle:
    context_file = metadata.get("context_bundle_file") or "00_context_bundle.md"
    context_path = run_dir / context_file
    if not context_path.exists():
        raise FileNotFoundError(f"Context bundle not found: {context_path}")

    sources = [
        ContextSource(**source)
        for source in metadata.get("context_sources", [])
    ]
    return ContextBundle(
        text=context_path.read_text(encoding="utf-8"),
        sources=sources,
    )


def model_results_from_run(run_dir: Path, metadata: dict[str, Any]) -> list[ModelResult]:
    results = []
    for response in metadata.get("responses", []):
        file_name = response["file_name"]
        output_path = run_dir / file_name
        content = (
            output_path.read_text(encoding="utf-8")
            if output_path.exists()
            else response.get("content", "")
        )
        results.append(
            ModelResult(
                key=response["key"],
                label=response["label"],
                model=response["model"],
                file_name=file_name,
                ok=not is_error_response(content),
                content=content,
            )
        )
    return results


def runtime_for_existing_run(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    cwd: Path,
) -> RuntimeConfig:
    rerun_args = argparse.Namespace(**vars(args))
    rerun_args.question = [metadata["question"]]
    rerun_args.workdir = metadata["project_dir"]
    rerun_args.mode = metadata["mode"]
    rerun_args.context = []
    rerun_args.no_follow_links = not bool(metadata.get("follow_links", True))
    rerun_args.no_auto_context = not bool(metadata.get("auto_context", True))
    rerun_args.dry_run_context = False
    rerun_args.context_budget_chars = (
        args.context_budget_chars
        if args.context_budget_chars is not None
        else metadata.get("context_budget_chars")
    )
    rerun_args.max_context_file_chars = (
        args.max_context_file_chars
        if args.max_context_file_chars is not None
        else metadata.get("max_context_file_chars")
    )
    return build_runtime_config(rerun_args, cwd=cwd)


def rerun_model_in_run(
    run_dir: Path,
    model_key: str,
    runtime: RuntimeConfig,
    context_bundle: ContextBundle,
    existing_results: list[ModelResult],
) -> list[ModelResult]:
    if model_key not in runtime.models:
        raise ValueError(f"Unknown model key: {model_key}")
    if model_key not in runtime.initial_model_keys:
        raise ValueError(f"Can only rerun initial worker models: {model_key}")

    prompt = compose_model_prompt(runtime.question, runtime.mode, context_bundle)
    client = build_model_client(runtime.models[model_key])
    index = runtime.initial_model_keys.index(model_key) + 1
    file_name = INITIAL_FILE_NAMES.get(model_key, f"{index:02d}_{model_key}.md")
    backup_file(run_dir / file_name)
    result = write_model_result(run_dir, index, client.answer(prompt, runtime))

    replaced = False
    updated_results = []
    for existing in existing_results:
        if existing.key == model_key:
            updated_results.append(result)
            replaced = True
        else:
            updated_results.append(existing)
    if not replaced:
        updated_results.append(result)
    return updated_results


def run_judge_for_existing_run(
    run_dir: Path,
    runtime: RuntimeConfig,
    context_bundle: ContextBundle,
    responses: list[ModelResult],
) -> tuple[str, str, str, list[ModelResult]]:
    successful = [response for response in responses if response.ok]
    if not successful:
        raw_output = "[ERROR] All worker outputs failed; skipping judge."
        return raw_output, raw_output, "", []

    judge_client = build_model_client(runtime.models[runtime.judge])
    prompt = build_judge_prompt(runtime.question, successful, runtime.mode, context_bundle)
    judge_result = judge_client.answer(prompt, runtime)
    final = extract_final_answer(judge_result.content)

    backup_file(run_dir / "04_judge_raw.md")
    backup_file(run_dir / "04_final.md")
    write_final_outputs(run_dir, judge_result.content, final)
    return judge_result.content, final, judge_client.model, successful


def rerun_existing_run(args: argparse.Namespace, cwd: Path | None = None) -> None:
    cwd = Path.cwd() if cwd is None else cwd
    run_dir = Path(args.rerun_run).expanduser()
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"--rerun-run is not a directory: {run_dir}")

    if not args.rerun_model and not args.rejudge:
        raise ValueError("Use --rerun-model, --rejudge, or both with --rerun-run.")

    metadata = load_run_metadata(run_dir)
    runtime = runtime_for_existing_run(args, metadata, cwd)
    context_bundle = context_bundle_from_run(run_dir, metadata)
    context_bundle_file = metadata.get("context_bundle_file") or "00_context_bundle.md"
    responses = model_results_from_run(run_dir, metadata)

    if args.rerun_model:
        print(f"Rerunning {args.rerun_model} for existing run...", flush=True)
        responses = rerun_model_in_run(
            run_dir,
            args.rerun_model,
            runtime,
            context_bundle,
            responses,
        )

    raw_judge_output = (run_dir / "04_judge_raw.md").read_text(encoding="utf-8")
    final = (run_dir / "04_final.md").read_text(encoding="utf-8")
    judge_model = metadata.get("judge_model")
    judge_backend = metadata.get("judge_backend")
    judge_responses = [
        response for response in responses
        if response.ok
    ]

    if args.rejudge:
        print(f"Rerunning {runtime.judge} judge for existing run...", flush=True)
        raw_judge_output, final, judge_model, judge_responses = run_judge_for_existing_run(
            run_dir,
            runtime,
            context_bundle,
            responses,
        )
        judge_backend = runtime.judge

    write_metadata(
        run_dir,
        runtime,
        context_bundle,
        context_bundle_file,
        responses,
        judge_responses,
        judge_backend,
        judge_model,
        raw_judge_output,
        final,
    )

    print(f"Updated run: {run_dir}")
