from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from fusion_ai_council.config import RuntimeConfig
from fusion_ai_council.context.bundle import build_context_bundle
from fusion_ai_council.context.modes import compose_model_prompt
from fusion_ai_council.judge.parse import extract_final_answer
from fusion_ai_council.judge.prompt import build_judge_prompt
from fusion_ai_council.models import build_model_client
from fusion_ai_council.models.base import is_error_response
from fusion_ai_council.storage import (
    create_run_dir,
    write_context_bundle,
    write_final_outputs,
    write_metadata,
    write_model_result,
)
from fusion_ai_council.types import ModelResult


@dataclass(frozen=True)
class CouncilRun:
    run_dir: str
    final: str | None
    context_only: bool


def shuffled_for_judge(responses: list[ModelResult]) -> list[ModelResult]:
    shuffled = list(responses)
    random.shuffle(shuffled)
    return shuffled


def run_initial_responses(prompt: str, runtime: RuntimeConfig, run_dir) -> list[ModelResult]:
    clients = [
        build_model_client(runtime.models[key])
        for key in runtime.initial_model_keys
    ]
    responses: dict[str, ModelResult] = {}

    print("Running initial model calls in parallel...", flush=True)
    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        future_to_client = {
            executor.submit(client.answer, prompt, runtime): client
            for client in clients
        }

        for future in as_completed(future_to_client):
            client = future_to_client[future]
            try:
                result = future.result()
            except Exception as exc:
                result = ModelResult(
                    key=client.key,
                    label=client.label,
                    model=client.model,
                    file_name="",
                    ok=False,
                    content=f"[ERROR] {client.label} raised {type(exc).__name__}: {exc}",
                )

            index = runtime.initial_model_keys.index(client.key) + 1
            result = write_model_result(run_dir, index, result)
            responses[client.key] = result
            status = "ok" if result.ok else "failed"
            print(f"Finished {client.label}: {status}", flush=True)

    return [responses[key] for key in runtime.initial_model_keys]


def run_judge(
    question: str,
    judge_responses: list[ModelResult],
    runtime: RuntimeConfig,
    context_bundle,
) -> tuple[str, str, str]:
    judge_client = build_model_client(runtime.models[runtime.judge])
    prompt = build_judge_prompt(question, judge_responses, runtime.mode, context_bundle)
    result = judge_client.answer(prompt, runtime)
    return result.content, judge_client.model, judge_client.key


def run_council(runtime: RuntimeConfig) -> CouncilRun:
    run_dir = create_run_dir(runtime.runs_dir)
    context_bundle = build_context_bundle(
        runtime.context_paths,
        runtime.project_dir,
        runtime.mode,
        follow_links=runtime.follow_links,
        auto_context=runtime.auto_context,
        budget_chars=runtime.context_budget_chars,
        max_file_chars=runtime.max_context_file_chars,
    )
    context_bundle_file = write_context_bundle(run_dir, context_bundle)

    if runtime.dry_run_context:
        print(f"Context bundle saved to: {run_dir / '00_context_bundle.md'}")
        print(f"Included sources: {len(context_bundle.sources)}")
        for source in context_bundle.sources:
            status = "truncated" if source.truncated else "full"
            print(
                f"- {source.path} ({source.reason}, {source.included_chars}/{source.chars}, {status})"
            )
        return CouncilRun(run_dir=str(run_dir), final=None, context_only=True)

    model_prompt = compose_model_prompt(runtime.question, runtime.mode, context_bundle)
    initial_responses = run_initial_responses(model_prompt, runtime, run_dir)
    successful_responses = [
        response for response in initial_responses
        if response.ok
    ]
    failed_responses = [
        response for response in initial_responses
        if not response.ok
    ]
    _ = failed_responses  # retained for clarity: failures are saved, not shown to the judge.

    judge_responses = shuffled_for_judge(successful_responses)
    if judge_responses:
        print(f"Running {runtime.judge} judge...", flush=True)
        raw_judge_output, judge_model, judge_backend = run_judge(
            runtime.question,
            judge_responses,
            runtime,
            context_bundle,
        )
        final = extract_final_answer(raw_judge_output)
    else:
        judge_model = None
        judge_backend = None
        final = "[ERROR] All initial model calls failed; skipping judge."
        raw_judge_output = final

    write_final_outputs(run_dir, raw_judge_output, final)
    write_metadata(
        run_dir,
        runtime,
        context_bundle,
        context_bundle_file,
        initial_responses,
        judge_responses,
        judge_backend,
        judge_model,
        raw_judge_output,
        final,
    )

    print("\n=== FINAL FUSION ANSWER ===\n")
    print(final)
    print(f"\nSaved run to: {run_dir}")
    return CouncilRun(run_dir=str(run_dir), final=final, context_only=False)

