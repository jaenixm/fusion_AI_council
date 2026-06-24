from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fusion_ai_council.config import SUPPORTED_MODES, build_runtime_config
from fusion_ai_council.rerun import rerun_existing_run
from fusion_ai_council.runner import run_council


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-model fusion council over a prompt and optional project context."
    )
    parser.add_argument("question", nargs="*", help="User task/question for the council.")
    parser.add_argument("--config", default=None, help="YAML config file.")
    parser.add_argument(
        "--workdir",
        default=str(Path.cwd()),
        help="Project/repo directory the models should use as their working directory.",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="general",
        help="Prompt/context strategy to use.",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Context file or directory to include. May be repeated.",
    )
    parser.add_argument(
        "--context-budget-chars",
        type=int,
        default=None,
        help="Total character budget for included context.",
    )
    parser.add_argument(
        "--max-context-file-chars",
        type=int,
        default=None,
        help="Per-file character budget for included context.",
    )
    parser.add_argument(
        "--no-follow-links",
        action="store_true",
        help="Do not include local markdown/file links from context files.",
    )
    parser.add_argument(
        "--no-auto-context",
        action="store_true",
        help="Disable mode-specific automatic context files.",
    )
    parser.add_argument(
        "--dry-run-context",
        action="store_true",
        help="Build and save the context bundle, print its source manifest, then exit.",
    )
    parser.add_argument(
        "--rerun-run",
        default=None,
        help="Existing run directory to update instead of starting a new run.",
    )
    parser.add_argument(
        "--rerun-model",
        choices=("gemini", "claude", "codex"),
        default=None,
        help="Rerun one initial worker in --rerun-run using current config.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="Rerun the judge for --rerun-run using current worker outputs.",
    )
    parser.add_argument(
        "--judge",
        choices=("codex", "claude", "gemini"),
        default=None,
        help="Override FUSION_JUDGE for this run.",
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Override FUSION_CODEX_MODEL for this run.",
    )
    parser.add_argument(
        "--codex-reasoning",
        default=None,
        help="Override FUSION_CODEX_REASONING_EFFORT for this run.",
    )
    parser.add_argument(
        "--claude-model",
        default=None,
        help="Override FUSION_CLAUDE_MODEL for this run.",
    )
    parser.add_argument(
        "--claude-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Override FUSION_CLAUDE_EFFORT for this run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.rerun_run:
        try:
            rerun_existing_run(args)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(2)
        return

    if not " ".join(args.question).strip():
        print('Usage: fusion-council [options] "Your question here"')
        sys.exit(1)

    try:
        runtime = build_runtime_config(args)
        run_council(runtime)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
