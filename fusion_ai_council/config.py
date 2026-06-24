from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_MODES = ("general", "coding", "scientific-writing")
SUPPORTED_JUDGES = ("codex", "claude", "gemini")


DEFAULT_CONFIG: dict[str, Any] = {
    "judge": "codex",
    "models": {
        "gemini": {
            "backend": "agy",
            "model": "Gemini 3.1 Pro (High)",
            "label": "Gemini 3.1 Pro High",
        },
        "claude": {
            "backend": "claude",
            "model": "opus",
            "reasoning_effort": "xhigh",
            "label": "Claude Opus 4.8 CLI xhigh",
        },
        "codex": {
            "backend": "codex",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "label": "Codex gpt-5.5 xhigh",
        },
    },
    "context": {
        "budget_chars": {
            "general": 150_000,
            "coding": 200_000,
            "scientific-writing": 600_000,
        },
        "max_file_chars": {
            "general": 60_000,
            "coding": 100_000,
            "scientific-writing": 180_000,
        },
    },
}


@dataclass(frozen=True)
class ModelConfig:
    key: str
    backend: str
    model: str
    label: str
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    question: str
    project_dir: Path
    runs_dir: Path
    mode: str
    context_paths: list[str]
    follow_links: bool
    auto_context: bool
    dry_run_context: bool
    context_budget_chars: int
    max_context_file_chars: int
    judge: str
    models: dict[str, ModelConfig]
    initial_model_keys: tuple[str, ...] = ("gemini", "claude", "codex")
    timeout_seconds: int = 1800


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | None, cwd: Path) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = cwd / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


def env_overlay(env: dict[str, str]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    if env.get("FUSION_JUDGE"):
        overlay["judge"] = env["FUSION_JUDGE"]

    codex_overlay: dict[str, Any] = {}
    if env.get("FUSION_CODEX_MODEL"):
        codex_overlay["model"] = env["FUSION_CODEX_MODEL"]
    if env.get("FUSION_CODEX_REASONING_EFFORT"):
        codex_overlay["reasoning_effort"] = env["FUSION_CODEX_REASONING_EFFORT"]
    if codex_overlay:
        overlay["models"] = {"codex": codex_overlay}

    claude_overlay: dict[str, Any] = {}
    if env.get("FUSION_CLAUDE_MODEL"):
        claude_overlay["model"] = env["FUSION_CLAUDE_MODEL"]
    if env.get("FUSION_CLAUDE_EFFORT"):
        claude_overlay["reasoning_effort"] = env["FUSION_CLAUDE_EFFORT"]
    if env.get("FUSION_CLAUDE_REASONING_EFFORT"):
        claude_overlay["reasoning_effort"] = env["FUSION_CLAUDE_REASONING_EFFORT"]
    if claude_overlay:
        overlay["models"] = deep_merge(overlay.get("models", {}), {"claude": claude_overlay})

    context_overlay: dict[str, Any] = {}
    if env.get("FUSION_CONTEXT_BUDGET_CHARS"):
        context_overlay["budget_chars"] = {
            mode: int(env["FUSION_CONTEXT_BUDGET_CHARS"])
            for mode in SUPPORTED_MODES
        }
    if env.get("FUSION_MAX_CONTEXT_FILE_CHARS"):
        context_overlay["max_file_chars"] = {
            mode: int(env["FUSION_MAX_CONTEXT_FILE_CHARS"])
            for mode in SUPPORTED_MODES
        }
    if context_overlay:
        overlay["context"] = context_overlay

    return overlay


def cli_overlay(args: Any) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    if args.judge:
        overlay["judge"] = args.judge

    codex_overlay: dict[str, Any] = {}
    if args.codex_model:
        codex_overlay["model"] = args.codex_model
    if args.codex_reasoning:
        codex_overlay["reasoning_effort"] = args.codex_reasoning
    if codex_overlay:
        overlay["models"] = {"codex": codex_overlay}

    claude_overlay: dict[str, Any] = {}
    if args.claude_model:
        claude_overlay["model"] = args.claude_model
    if args.claude_effort:
        claude_overlay["reasoning_effort"] = args.claude_effort
    if claude_overlay:
        overlay["models"] = deep_merge(overlay.get("models", {}), {"claude": claude_overlay})

    context_overlay: dict[str, Any] = {}
    if args.context_budget_chars is not None:
        context_overlay["budget_chars"] = {
            args.mode: args.context_budget_chars,
        }
    if args.max_context_file_chars is not None:
        context_overlay["max_file_chars"] = {
            args.mode: args.max_context_file_chars,
        }
    if context_overlay:
        overlay["context"] = context_overlay

    return overlay


def model_configs(raw_config: dict[str, Any]) -> dict[str, ModelConfig]:
    models = {}
    for key, model_data in raw_config["models"].items():
        reasoning_effort = model_data.get("reasoning_effort")
        label = model_data.get("label")
        if not label:
            label = f"{key.title()} {model_data['model']}"
            if reasoning_effort:
                label = f"{label} {reasoning_effort}"
        models[key] = ModelConfig(
            key=key,
            backend=model_data["backend"],
            model=model_data["model"],
            label=label,
            reasoning_effort=reasoning_effort,
        )
    return models


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def build_runtime_config(args: Any, env: dict[str, str] | None = None, cwd: Path | None = None) -> RuntimeConfig:
    env = dict(os.environ if env is None else env)
    cwd = Path.cwd() if cwd is None else cwd

    raw = deep_merge(DEFAULT_CONFIG, load_yaml_config(args.config, cwd))
    raw = deep_merge(raw, env_overlay(env))
    raw = deep_merge(raw, cli_overlay(args))

    judge = str(raw.get("judge", "codex")).strip().lower() or "codex"
    if judge not in SUPPORTED_JUDGES:
        judge = "codex"

    question = " ".join(args.question).strip()
    project_dir = resolve_path(args.workdir, cwd)
    if not project_dir.is_dir():
        raise NotADirectoryError(f"--workdir is not a directory: {project_dir}")

    runs_dir = cwd / "runs"
    context_config = raw["context"]
    context_budget = int(context_config["budget_chars"][args.mode])
    max_file_chars = int(context_config["max_file_chars"][args.mode])

    return RuntimeConfig(
        question=question,
        project_dir=project_dir,
        runs_dir=runs_dir,
        mode=args.mode,
        context_paths=list(args.context),
        follow_links=not args.no_follow_links,
        auto_context=not args.no_auto_context,
        dry_run_context=args.dry_run_context,
        context_budget_chars=context_budget,
        max_context_file_chars=max_file_chars,
        judge=judge,
        models=model_configs(raw),
    )
