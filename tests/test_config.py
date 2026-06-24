from __future__ import annotations

from fusion_ai_council.cli import parse_args
from fusion_ai_council.config import build_runtime_config


def test_config_precedence_cli_over_env_over_yaml(tmp_path):
    config_path = tmp_path / "fusion-council.yaml"
    config_path.write_text(
        """
judge: claude
models:
  codex:
    model: yaml-codex
    reasoning_effort: medium
context:
  budget_chars:
    scientific-writing: 222
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--mode",
            "scientific-writing",
            "--judge",
            "gemini",
            "--codex-model",
            "cli-codex",
            "--claude-model",
            "cli-claude",
            "--claude-effort",
            "max",
            "--context-budget-chars",
            "444",
            "Question",
        ]
    )
    runtime = build_runtime_config(
        args,
        env={
            "FUSION_JUDGE": "codex",
            "FUSION_CODEX_MODEL": "env-codex",
            "FUSION_CODEX_REASONING_EFFORT": "high",
            "FUSION_CLAUDE_MODEL": "env-claude",
            "FUSION_CLAUDE_EFFORT": "high",
        },
        cwd=tmp_path,
    )

    assert runtime.judge == "gemini"
    assert runtime.models["codex"].model == "cli-codex"
    assert runtime.models["codex"].reasoning_effort == "high"
    assert runtime.models["claude"].model == "cli-claude"
    assert runtime.models["claude"].reasoning_effort == "max"
    assert runtime.context_budget_chars == 444


def test_yaml_used_when_no_env_or_cli_override(tmp_path):
    config_path = tmp_path / "fusion-council.yaml"
    config_path.write_text(
        """
judge: claude
models:
  codex:
    model: yaml-codex
    reasoning_effort: medium
context:
  max_file_chars:
    coding: 1234
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--mode",
            "coding",
            "Question",
        ]
    )
    runtime = build_runtime_config(args, env={}, cwd=tmp_path)

    assert runtime.judge == "claude"
    assert runtime.models["codex"].model == "yaml-codex"
    assert runtime.models["codex"].reasoning_effort == "medium"
    assert runtime.max_context_file_chars == 1234


def test_default_claude_uses_claude_cli_backend(tmp_path):
    args = parse_args(["--workdir", str(tmp_path), "Question"])
    runtime = build_runtime_config(args, env={}, cwd=tmp_path)

    assert runtime.models["claude"].backend == "claude"
    assert runtime.models["claude"].model == "opus"
    assert runtime.models["claude"].reasoning_effort == "xhigh"
