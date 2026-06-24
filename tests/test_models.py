from __future__ import annotations

from pathlib import Path

from fusion_ai_council.config import RuntimeConfig, DEFAULT_CONFIG, model_configs
from fusion_ai_council.models import agy as agy_module
from fusion_ai_council.models.agy import AgyModelClient
from fusion_ai_council.models.base import is_error_response
from fusion_ai_council.models.claude import ClaudeModelClient
from fusion_ai_council.models.codex import CodexModelClient


def runtime(tmp_path: Path) -> RuntimeConfig:
    models = model_configs(DEFAULT_CONFIG)
    return RuntimeConfig(
        question="Question",
        project_dir=tmp_path,
        runs_dir=tmp_path / "runs",
        mode="general",
        context_paths=[],
        follow_links=True,
        auto_context=True,
        dry_run_context=False,
        context_budget_chars=150,
        max_context_file_chars=50,
        judge="codex",
        models=models,
    )


def test_agy_command_shape():
    client = AgyModelClient(key="gemini", label="Gemini", model="Gemini Model")
    assert client.command("prompt", print_timeout_seconds=1800) == [
        "agy",
        "--model",
        "Gemini Model",
        "--print-timeout",
        "1800s",
        "-p",
        "prompt",
    ]


def test_agy_inlines_project_artifact_link(monkeypatch, tmp_path):
    artifact = tmp_path / "review.md"
    artifact.write_text("# Review\n\nSubstantive artifact content.", encoding="utf-8")

    def fake_run_command(cmd, cwd, timeout, input_text=None):
        return f"Short summary. Full review: file://{artifact}"

    monkeypatch.setattr(agy_module, "run_command", fake_run_command)

    client = AgyModelClient(key="gemini", label="Gemini", model="Gemini Model")
    result = client.answer("prompt", runtime(tmp_path))

    assert result.ok is True
    assert "Referenced local artifact content included for synthesis" in result.content
    assert "Substantive artifact content." in result.content


def test_agy_retries_timeout_transcript_once(monkeypatch, tmp_path):
    calls = []

    def fake_run_command(cmd, cwd, timeout, input_text=None):
        calls.append((cmd, timeout))
        if len(calls) == 1:
            return "I gathered notes.Error: timed out waiting for response"
        return "Complete final answer."

    monkeypatch.setattr(agy_module, "run_command", fake_run_command)

    client = AgyModelClient(key="claude", label="Claude", model="Claude Model")
    result = client.answer("original prompt", runtime(tmp_path))

    assert result.ok is True
    assert result.content == "Complete final answer."
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("--print-timeout") + 1] == "1800s"
    assert calls[0][1] == 1860
    assert "Partial timed-out transcript" in calls[1][0][-1]


def test_error_detection_marks_agy_timeout_transcript_failed():
    assert is_error_response("I started work.\nError: timed out waiting for response")
    assert is_error_response("I started work.Error: timed out waiting for response")


def test_claude_command_uses_print_mode_and_model_alias():
    client = ClaudeModelClient(
        key="claude",
        label="Claude",
        model="opus",
        reasoning_effort="xhigh",
    )
    assert client.command() == [
        "claude",
        "--print",
        "--model",
        "opus",
        "--effort",
        "xhigh",
        "--input-format",
        "text",
    ]


def test_codex_command_uses_runtime_workdir_and_last_message(tmp_path):
    rt = runtime(tmp_path)
    client = CodexModelClient(
        key="codex",
        label="Codex",
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )
    cmd = client.command(rt, output_path=tmp_path / "last.md")

    assert cmd[cmd.index("--cd") + 1] == str(tmp_path)
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    assert cmd[cmd.index("--config") + 1] == 'model_reasoning_effort="xhigh"'
    assert cmd[cmd.index("--output-last-message") + 1] == str(tmp_path / "last.md")
    assert cmd[-1] == "-"
