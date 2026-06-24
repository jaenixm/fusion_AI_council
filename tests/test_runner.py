from __future__ import annotations

import json
from dataclasses import dataclass

from fusion_ai_council.cli import parse_args
from fusion_ai_council.config import RuntimeConfig, DEFAULT_CONFIG, model_configs
from fusion_ai_council.rerun import rerun_existing_run
from fusion_ai_council.runner import run_council
from fusion_ai_council.types import ModelResult


def runtime(tmp_path, dry_run=False, context_paths=None) -> RuntimeConfig:
    return RuntimeConfig(
        question="Question",
        project_dir=tmp_path,
        runs_dir=tmp_path / "runs",
        mode="general",
        context_paths=context_paths or [],
        follow_links=True,
        auto_context=True,
        dry_run_context=dry_run,
        context_budget_chars=1000,
        max_context_file_chars=500,
        judge="codex",
        models=model_configs(DEFAULT_CONFIG),
    )


@dataclass
class FakeClient:
    key: str
    label: str
    model: str

    def answer(self, prompt, runtime):
        if "Synthesis Judge" in prompt:
            content = "<synthesis_audit>ok</synthesis_audit><final_answer>final text</final_answer>"
            ok = True
        elif self.key == "claude":
            content = "[ERROR] failed"
            ok = False
        else:
            content = f"{self.key} answer"
            ok = True
        return ModelResult(
            key=self.key,
            label=self.label,
            model=self.model,
            file_name="",
            ok=ok,
            content=content,
        )


def test_successful_run_writes_outputs_and_excludes_failed_from_judge(monkeypatch, tmp_path):
    seen_prompts = {}

    def fake_build_model_client(config):
        client = FakeClient(config.key, config.label, config.model)
        original_answer = client.answer

        def answer(prompt, runtime):
            seen_prompts.setdefault(config.key, []).append(prompt)
            return original_answer(prompt, runtime)

        client.answer = answer
        return client

    monkeypatch.setattr("fusion_ai_council.runner.build_model_client", fake_build_model_client)

    result = run_council(runtime(tmp_path))
    run_dir = next((tmp_path / "runs").iterdir())
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert result.final == "final text"
    assert (run_dir / "01_gemini.md").exists()
    assert (run_dir / "02_claude.md").read_text(encoding="utf-8") == "[ERROR] failed"
    assert (run_dir / "03_codex_initial.md").exists()
    assert (run_dir / "04_judge_raw.md").exists()
    assert (run_dir / "04_final.md").read_text(encoding="utf-8") == "final text"
    assert metadata["responses"][1]["ok"] is False
    assert "[ERROR] failed" not in seen_prompts["codex"][-1]


def test_dry_run_context_writes_only_context_and_skips_models(monkeypatch, tmp_path):
    context = tmp_path / "context.md"
    context.write_text("context text", encoding="utf-8")

    def fail_build_model_client(config):
        raise AssertionError("models should not be built during dry-run context")

    monkeypatch.setattr("fusion_ai_council.runner.build_model_client", fail_build_model_client)

    result = run_council(runtime(tmp_path, dry_run=True, context_paths=["context.md"]))
    run_dir = next((tmp_path / "runs").iterdir())

    assert result.context_only is True
    assert (run_dir / "00_context_bundle.md").exists()
    assert not (run_dir / "01_gemini.md").exists()
    assert not (run_dir / "metadata.json").exists()


def test_rerun_existing_run_updates_one_worker_and_rejudges(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "old"
    run_dir.mkdir(parents=True)
    (run_dir / "00_context_bundle.md").write_text("context text", encoding="utf-8")
    (run_dir / "01_gemini.md").write_text("old gemini", encoding="utf-8")
    (run_dir / "02_claude.md").write_text("old claude", encoding="utf-8")
    (run_dir / "03_codex_initial.md").write_text("old codex", encoding="utf-8")
    (run_dir / "04_judge_raw.md").write_text("old judge", encoding="utf-8")
    (run_dir / "04_final.md").write_text("old final", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "question": "Question",
                "mode": "general",
                "project_dir": str(tmp_path),
                "context_bundle_file": "00_context_bundle.md",
                "context_sources": [],
                "context_budget_chars": 1000,
                "max_context_file_chars": 500,
                "follow_links": True,
                "auto_context": True,
                "judge_backend": "codex",
                "judge_model": "gpt-5.5",
                "responses": [
                    {
                        "key": "gemini",
                        "label": "Gemini",
                        "model": "gemini-model",
                        "file_name": "01_gemini.md",
                        "ok": True,
                        "content": "old gemini",
                    },
                    {
                        "key": "claude",
                        "label": "Old Claude",
                        "model": "old-claude-model",
                        "file_name": "02_claude.md",
                        "ok": True,
                        "content": "old claude",
                    },
                    {
                        "key": "codex",
                        "label": "Codex",
                        "model": "codex-model",
                        "file_name": "03_codex_initial.md",
                        "ok": True,
                        "content": "old codex",
                    },
                ],
                "judge_response_order": [],
            }
        ),
        encoding="utf-8",
    )

    seen = []

    @dataclass
    class FakeRerunClient:
        key: str
        label: str
        model: str

        def answer(self, prompt, runtime):
            seen.append((self.key, prompt))
            if self.key == "claude":
                content = "new claude"
            else:
                content = "<synthesis_audit>ok</synthesis_audit><final_answer>new final</final_answer>"
            return ModelResult(
                key=self.key,
                label=self.label,
                model=self.model,
                file_name="",
                ok=True,
                content=content,
            )

    def fake_build_model_client(config):
        return FakeRerunClient(config.key, config.label, config.model)

    monkeypatch.setattr("fusion_ai_council.rerun.build_model_client", fake_build_model_client)

    args = parse_args(
        [
            "--rerun-run",
            str(run_dir),
            "--rerun-model",
            "claude",
            "--rejudge",
        ]
    )
    rerun_existing_run(args, cwd=tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert (run_dir / "01_gemini.md").read_text(encoding="utf-8") == "old gemini"
    assert (run_dir / "02_claude.md").read_text(encoding="utf-8") == "new claude"
    assert (run_dir / "03_codex_initial.md").read_text(encoding="utf-8") == "old codex"
    assert (run_dir / "04_final.md").read_text(encoding="utf-8") == "new final"
    assert list(run_dir.glob("02_claude.md.bak_*"))
    assert list(run_dir.glob("04_final.md.bak_*"))
    assert metadata["responses"][1]["model"] == "opus"
    assert metadata["judge_model"] == "gpt-5.5"
    assert [key for key, _ in seen] == ["claude", "codex"]
