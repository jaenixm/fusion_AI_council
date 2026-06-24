from __future__ import annotations

from fusion_ai_council.context.bundle import build_context_bundle
from fusion_ai_council.context.modes import compose_model_prompt
from fusion_ai_council.judge.parse import extract_final_answer
from fusion_ai_council.judge.prompt import build_judge_prompt
from fusion_ai_council.types import ContextBundle, ModelResult


def test_context_bundle_explicit_first_links_and_truncation(tmp_path):
    project = tmp_path
    linked = project / "linked.md"
    linked.write_text("linked content", encoding="utf-8")
    explicit = project / "main.md"
    explicit.write_text("[Linked](linked.md)\n" + ("x" * 20), encoding="utf-8")
    skipped_dir = project / ".git"
    skipped_dir.mkdir()
    (skipped_dir / "skip.md").write_text("skip", encoding="utf-8")

    bundle = build_context_bundle(
        ["main.md", ".git"],
        project,
        "general",
        follow_links=True,
        auto_context=True,
        budget_chars=30,
        max_file_chars=20,
    )

    assert bundle.sources[0].path == "main.md"
    assert bundle.sources[0].reason == "user"
    assert bundle.sources[0].included_chars == 20
    assert bundle.sources[0].truncated is True
    assert bundle.sources[1].path == "linked.md"
    assert "skip.md" not in bundle.text


def test_scientific_prompt_and_judge_rules_hide_model_identity():
    context = ContextBundle(text="CONTEXT", sources=[])
    worker_prompt = compose_model_prompt("Improve this", "scientific-writing", context)
    assert "Do not invent citations" in worker_prompt
    assert "reference material" in worker_prompt

    response = ModelResult(
        key="gemini",
        label="Gemini",
        model="secret-model",
        file_name="01_gemini.md",
        ok=True,
        content="candidate answer",
    )
    prompt = build_judge_prompt("Improve this", [response], "scientific-writing", context)
    assert "Synthesis Judge" in prompt
    assert "Scientific-writing rules" in prompt
    assert "secret-model" not in prompt
    assert "Failed model calls" not in prompt


def test_final_answer_parser():
    raw = "<synthesis_audit>x</synthesis_audit>\n<final_answer> final text </final_answer>"
    assert extract_final_answer(raw) == "final text"
    assert extract_final_answer("plain") == "plain"

