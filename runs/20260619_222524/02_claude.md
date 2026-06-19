Let me also look at some run outputs to understand what the system is actually producing:
Now I have a thorough understanding of the project. Let me create a detailed analysis artifact.
Here's the executive summary — the full analysis is in the artifact above.

### Your code is in genuinely good shape as a prototype

You've already fixed the major architectural issues (parallelisation, error separation, explicit model params, relative `WORKDIR`). The remaining leverage is almost entirely **in the judge prompt itself**.

### The 3 most critical judge prompt weaknesses:

1. **`judge_label` leaks model identity** (line 163) — Saying `"You are Codex (gpt-5.5-codex) with xhigh reasoning"` anchors the judge in its own identity and signals it might be evaluating its own output. Replace with a neutral `"Synthesis Judge"`.

2. **No task-type awareness** — The prompt treats a coding question, a factual lookup, and a creative request identically. Fusion strategy should differ radically: consensus for facts, correctness-tracing for code, "pick the best" for creative.

3. **Failed responses still leak information** — Even marked as "ignore these", the failed section tells the judge the council composition and which models broke. If 2/3 fail, the judge knows it's rubber-stamping a single answer.

### And the elephant in the room:

**Codex is still both council member and default judge.** This is the single highest-impact bias vector. The fix is straightforward — default to a model that isn't in the council, or at minimum rotate.

The artifact includes a **fully rewritten judge prompt** with task-type classification, structured scratch-pad for checklist enforcement, confidence calibration rules, and all identity/leakage issues resolved. It's a drop-in replacement for `build_judge_prompt()`. Check it out and let me know if you want me to apply it.