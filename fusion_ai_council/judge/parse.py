from __future__ import annotations

import re


FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(raw_judge_output: str) -> str:
    match = FINAL_ANSWER_RE.search(raw_judge_output)
    if match:
        return match.group(1).strip()
    return raw_judge_output.strip()

