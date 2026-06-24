from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote, urlparse

from fusion_ai_council.context.modes import delimited_block, make_delimiter
from fusion_ai_council.types import ContextBundle, ContextSource


TEXT_CONTEXT_EXTENSIONS = {".md", ".txt", ".bib", ".tex", ".rst", ".json", ".yaml", ".yml"}
EXCLUDED_CONTEXT_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "node_modules",
    "vendors",
}


def is_context_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_CONTEXT_EXTENSIONS


def relative_source_path(path: Path, project_dir: Path) -> str:
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def resolve_path(path_text: str, project_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def default_scientific_contexts(project_dir: Path) -> list[Path]:
    candidates = [
        project_dir / "context/llm/llm_thesis_context_compressed.md",
        project_dir / "context/literature/_index.md",
        project_dir / "context/progress/thesis_progress.md",
    ]
    return [path for path in candidates if path.exists()]


def should_skip_context_path(path: Path) -> bool:
    return any(part in EXCLUDED_CONTEXT_DIRS for part in path.parts)


def collect_directory_contexts(directory: Path) -> list[Path]:
    paths = []
    for path in sorted(directory.rglob("*")):
        if should_skip_context_path(path):
            continue
        if is_context_file(path):
            paths.append(path)
    return paths


def markdown_link_targets(text: str, source_path: Path, project_dir: Path) -> list[Path]:
    targets = []
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        raw_target = match.group(1).strip()
        if not raw_target or raw_target.startswith(("http://", "https://", "#")):
            continue

        raw_target = raw_target.split("#", 1)[0]
        if raw_target.startswith("file://"):
            parsed = urlparse(raw_target)
            target = Path(unquote(parsed.path))
        else:
            target = Path(unquote(raw_target))
            if not target.is_absolute():
                target = source_path.parent / target

        try:
            resolved = target.resolve()
        except OSError:
            continue

        if resolved.exists() and is_context_file(resolved):
            targets.append(resolved)

    return targets


def add_context_candidate(
    candidates: OrderedDict[str, dict],
    path: Path,
    reason: str,
    priority: int,
) -> None:
    if should_skip_context_path(path) or not path.exists():
        return

    if path.is_dir():
        for child in collect_directory_contexts(path):
            add_context_candidate(candidates, child, f"{reason}:dir", priority + 1)
        return

    if not is_context_file(path):
        return

    key = str(path)
    existing = candidates.get(key)
    if existing is None or priority < existing["priority"]:
        candidates[key] = {"path": path, "reason": reason, "priority": priority}


def collect_context_candidates(
    context_paths: list[str],
    project_dir: Path,
    mode: str,
    follow_links: bool,
    auto_context: bool,
) -> list[dict]:
    candidates: OrderedDict[str, dict] = OrderedDict()

    for context_path in context_paths:
        add_context_candidate(
            candidates,
            resolve_path(context_path, project_dir),
            "user",
            0,
        )

    if follow_links:
        direct_items = list(candidates.values())
        for item in direct_items:
            path = item["path"]
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for linked_path in markdown_link_targets(text, path, project_dir):
                add_context_candidate(candidates, linked_path, f"linked-from:{path.name}", 1)

    if auto_context and mode == "scientific-writing":
        for path in default_scientific_contexts(project_dir):
            add_context_candidate(candidates, path.resolve(), "auto-scientific", 3)

    return sorted(
        candidates.values(),
        key=lambda item: (item["priority"], relative_source_path(item["path"], project_dir)),
    )


def build_context_bundle(
    context_paths: list[str],
    project_dir: Path,
    mode: str,
    follow_links: bool,
    auto_context: bool,
    budget_chars: int,
    max_file_chars: int,
) -> ContextBundle:
    candidates = collect_context_candidates(
        context_paths,
        project_dir,
        mode,
        follow_links=follow_links,
        auto_context=auto_context,
    )

    if not candidates:
        return ContextBundle(text="", sources=[])

    delimiter = make_delimiter()
    sections = []
    sources = []
    remaining = max(0, budget_chars)

    for item in candidates:
        path = item["path"]
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue

        rel_path = relative_source_path(path, project_dir)
        file_budget = min(max_file_chars, remaining)
        if file_budget <= 0:
            sources.append(
                ContextSource(
                    path=rel_path,
                    reason=item["reason"],
                    chars=len(text),
                    included_chars=0,
                    truncated=True,
                )
            )
            continue

        included = text[:file_budget]
        truncated = len(included) < len(text)
        sections.append(
            "\n".join(
                [
                    f"CONTEXT SOURCE: {rel_path}",
                    f"CONTEXT REASON: {item['reason']}",
                    delimited_block("CONTEXT", included, delimiter),
                ]
            )
        )
        sources.append(
            ContextSource(
                path=rel_path,
                reason=item["reason"],
                chars=len(text),
                included_chars=len(included),
                truncated=truncated,
            )
        )
        remaining -= len(included)

    return ContextBundle(text="\n\n".join(sections), sources=sources)

