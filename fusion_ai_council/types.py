from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    key: str
    label: str
    model: str
    file_name: str
    ok: bool
    content: str


@dataclass(frozen=True)
class ContextSource:
    path: str
    reason: str
    chars: int
    included_chars: int
    truncated: bool


@dataclass(frozen=True)
class ContextBundle:
    text: str
    sources: list[ContextSource]

