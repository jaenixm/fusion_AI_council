from __future__ import annotations

from fusion_ai_council.config import ModelConfig
from fusion_ai_council.models.agy import AgyModelClient
from fusion_ai_council.models.base import ModelClient
from fusion_ai_council.models.claude import ClaudeModelClient
from fusion_ai_council.models.codex import CodexModelClient


def build_model_client(config: ModelConfig) -> ModelClient:
    if config.backend == "agy":
        return AgyModelClient.from_config(config)
    if config.backend == "claude":
        return ClaudeModelClient.from_config(config)
    if config.backend == "codex":
        return CodexModelClient.from_config(config)
    raise ValueError(f"Unsupported model backend for {config.key}: {config.backend}")
