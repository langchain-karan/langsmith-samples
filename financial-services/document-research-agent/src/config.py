"""Configuration helpers for the sample project."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_haiku_model: str = os.getenv(
        "ANTHROPIC_HAIKU_MODEL",
        "claude-haiku-4-5-20251001",
    )
    anthropic_sonnet_model: str = os.getenv(
        "ANTHROPIC_SONNET_MODEL",
        "claude-sonnet-4-5-20250929",
    )
    anthropic_opus_model: str = os.getenv(
        "ANTHROPIC_OPUS_MODEL",
        "claude-opus-4-20250514",
    )
    enable_deepagents: bool = os.getenv("ENABLE_DEEPAGENTS", "false").lower() == "true"
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "document-research-agent")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


def get_settings() -> Settings:
    """Return the singleton-style settings object."""
    return Settings()

