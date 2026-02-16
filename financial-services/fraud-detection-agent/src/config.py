"""Configuration helpers for the fraud detection sample."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_haiku_model: str = os.getenv("ANTHROPIC_HAIKU_MODEL", "claude-haiku-4-5-20251001")
    anthropic_sonnet_model: str = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929")
    anthropic_opus_model: str = os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-20250514")
    alert_channel: str = os.getenv("ALERT_CHANNEL", "console")
    enable_deepagents: bool = os.getenv("ENABLE_DEEPAGENTS", "false").lower() == "true"
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "fraud-detection-agent")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


def get_settings() -> Settings:
    return Settings()


def get_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "sample_data"

