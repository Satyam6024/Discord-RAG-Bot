from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://ollama:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings required at startup."""

    discord_token: str


def load_settings() -> Settings:
    """Load the Discord token from the environment."""
    discord_token = os.getenv("DISCORD_TOKEN", "").strip()
    if not discord_token:
        raise RuntimeError("DISCORD_TOKEN is missing. Add it to your .env file before starting the bot.")

    return Settings(discord_token=discord_token)
