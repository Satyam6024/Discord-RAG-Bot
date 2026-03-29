from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands

from bot import commands as bot_commands
from config import load_settings
from rag.ingester import ingest_all

KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "data/vectors.db"))
SUPPORTED_KNOWLEDGE_EXTENSIONS = {".md", ".txt"}

logger = logging.getLogger(__name__)


def _knowledge_base_has_documents(folder_path: Path) -> bool:
    """Return True when the knowledge base contains at least one supported document."""
    if not folder_path.exists():
        return False

    return any(
        path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS
        for path in folder_path.rglob("*")
    )


class TechNovaBot(commands.Bot):
    """Discord bot that wires command registration, ingestion, and slash command sync."""

    def __init__(self) -> None:
        """Create the Discord bot instance with the intents required by this project."""
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self._startup_complete = False

    async def setup_hook(self) -> None:
        """Register app commands before the bot connects."""
        await bot_commands.register_commands(self.tree)

    async def on_ready(self) -> None:
        """Print startup information, ensure the vector DB exists, and sync commands."""
        bot_name = self.user.name if self.user is not None else "Unknown Bot"
        logger.info("Logged in as %s | Guilds: %s", bot_name, len(self.guilds))

        if self._startup_complete:
            return

        try:
            if not VECTOR_DB_PATH.exists():
                chunk_count = await asyncio.to_thread(ingest_all, KNOWLEDGE_BASE_DIR, VECTOR_DB_PATH)
                logger.info("Built knowledge base with %s chunks at %s.", chunk_count, VECTOR_DB_PATH)

            if not self.guilds:
                logger.info("No guilds available yet, so slash command sync was skipped.")
                self._startup_complete = True
                return

            for guild in self.guilds:
                self.tree.copy_global_to(guild=guild)
                synced_commands = await self.tree.sync(guild=guild)
                logger.info(
                    "Synced %s app commands to guild '%s' (%s).",
                    len(synced_commands),
                    guild.name,
                    guild.id,
                )
        except Exception as exc:
            logger.exception("Startup failed: %s", exc)
            await self.close()
            return

        self._startup_complete = True


def main() -> None:
    """Bootstrap the application and start the bot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = load_settings()

    if not _knowledge_base_has_documents(KNOWLEDGE_BASE_DIR):
        logger.warning("knowledge_base/ is empty. Add at least one .md or .txt file before starting the bot.")
        return

    bot = TechNovaBot()
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
