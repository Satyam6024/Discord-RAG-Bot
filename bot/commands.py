from __future__ import annotations

import asyncio
import os

import discord
from discord import app_commands

from bot.history import ConversationHistory
from rag import llm, retriever

DEFAULT_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vectors.db")
DEFAULT_TOP_K = 3
MAX_DISCORD_MESSAGE_LENGTH = 2000

conversation_history = ConversationHistory()


def _build_answer_message(answer: str, sources: list[str]) -> str:
    """Format the bot answer for Discord and keep it within message limits."""
    source_text = ", ".join(sources) if sources else "None"
    prefix = "**Answer:** "
    suffix = f"\n📄 **Sources:** {source_text}"
    max_answer_length = MAX_DISCORD_MESSAGE_LENGTH - len(prefix) - len(suffix)

    safe_answer = answer.strip()
    if len(safe_answer) > max_answer_length:
        safe_answer = f"{safe_answer[: max_answer_length - 3].rstrip()}..."

    return f"{prefix}{safe_answer}{suffix}"


async def _send_ephemeral_error(interaction: discord.Interaction, message: str) -> None:
    """Send an error message visible only to the command invoker."""
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


def _friendly_error_message(error: Exception) -> str:
    """Convert internal failures into user-facing error text."""
    error_text = str(error).strip()
    if error_text == retriever.KNOWLEDGE_BASE_ERROR:
        return "The knowledge base is not ready yet. Please run `ingester.py` first."
    if error_text == llm.OLLAMA_NOT_RUNNING_ERROR:
        return "Ollama is not running. Start it with: `ollama serve`."
    if "ollama Python package is not installed" in error_text:
        return "The Ollama Python package is missing. Install it with `pip install ollama`."
    return "Sorry, I ran into a problem while processing that request. Please try again in a moment."


async def ask_command(interaction: discord.Interaction, query: str) -> None:
    """Handle the /ask command."""
    cleaned_query = query.strip()
    if not cleaned_query:
        await _send_ephemeral_error(interaction, "Please provide a question for `/ask`.")
        return

    await interaction.response.defer(thinking=True)

    try:
        user_history = conversation_history.get(interaction.user.id)
        chunks = await asyncio.to_thread(
            retriever.retrieve,
            cleaned_query,
            DEFAULT_DB_PATH,
            DEFAULT_TOP_K,
        )
        result = await asyncio.to_thread(
            llm.generate_answer,
            cleaned_query,
            chunks,
            user_history,
        )
    except Exception as exc:
        await _send_ephemeral_error(interaction, _friendly_error_message(exc))
        return

    answer_text = str(result.get("answer", "")).strip()
    sources = [str(source) for source in result.get("sources", [])]
    conversation_history.add(interaction.user.id, cleaned_query, answer_text)

    await interaction.followup.send(_build_answer_message(answer_text, sources))


async def help_command(interaction: discord.Interaction) -> None:
    """Handle the /help command."""
    try:
        embed = discord.Embed(
            title="TechNova RAG Bot Commands",
            description="Ask questions about the TechNova knowledge base and manage your chat history.",
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name="/ask <query>",
            value="Retrieve relevant knowledge base chunks and generate an answer.",
            inline=False,
        )
        embed.add_field(
            name="/help",
            value="Show this command reference.",
            inline=False,
        )
        embed.add_field(
            name="/clear",
            value="Clear your saved conversation history for future `/ask` prompts.",
            inline=False,
        )
        await interaction.response.send_message(embed=embed)
    except Exception as exc:
        await _send_ephemeral_error(interaction, _friendly_error_message(exc))


async def clear_command(interaction: discord.Interaction) -> None:
    """Handle the /clear command."""
    try:
        conversation_history.clear(interaction.user.id)
        await interaction.response.send_message("Your conversation history has been cleared.", ephemeral=True)
    except Exception as exc:
        await _send_ephemeral_error(interaction, _friendly_error_message(exc))


@app_commands.command(name="ask", description="Ask the TechNova knowledge base a question.")
@app_commands.describe(query="The question you want answered")
async def ask_slash_command(interaction: discord.Interaction, query: str) -> None:
    """Registered /ask slash command."""
    await ask_command(interaction, query)


@app_commands.command(name="help", description="Show the available bot commands.")
async def help_slash_command(interaction: discord.Interaction) -> None:
    """Registered /help slash command."""
    await help_command(interaction)


@app_commands.command(name="clear", description="Clear your saved conversation history.")
async def clear_slash_command(interaction: discord.Interaction) -> None:
    """Registered /clear slash command."""
    await clear_command(interaction)


async def register_commands(tree: app_commands.CommandTree) -> None:
    """Register slash commands with the bot."""
    for command in (ask_slash_command, help_slash_command, clear_slash_command):
        if tree.get_command(command.name) is None:
            tree.add_command(command)
