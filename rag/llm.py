from __future__ import annotations

from collections.abc import Sequence

from config import OLLAMA_HOST, OLLAMA_MODEL

OLLAMA_NOT_RUNNING_ERROR = "Ollama is not running. Start it with: ollama serve"


def _format_chunks(chunks: Sequence[dict[str, str]]) -> str:
    """Render retrieved chunks with source labels."""
    if not chunks:
        return "No retrieved knowledge base chunks were provided."

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown")
        chunk_text = chunk.get("chunk_text", "").strip()
        lines.append(f"[Source {index}: {source_file}]")
        lines.append(chunk_text or "(empty chunk)")
        lines.append("")

    return "\n".join(lines).strip()


def _format_history(history: Sequence[tuple[str, str]]) -> str:
    """Render up to the last 3 user and bot turns."""
    if not history:
        return "No prior conversation history."

    lines: list[str] = []
    for turn_index, (user_message, bot_message) in enumerate(history[-3:], start=1):
        lines.append(f"Turn {turn_index} User: {user_message}")
        lines.append(f"Turn {turn_index} Bot: {bot_message}")

    return "\n".join(lines)


def build_prompt(
    query: str,
    chunks: Sequence[dict[str, str]],
    history: Sequence[tuple[str, str]],
) -> str:
    """Build the prompt for the local Ollama model."""
    system_context = (
        "You are TechNova's support assistant. "
        "Answer using the retrieved knowledge base context when it is relevant. "
        "If the context does not contain the answer, say that the information is not available in the knowledge base. "
        "Be concise, accurate, and do not invent policies, pricing, or product details."
    )

    sections = [
        "System Context:",
        system_context,
        "",
        "Retrieved Chunks:",
        _format_chunks(chunks),
        "",
        "Conversation History:",
        _format_history(history),
        "",
        "Current User Query:",
        query.strip(),
        "",
        "Instructions:",
        "Answer the current query in a helpful way. When relevant, rely on the retrieved chunks and mention the source names naturally if useful.",
    ]
    return "\n".join(sections).strip()


def _unique_sources(chunks: Sequence[dict[str, str]]) -> list[str]:
    """Return source filenames in first-seen order without duplicates."""
    sources: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        source_file = chunk.get("source_file")
        if not source_file or source_file in seen:
            continue
        seen.add(source_file)
        sources.append(source_file)
    return sources


def _import_ollama():
    """Import the Ollama package lazily."""
    try:
        import ollama
    except ModuleNotFoundError as exc:
        raise RuntimeError("The ollama Python package is not installed. Install it with: pip install ollama") from exc
    return ollama


def _is_connection_error(exc: Exception) -> bool:
    """Return True when the exception indicates Ollama is unavailable."""
    error_text = str(exc).lower()
    if any(
        marker in error_text
        for marker in (
            "connection refused",
            "connect error",
            "connection error",
            "failed to connect",
            "all connection attempts failed",
            "actively refused",
            "nodename nor servname provided",
            "name or service not known",
        )
    ):
        return True

    exception_name = exc.__class__.__name__.lower()
    exception_module = exc.__class__.__module__.lower()
    return "connect" in exception_name or exception_module.startswith("httpx")


def _chat_with_ollama(prompt: str, model: str, host: str) -> str:
    """Send a prompt to Ollama and return the generated text."""
    ollama = _import_ollama()

    try:
        client = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
    except Exception as exc:
        if _is_connection_error(exc):
            raise RuntimeError(OLLAMA_NOT_RUNNING_ERROR) from exc
        raise

    return response["message"]["content"].strip()


def generate_answer(
    query: str,
    chunks: Sequence[dict[str, str]],
    history: Sequence[tuple[str, str]],
) -> dict[str, str | list[str]]:
    """Generate an answer and return it with the retrieved source list."""
    prompt = build_prompt(query, chunks, history)
    answer = _chat_with_ollama(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST)
    return {
        "answer": answer,
        "sources": _unique_sources(chunks),
    }


class LLMClient:
    """Build prompts and request answers from a local Ollama model."""

    def __init__(self, model_name: str = OLLAMA_MODEL, base_url: str = OLLAMA_HOST) -> None:
        """Store the model name and Ollama host used for chat requests."""
        self.model_name = model_name
        self.base_url = base_url

    def build_prompt(
        self,
        query: str,
        context_chunks: Sequence[dict[str, str]],
        conversation_history: Sequence[tuple[str, str]],
    ) -> str:
        """Build the prompt sent to the language model."""
        return build_prompt(query, context_chunks, conversation_history)

    def generate_answer(
        self,
        query: str,
        chunks: Sequence[dict[str, str]],
        history: Sequence[tuple[str, str]],
    ) -> dict[str, str | list[str]]:
        """Generate an answer from the local language model."""
        prompt = build_prompt(query, chunks, history)
        answer = _chat_with_ollama(prompt, model=self.model_name, host=self.base_url)
        return {
            "answer": answer,
            "sources": _unique_sources(chunks),
        }
