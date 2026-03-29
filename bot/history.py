from __future__ import annotations

from collections import defaultdict


class ConversationHistory:
    """Store the last 3 query and answer pairs for each user in memory."""

    def __init__(self, max_turns: int = 3) -> None:
        """Initialize the in-memory history store with a maximum number of turns per user."""
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")
        self.max_turns = max_turns
        self._history: defaultdict[int, list[tuple[str, str]]] = defaultdict(list)

    def add(self, user_id: int, query: str, answer: str) -> None:
        """Append a new conversation turn and keep only the latest entries."""
        user_history = self._history[user_id]
        user_history.append((query, answer))
        if len(user_history) > self.max_turns:
            del user_history[:-self.max_turns]

    def get(self, user_id: int) -> list[tuple[str, str]]:
        """Return the stored conversation turns for a user."""
        return list(self._history.get(user_id, []))

    def clear(self, user_id: int) -> None:
        """Clear all stored conversation history for a user."""
        self._history.pop(user_id, None)


HistoryStore = ConversationHistory
