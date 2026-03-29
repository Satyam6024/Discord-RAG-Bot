from __future__ import annotations

from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Singleton embedding service with lazy model loading and query caching."""

    _instance: Embedder | None = None
    _instance_lock = Lock()

    def __new__(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
        """Return the shared embedder instance and reject unsupported model names."""
        if model_name != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(f"Embedder only supports {DEFAULT_EMBEDDING_MODEL!r}")

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False

        return cls._instance

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Initialize singleton state the first time the shared embedder is created."""
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._model_lock = Lock()
        self._embedding_dimension: int | None = None
        self._query_cache: dict[str, np.ndarray] = {}
        self._initialized = True

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding size for the configured sentence transformer."""
        if self._embedding_dimension is None:
            self._embedding_dimension = int(self._get_model().get_sentence_embedding_dimension())
        return self._embedding_dimension

    def _get_model(self) -> SentenceTransformer:
        """Load the sentence transformer only when embeddings are requested."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
                    self._embedding_dimension = int(self._model.get_sentence_embedding_dimension())

        return self._model

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Return a float32 L2-normalized vector."""
        normalized = np.asarray(vector, dtype=np.float32)
        magnitude = float(np.linalg.norm(normalized))
        if magnitude == 0.0:
            return normalized.copy()
        return normalized / magnitude

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        """Encode a list of texts into normalized NumPy vectors."""
        if not texts:
            return []

        embeddings = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [self._normalize(vector) for vector in np.asarray(embeddings)]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string and reuse cached query vectors."""
        cached_vector = self._query_cache.get(text)
        if cached_vector is not None:
            return cached_vector.copy()

        vector = self._encode([text])[0]
        self._query_cache[text] = vector
        return vector.copy()

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts without mutating the query cache."""
        return self._encode(texts)


def get_embedder() -> Embedder:
    """Return the shared singleton embedder instance."""
    return Embedder()
