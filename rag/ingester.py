from __future__ import annotations

import argparse
from contextlib import closing
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import sqlite_vec

try:
    from rag.embedder import Embedder, get_embedder
except ModuleNotFoundError:
    from embedder import Embedder, get_embedder

DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50
VECTOR_TABLE_NAME = "vec_chunks"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentChunk:
    """A chunk prepared for embedding and storage."""

    source_name: str
    content: str
    chunk_index: int


def load_documents(folder_path: str | Path) -> list[dict[str, str]]:
    """Read Markdown and text documents from a folder."""
    base_path = Path(folder_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {base_path}")

    documents: list[dict[str, str]] = []
    for path in sorted(base_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue

        documents.append(
            {
                "filename": str(path.relative_to(base_path)),
                "content": path.read_text(encoding="utf-8").strip(),
            }
        )

    return documents


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return []

    words = normalized_text.split(" ")
    if len(words) <= chunk_size:
        return [normalized_text]

    # Step forward by less than the chunk size so neighboring chunks share
    # context and important facts near a boundary are not isolated.
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue

        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks


def _build_chunk_records(documents: list[dict[str, str]]) -> list[DocumentChunk]:
    """Create chunk records from loaded documents."""
    chunk_records: list[DocumentChunk] = []
    for document in documents:
        for chunk_index, chunk in enumerate(chunk_text(document["content"])):
            chunk_records.append(
                DocumentChunk(
                    source_name=document["filename"],
                    content=chunk,
                    chunk_index=chunk_index,
                )
            )

    return chunk_records


def _prepare_chunk_records(folder_path: str | Path) -> list[DocumentChunk]:
    """Load documents from disk and convert them into chunk records."""
    documents = load_documents(folder_path)
    if not documents:
        raise ValueError(f"No .md or .txt files found in {folder_path}")

    chunk_records = _build_chunk_records(documents)
    if not chunk_records:
        raise ValueError("No non-empty chunks were generated from the knowledge base documents")

    return chunk_records


def _connect_database(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with the sqlite-vec extension loaded."""
    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def _initialize_schema(connection: sqlite3.Connection, embedding_dimension: int) -> None:
    """Create the vector table used for chunk storage."""
    connection.execute(
        f"""
        create virtual table if not exists {VECTOR_TABLE_NAME} using vec0(
            id integer primary key,
            embedding float[{embedding_dimension}] distance_metric=cosine,
            source_file text,
            +chunk_text text
        )
        """
    )


def _remove_existing_database(db_path: Path) -> None:
    """Delete an existing SQLite database and sidecar files."""
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal"), db_path.with_name(f"{db_path.name}-shm")):
        if path.exists():
            path.unlink()


def _store_chunks(
    chunk_records: list[DocumentChunk],
    database_path: Path,
    embedder: Embedder,
    force: bool,
) -> int:
    """Embed chunk records and store them in a sqlite-vec database."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        _remove_existing_database(database_path)

    embeddings = embedder.embed_batch([chunk.content for chunk in chunk_records])

    with closing(_connect_database(database_path)) as connection:
        _initialize_schema(connection, embedder.embedding_dimension)
        # sqlite-vec expects float32 arrays for inserts into a vec0 table, so
        # convert each embedding before writing it alongside its source text.
        rows = [
            (
                index,
                np.asarray(embedding, dtype=np.float32),
                chunk.source_name,
                chunk.content,
            )
            for index, (chunk, embedding) in enumerate(zip(chunk_records, embeddings, strict=True), start=1)
        ]
        connection.executemany(
            f"""
            insert into {VECTOR_TABLE_NAME}(id, embedding, source_file, chunk_text)
            values (?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    return len(chunk_records)


def ingest_all(folder_path: str | Path, db_path: str | Path, force: bool = False) -> int:
    """Load, chunk, embed, and store all supported knowledge base documents."""
    database_path = Path(db_path)
    if database_path.exists() and not force:
        return 0

    chunk_records = _prepare_chunk_records(folder_path)
    embedder = get_embedder()
    return _store_chunks(chunk_records, database_path, embedder, force=force)


class Ingester:
    """Load documents, split them into chunks, and store embeddings."""

    def __init__(self, database_path: str, knowledge_base_dir: str, embedder: Embedder | None = None) -> None:
        """Store the ingestion inputs used when the class-based API is called."""
        self.database_path = database_path
        self.knowledge_base_dir = knowledge_base_dir
        self.embedder = embedder

    def ingest(self, force: bool = False) -> int:
        """Ingest knowledge base documents into the vector store."""
        database_path = Path(self.database_path)
        if database_path.exists() and not force:
            return 0

        chunk_records = _prepare_chunk_records(self.knowledge_base_dir)
        embedder = self.embedder or get_embedder()
        return _store_chunks(chunk_records, database_path, embedder, force=force)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone ingestion."""
    parser = argparse.ArgumentParser(description="Build the TechNova sqlite-vec knowledge base.")
    parser.add_argument(
        "--folder",
        default="knowledge_base",
        help="Path to the folder containing .md and .txt knowledge base documents.",
    )
    parser.add_argument(
        "--db",
        default="data/vectors.db",
        help="Path to the sqlite-vec database file to create.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the database even if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    db_path = Path(args.db)

    if db_path.exists() and not args.force:
        logger.info("Database already exists at %s. Pass --force to rebuild it.", db_path)
    else:
        total_chunks = ingest_all(args.folder, db_path, force=args.force)
        logger.info("Ingested %s chunks into %s.", total_chunks, db_path)
