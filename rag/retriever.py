from __future__ import annotations

from contextlib import closing
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

try:
    from rag.embedder import Embedder, get_embedder
except ModuleNotFoundError:
    from embedder import Embedder, get_embedder

KNOWLEDGE_BASE_ERROR = "Knowledge base not initialized. Run ingester.py first."
VECTOR_TABLE_NAME = "vec_chunks"


def _connect_database(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with the sqlite-vec extension loaded."""
    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def _validate_database(connection: sqlite3.Connection) -> None:
    """Ensure the expected vector table exists and contains rows."""
    table_row = connection.execute(
        "select count(*) from sqlite_master where type = 'table' and name = ?",
        (VECTOR_TABLE_NAME,),
    ).fetchone()
    if table_row is None or int(table_row[0]) == 0:
        raise RuntimeError(KNOWLEDGE_BASE_ERROR)

    row_count = connection.execute(f"select count(*) from {VECTOR_TABLE_NAME}").fetchone()
    if row_count is None or int(row_count[0]) == 0:
        raise RuntimeError(KNOWLEDGE_BASE_ERROR)


def _uses_cosine_knn_schema(connection: sqlite3.Connection) -> bool:
    """Return True when the vector table is configured for cosine KNN search."""
    schema_row = connection.execute(
        "select sql from sqlite_master where type = 'table' and name = ?",
        (VECTOR_TABLE_NAME,),
    ).fetchone()
    schema_sql = "" if schema_row is None or schema_row[0] is None else str(schema_row[0]).lower()
    return "distance_metric=cosine" in schema_sql


def _retrieve_with_embedder(
    query: str,
    db_path: str,
    embedder: Embedder,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    """Run a cosine similarity search against the sqlite-vec database."""
    if top_k <= 0:
        return []

    database_path = Path(db_path)
    if not database_path.exists():
        raise RuntimeError(KNOWLEDGE_BASE_ERROR)

    query_vector = np.asarray(embedder.embed_text(query), dtype=np.float32)

    try:
        with closing(_connect_database(database_path)) as connection:
            _validate_database(connection)
            # Prefer sqlite-vec's indexed KNN path when the table was created
            # with cosine distance support.
            if _uses_cosine_knn_schema(connection):
                rows = connection.execute(
                    f"""
                    select
                        chunk_text,
                        source_file,
                        1.0 - distance as similarity_score
                    from {VECTOR_TABLE_NAME}
                    where embedding match ?
                      and k = ?
                    order by distance asc
                    """,
                    (query_vector, top_k),
                ).fetchall()
            else:
                # Older databases may predate the cosine-aware schema. Fall back
                # to computing cosine similarity directly so retrieval still works
                # until the DB is rebuilt.
                rows = connection.execute(
                    f"""
                    select
                        chunk_text,
                        source_file,
                        1.0 - vec_distance_cosine(embedding, ?) as similarity_score
                    from {VECTOR_TABLE_NAME}
                    order by similarity_score desc
                    limit ?
                    """,
                    (query_vector, top_k),
                ).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(KNOWLEDGE_BASE_ERROR) from exc

    return [
        {
            "chunk_text": chunk_text,
            "source_file": source_file,
            "similarity_score": float(similarity_score),
        }
        for chunk_text, source_file, similarity_score in rows
    ]


def retrieve(query: str, db_path: str, top_k: int = 3) -> list[dict[str, str | float]]:
    """Retrieve the most similar chunks for a query using cosine similarity."""
    return _retrieve_with_embedder(query, db_path, get_embedder(), top_k=top_k)


class Retriever:
    """Fetch the top matching chunks for a user query."""

    def __init__(self, database_path: str, embedder: Embedder | None = None) -> None:
        """Store the database location and embedder used for retrieval."""
        self.database_path = database_path
        self.embedder = embedder or get_embedder()

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, str | float]]:
        """Return the best matching chunks for the query."""
        return _retrieve_with_embedder(query, self.database_path, self.embedder, top_k=top_k)
