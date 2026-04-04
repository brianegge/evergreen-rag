"""pgvector similarity search and embedding storage."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from evergreen_rag.models.search import SearchQuery, SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class VectorSearch:
    """Vector similarity search against ``rag.biblio_embedding`` using pgvector.

    Parameters
    ----------
    db_url:
        PostgreSQL connection string.  Defaults to ``DATABASE_URL`` env var.
    pool_size:
        Maximum number of connections in the pool.
    pool_timeout:
        Seconds to wait for a connection from the pool.
    """

    def __init__(
        self,
        db_url: str | None = None,
        pool_size: int = 10,
        pool_timeout: float = 30.0,
    ) -> None:
        self.db_url = db_url or os.environ.get(
            "DATABASE_URL", "postgresql://evergreen:evergreen@localhost:5432/evergreen"
        )
        self.pool_size = pool_size
        self.pool_timeout = pool_timeout
        self._pool: ConnectionPool | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the connection pool."""
        if self._pool is not None:
            return
        self._pool = ConnectionPool(
            self.db_url,
            min_size=1,
            max_size=self.pool_size,
            timeout=self.pool_timeout,
        )
        logger.info("Connection pool opened (max_size=%d)", self.pool_size)

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            logger.info("Connection pool closed")

    @contextmanager
    def _get_conn(self) -> Generator[psycopg.Connection, None, None]:  # type: ignore[type-arg]
        """Get a connection from the pool (or create a direct one)."""
        if self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
        else:
            with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
                yield conn

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: list[float],
        query: SearchQuery,
    ) -> SearchResponse:
        """Perform cosine similarity search and return ranked results.

        Parameters
        ----------
        query_embedding:
            The embedding vector for the query text.
        query:
            Search parameters (limit, min_similarity, filters).
        """
        sql, params = self._build_search_query(query_embedding, query)

        with self._get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = [
            SearchResult(
                record_id=row["record"],
                similarity=float(row["similarity"]),
                chunk_text=row["chunk_text"],
            )
            for row in rows
        ]

        return SearchResponse(
            query=query.query,
            results=results,
            total=len(results),
            model=query.format or "nomic-embed-text",
        )

    def _build_search_query(
        self,
        query_embedding: list[float],
        query: SearchQuery,
    ) -> tuple[str, dict[str, Any]]:
        """Build the similarity search SQL query."""
        conditions: list[str] = []
        params: dict[str, Any] = {
            "embedding": str(query_embedding),
            "limit": query.limit,
        }

        # Base similarity expression
        similarity_expr = "1 - (embedding <=> %(embedding)s::vector)"

        if query.min_similarity > 0:
            conditions.append(f"{similarity_expr} >= %(min_similarity)s")
            params["min_similarity"] = query.min_similarity

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT record, chunk_text,
                   {similarity_expr} AS similarity
            FROM rag.biblio_embedding
            {where_clause}
            ORDER BY embedding <=> %(embedding)s::vector
            LIMIT %(limit)s
        """

        return sql, params

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------

    def store_embedding(
        self,
        record_id: int,
        embedding: list[float],
        chunk_text: str,
        model_name: str = "nomic-embed-text",
        chunk_index: int = 0,
    ) -> None:
        """Insert or update an embedding in ``rag.biblio_embedding``."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag.biblio_embedding
                        (record, chunk_index, chunk_text, embedding, model_name)
                    VALUES (%(record)s, %(chunk_index)s, %(chunk_text)s,
                            %(embedding)s::vector, %(model_name)s)
                    ON CONFLICT (record, chunk_index, model_name)
                    DO UPDATE SET chunk_text = EXCLUDED.chunk_text,
                                  embedding = EXCLUDED.embedding,
                                  created_at = NOW()
                    """,
                    {
                        "record": record_id,
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text,
                        "embedding": str(embedding),
                        "model_name": model_name,
                    },
                )
            conn.commit()

    def has_embedding(self, record_id: int, model_name: str = "nomic-embed-text") -> bool:
        """Check whether a record already has an embedding stored."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM rag.biblio_embedding
                    WHERE record = %(record)s AND model_name = %(model_name)s
                    LIMIT 1
                    """,
                    {"record": record_id, "model_name": model_name},
                )
                return cur.fetchone() is not None

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about embedded records."""
        with self._get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_embeddings,
                        COUNT(DISTINCT record) AS unique_records,
                        MAX(created_at) AS last_embedded_at,
                        model_name
                    FROM rag.biblio_embedding
                    GROUP BY model_name
                    """
                )
                rows = cur.fetchall()

        if not rows:
            return {
                "total_embeddings": 0,
                "unique_records": 0,
                "last_embedded_at": None,
                "model_name": None,
            }

        row = rows[0]
        return {
            "total_embeddings": row["total_embeddings"],
            "unique_records": row["unique_records"],
            "last_embedded_at": str(row["last_embedded_at"]) if row["last_embedded_at"] else None,
            "model_name": row["model_name"],
        }

    def health_check(self) -> bool:
        """Check database connectivity and pgvector extension."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute(
                        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                    )
                    return cur.fetchone() is not None
        except Exception:
            logger.debug("Database health check failed", exc_info=True)
            return False
