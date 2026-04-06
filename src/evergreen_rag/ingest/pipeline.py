"""Batch ingest pipeline: read MARC from DB, extract, embed, store."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import psycopg

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.extractor.marc_extractor import detect_language, extract_record
from evergreen_rag.models.embedding import EmbeddingResponse

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Statistics for an ingest run."""

    total: int = 0
    extracted: int = 0
    embedded: int = 0
    failed: int = 0
    skipped: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class IngestPipeline:
    """Orchestrates batch ingest: read MARC records -> extract -> embed -> store.

    Parameters
    ----------
    db_url:
        PostgreSQL connection string.  Defaults to ``DATABASE_URL`` env var.
    embedding_service:
        An ``EmbeddingService`` instance for generating embeddings.
    batch_size:
        Number of records to process per batch.
    """

    def __init__(
        self,
        db_url: str | None = None,
        embedding_service: EmbeddingService | None = None,
        batch_size: int = 50,
    ) -> None:
        self.db_url = db_url or os.environ.get(
            "DATABASE_URL", "postgresql://evergreen:evergreen@localhost:5432/evergreen"
        )
        self.embedding_service = embedding_service or EmbeddingService()
        self.batch_size = batch_size

    def run(
        self,
        record_ids: list[int] | None = None,
        full: bool = False,
    ) -> IngestStats:
        """Run the ingest pipeline.

        Parameters
        ----------
        record_ids:
            Specific record IDs to ingest.  If *None* and *full* is *False*,
            only records not yet embedded (incremental) are processed.
        full:
            If *True*, re-ingest all records regardless of prior state.
        """
        stats = IngestStats()

        with psycopg.connect(self.db_url) as conn:
            rows = self._fetch_records(conn, record_ids, full)
        stats.total = len(rows)
        logger.info("Fetched %d records to process", stats.total)

        for batch_start in range(0, len(rows), self.batch_size):
            batch = rows[batch_start : batch_start + self.batch_size]
            with psycopg.connect(self.db_url) as conn:
                self._process_batch(conn, batch, stats)

        stats.finished_at = datetime.now(timezone.utc)
        logger.info(
            "Ingest complete: %d extracted, %d embedded, %d failed, %d skipped",
            stats.extracted,
            stats.embedded,
            stats.failed,
            stats.skipped,
        )
        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_records(
        self,
        conn: psycopg.Connection,  # type: ignore[type-arg]
        record_ids: list[int] | None,
        full: bool,
    ) -> list[tuple[int, str]]:
        """Fetch MARC XML from ``biblio.record_entry``."""
        with conn.cursor() as cur:
            if record_ids:
                cur.execute(
                    "SELECT id, marc FROM biblio.record_entry WHERE id = ANY(%s)",
                    (record_ids,),
                )
            elif full:
                cur.execute(
                    "SELECT id, marc FROM biblio.record_entry WHERE marc IS NOT NULL"
                )
            else:
                # Incremental: only records not yet embedded
                cur.execute(
                    """
                    SELECT bre.id, bre.marc
                    FROM biblio.record_entry bre
                    LEFT JOIN rag.biblio_embedding rbe ON bre.id = rbe.record
                    WHERE bre.marc IS NOT NULL AND rbe.record IS NULL
                    """
                )
            return list(cur.fetchall())

    def _process_batch(
        self,
        conn: psycopg.Connection,  # type: ignore[type-arg]
        batch: list[tuple[int, str]],
        stats: IngestStats,
    ) -> None:
        """Extract, embed, and store a batch of records."""
        texts: list[str] = []
        extracted_ids: list[int] = []
        languages: list[str] = []

        for record_id, marc_xml in batch:
            record = extract_record(marc_xml, record_id=record_id)
            if record is None:
                stats.failed += 1
                continue
            stats.extracted += 1
            text = record.to_embedding_text()
            if not text.strip():
                stats.skipped += 1
                continue
            texts.append(text)
            extracted_ids.append(record_id)
            languages.append(detect_language(marc_xml))

        if not texts:
            return

        try:
            response = self._embed_with_languages(texts, languages)
        except Exception:
            logger.exception("Embedding failed for batch of %d", len(texts))
            stats.failed += len(texts)
            return

        with conn.cursor() as cur:
            for record_id, embedding, text in zip(
                extracted_ids, response.embeddings, texts
            ):
                try:
                    self._store_embedding(
                        cur, record_id, embedding, text, response.model
                    )
                    stats.embedded += 1
                except Exception:
                    logger.exception("Failed to store embedding for record %d", record_id)
                    stats.failed += 1
            conn.commit()

        self._log_ingest(conn, extracted_ids)

    def _embed_with_languages(
        self, texts: list[str], languages: list[str]
    ) -> EmbeddingResponse:
        """Embed texts, using per-language models when a model_map is configured."""
        if not self.embedding_service.model_map:
            return self.embedding_service.embed_batch(texts)

        # Group by resolved model to batch efficiently
        from collections import defaultdict

        model_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for idx, (text, lang) in enumerate(zip(texts, languages)):
            model = self.embedding_service._resolve_model(lang)
            model_groups[model].append((idx, text))

        # Embed each group and reassemble in original order
        all_embeddings: list[list[float]] = [[] for _ in texts]
        last_response_model = self.embedding_service.model
        last_dimensions = 0

        for model, items in model_groups.items():
            group_texts = [t for _, t in items]
            resp = self.embedding_service.embed_batch_with_language(
                group_texts, languages[items[0][0]]
            )
            last_response_model = resp.model
            last_dimensions = resp.dimensions
            for (orig_idx, _), emb in zip(items, resp.embeddings):
                all_embeddings[orig_idx] = emb

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=last_response_model,
            dimensions=last_dimensions,
        )

    def _store_embedding(
        self,
        cur: psycopg.Cursor,  # type: ignore[type-arg]
        record_id: int,
        embedding: list[float],
        chunk_text: str,
        model_name: str,
    ) -> None:
        """Insert or update an embedding in ``rag.biblio_embedding``."""
        cur.execute(
            """
            INSERT INTO rag.biblio_embedding
                (record, chunk_index, chunk_text, embedding, model_name)
            VALUES (%s, 0, %s, %s::vector, %s)
            ON CONFLICT (record, chunk_index, model_name)
            DO UPDATE SET chunk_text = EXCLUDED.chunk_text,
                          embedding = EXCLUDED.embedding,
                          created_at = NOW()
            """,
            (record_id, chunk_text, str(embedding), model_name),
        )

    def _log_ingest(
        self,
        conn: psycopg.Connection,  # type: ignore[type-arg]
        record_ids: list[int],
    ) -> None:
        """Log completed ingest in ``rag.ingest_log`` (one row per record)."""
        now = datetime.now(timezone.utc)
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO rag.ingest_log (record, status, started_at, completed_at)
                    VALUES (%s, 'complete', %s, %s)
                    """,
                    [(rid, now, now) for rid in record_ids],
                )
            conn.commit()
        except Exception:
            logger.debug("Could not log ingest (table may not exist)", exc_info=True)
