"""PostgreSQL LISTEN/NOTIFY listener for automatic RAG ingest.

Listens on the ``rag_ingest`` channel for events emitted by the
``biblio.notify_rag_ingest()`` trigger. Payload format:

    upsert:<record_id>   — record inserted or MARC updated
    delete:<record_id>   — record deleted or soft-deleted

Batches incoming events and processes them after a short debounce
window, so rapid bulk operations are handled efficiently.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import psycopg

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.ingest.pipeline import IngestPipeline

logger = logging.getLogger(__name__)

# How long to wait after the last notification before flushing the batch.
DEBOUNCE_SECONDS = float(os.environ.get("RAG_INGEST_DEBOUNCE", "5.0"))

# Maximum batch size before forcing a flush regardless of debounce.
MAX_BATCH_SIZE = int(os.environ.get("RAG_INGEST_MAX_BATCH", "500"))


def _parse_payload(payload: str) -> tuple[str, int] | None:
    """Parse 'upsert:123' or 'delete:456' into (action, record_id)."""
    try:
        action, id_str = payload.split(":", 1)
        if action in ("upsert", "delete"):
            return action, int(id_str)
    except (ValueError, TypeError):
        pass
    return None


class IngestListener:
    """Background thread that listens for ``rag_ingest`` notifications.

    Parameters
    ----------
    db_url:
        PostgreSQL connection string.
    embedding_service:
        Shared embedding service instance from the app.
    batch_size:
        Pipeline batch size for embedding.
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
        self.embedding_service = embedding_service
        self.batch_size = batch_size

        self._pending_upserts: set[int] = set()
        self._pending_deletes: set[int] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the listener thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="rag-ingest-listener",
            daemon=True,
        )
        self._thread.start()
        logger.info("Ingest listener started (debounce=%.1fs)", DEBOUNCE_SECONDS)

    def stop(self) -> None:
        """Signal the listener to stop and wait for it."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        self._thread = None
        # Flush any remaining events
        self._flush()
        logger.info("Ingest listener stopped")

    def _listen_loop(self) -> None:
        """Main loop: LISTEN on rag_ingest, collect events, flush periodically."""
        while not self._stop_event.is_set():
            try:
                self._listen_once()
            except Exception:
                logger.exception("Listener connection failed, reconnecting in 5s")
                self._stop_event.wait(5.0)

    def _listen_once(self) -> None:
        """Open a connection, LISTEN, and process notifications until stopped."""
        conn = psycopg.connect(self.db_url, autocommit=True)
        try:
            conn.execute("LISTEN rag_ingest")
            logger.debug("LISTEN rag_ingest active")

            last_notify_time = 0.0

            while not self._stop_event.is_set():
                # Wait up to 1 second for a notification
                gen = conn.notifies(timeout=1.0)
                for notify in gen:
                    parsed = _parse_payload(notify.payload)
                    if parsed is None:
                        continue
                    action, record_id = parsed

                    with self._lock:
                        if action == "delete":
                            self._pending_deletes.add(record_id)
                            self._pending_upserts.discard(record_id)
                        else:
                            self._pending_upserts.add(record_id)
                            self._pending_deletes.discard(record_id)

                    last_notify_time = time.monotonic()

                    # Force flush if batch is large
                    pending_count = len(self._pending_upserts) + len(self._pending_deletes)
                    if pending_count >= MAX_BATCH_SIZE:
                        self._flush()
                        last_notify_time = 0.0
                    break  # exit gen to check stop_event and debounce

                # Debounce flush
                pending_count = len(self._pending_upserts) + len(self._pending_deletes)
                if pending_count > 0 and last_notify_time > 0:
                    elapsed = time.monotonic() - last_notify_time
                    if elapsed >= DEBOUNCE_SECONDS:
                        self._flush()
                        last_notify_time = 0.0

        finally:
            conn.close()

    def _flush(self) -> None:
        """Process accumulated upserts and deletes."""
        with self._lock:
            upsert_ids = list(self._pending_upserts)
            delete_ids = list(self._pending_deletes)
            self._pending_upserts.clear()
            self._pending_deletes.clear()

        if delete_ids:
            self._process_deletes(delete_ids)

        if upsert_ids:
            self._process_upserts(upsert_ids)

    def _process_upserts(self, record_ids: list[int]) -> None:
        """Run the ingest pipeline to embed new/updated records."""
        logger.info("Auto-ingest: embedding %d records", len(record_ids))
        try:
            pipeline = IngestPipeline(
                db_url=self.db_url,
                embedding_service=self.embedding_service,
                batch_size=self.batch_size,
            )
            stats = pipeline.run(record_ids=record_ids)
            logger.info(
                "Auto-ingest complete: %d embedded, %d failed",
                stats.embedded,
                stats.failed,
            )
        except Exception:
            logger.exception("Auto-ingest failed for %d records", len(record_ids))

    def _process_deletes(self, record_ids: list[int]) -> None:
        """Remove embeddings for deleted records."""
        logger.info("Auto-ingest: removing embeddings for %d records", len(record_ids))
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM rag.biblio_embedding WHERE record = ANY(%s)",
                        (record_ids,),
                    )
                    deleted = cur.rowcount
                conn.commit()
            logger.info("Removed %d embeddings for %d deleted records", deleted, len(record_ids))
        except Exception:
            logger.exception("Failed to remove embeddings for deleted records")
