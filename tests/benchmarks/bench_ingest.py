"""Ingest throughput benchmark.

Measures records/sec for batch ingest at various batch sizes, with a
breakdown of extract time vs embed time vs store time.

Usage:
    pytest tests/benchmarks/bench_ingest.py -v -s
"""

from __future__ import annotations

import json
import os
import time

import psycopg

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.extractor.marc_extractor import extract_record
from tests.benchmarks.conftest import requires_services
from tests.benchmarks.generate_marc import _marc_record_xml


def _insert_synthetic_records(
    conn: psycopg.Connection, count: int  # type: ignore[type-arg]
) -> list[int]:
    """Insert synthetic MARC records into biblio.record_entry, return IDs."""
    ids: list[int] = []
    with conn.cursor() as cur:
        for i in range(count):
            marc = _marc_record_xml(100_000 + i)
            cur.execute(
                "INSERT INTO biblio.record_entry (marc) VALUES (%s) RETURNING id",
                (marc,),
            )
            row = cur.fetchone()
            if row:
                ids.append(row[0])
    conn.commit()
    return ids


def _cleanup_records(
    conn: psycopg.Connection, ids: list[int]  # type: ignore[type-arg]
) -> None:
    """Remove synthetic records and their embeddings."""
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM rag.biblio_embedding WHERE record = ANY(%s)", (ids,)
        )
        cur.execute(
            "DELETE FROM rag.ingest_log WHERE record = ANY(%s)", (ids,)
        )
        cur.execute(
            "DELETE FROM biblio.record_entry WHERE id = ANY(%s)", (ids,)
        )
    conn.commit()


def _bench_batch(
    conn: psycopg.Connection,  # type: ignore[type-arg]
    embedding_service: EmbeddingService,
    records: list[tuple[int, str]],
    batch_size: int,
) -> dict:
    """Run ingest for a set of records at the given batch_size, return timing."""
    total_extract = 0.0
    total_embed = 0.0
    total_store = 0.0
    embedded_count = 0

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]

        # Extract
        t0 = time.perf_counter()
        texts: list[str] = []
        extracted_ids: list[int] = []
        for record_id, marc_xml in batch:
            rec = extract_record(marc_xml, record_id=record_id)
            if rec is not None:
                text = rec.to_embedding_text()
                if text.strip():
                    texts.append(text)
                    extracted_ids.append(record_id)
        t1 = time.perf_counter()
        total_extract += t1 - t0

        if not texts:
            continue

        # Embed
        t2 = time.perf_counter()
        response = embedding_service.embed_batch(texts)
        t3 = time.perf_counter()
        total_embed += t3 - t2

        # Store
        t4 = time.perf_counter()
        with conn.cursor() as cur:
            for rid, emb, txt in zip(extracted_ids, response.embeddings, texts):
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
                    (rid, txt, str(emb), response.model),
                )
                embedded_count += 1
        conn.commit()
        t5 = time.perf_counter()
        total_store += t5 - t4

    total_time = total_extract + total_embed + total_store
    records_per_sec = embedded_count / total_time if total_time > 0 else 0

    return {
        "batch_size": batch_size,
        "record_count": len(records),
        "embedded_count": embedded_count,
        "total_time_sec": round(total_time, 3),
        "extract_time_sec": round(total_extract, 3),
        "embed_time_sec": round(total_embed, 3),
        "store_time_sec": round(total_store, 3),
        "records_per_sec": round(records_per_sec, 2),
    }


BATCH_SIZES = [10, 50, 100, 200]
RECORD_COUNT = int(os.environ.get("BENCH_RECORD_COUNT", "100"))


@requires_services
class TestIngestBenchmark:
    """Ingest throughput benchmarks at various batch sizes."""

    def test_ingest_throughput(self, db_conn, embedding_service):
        """Measure ingest throughput at different batch sizes."""
        conn = db_conn
        record_ids = _insert_synthetic_records(conn, RECORD_COUNT)
        try:
            # Fetch the inserted records
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, marc FROM biblio.record_entry WHERE id = ANY(%s)",
                    (record_ids,),
                )
                records = list(cur.fetchall())

            results = []
            for batch_size in BATCH_SIZES:
                # Clean embeddings between runs
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM rag.biblio_embedding WHERE record = ANY(%s)",
                        (record_ids,),
                    )
                conn.commit()

                result = _bench_batch(conn, embedding_service, records, batch_size)
                results.append(result)
                print(
                    f"\n  batch_size={batch_size}: "
                    f"{result['records_per_sec']} rec/s, "
                    f"total={result['total_time_sec']}s "
                    f"(extract={result['extract_time_sec']}s, "
                    f"embed={result['embed_time_sec']}s, "
                    f"store={result['store_time_sec']}s)"
                )

            # Write JSON report
            report_path = os.environ.get(
                "BENCH_REPORT_DIR", "/tmp/evergreen_benchmarks"
            )
            os.makedirs(report_path, exist_ok=True)
            with open(os.path.join(report_path, "ingest.json"), "w") as f:
                json.dump(results, f, indent=2)

        finally:
            _cleanup_records(conn, record_ids)
