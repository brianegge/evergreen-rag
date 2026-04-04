"""Embedding throughput benchmark.

Measures texts/sec for single and batch embedding at various batch sizes.

Usage:
    pytest tests/benchmarks/bench_embedding.py -v -s
"""

from __future__ import annotations

import json
import os
import random
import time

from tests.benchmarks.conftest import requires_ollama

SAMPLE_TEXTS = [
    "The Art of Programming by John Smith. Subjects: Computer science; Software engineering. "
    "A comprehensive guide covering fundamental concepts and advanced topics.",
    "Introduction to Library Science by Maria Johnson. Subjects: Library science; Cataloging. "
    "This text provides an in-depth exploration of theory and practice.",
    "Modern Database Systems by Robert Williams. Subjects: Database management; Data mining. "
    "An essential reference for students and professionals in the field.",
    "Principles of Information Retrieval by Patricia Brown. Subjects: Information retrieval; "
    "Natural language processing. Covers the latest developments and emerging trends.",
    "A History of Cataloging by Michael Jones. Subjects: Cataloging; Metadata. "
    "A practical handbook with real-world examples and case studies.",
    "Digital Libraries and Archives by Ana Garcia. Subjects: Digital preservation; "
    "Knowledge management. Explores the intersection of technology and information.",
    "Search Engine Design by David Miller. Subjects: Web technologies; Information systems. "
    "A thorough examination of methods, tools, and best practices.",
    "Machine Learning in Practice by Sarah Davis. Subjects: Machine learning; "
    "Artificial intelligence. Applied techniques for real-world problems.",
]

BATCH_SIZES = [1, 5, 10, 25, 50]
ITERATIONS = int(os.environ.get("BENCH_EMBED_ITERATIONS", "10"))


@requires_ollama
class TestEmbeddingBenchmark:
    """Embedding throughput benchmarks."""

    def test_single_embedding(self, embedding_service):
        """Measure throughput for single text embedding."""
        latencies: list[float] = []
        total_texts = ITERATIONS * 5  # 5 texts per iteration

        for _ in range(total_texts):
            text = random.choice(SAMPLE_TEXTS)
            t0 = time.perf_counter()
            embedding_service.embed_text(text)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        total_time = sum(latencies)
        texts_per_sec = total_texts / total_time if total_time > 0 else 0

        result = {
            "mode": "single",
            "total_texts": total_texts,
            "total_time_sec": round(total_time, 3),
            "texts_per_sec": round(texts_per_sec, 2),
            "avg_latency_ms": round((total_time / total_texts) * 1000, 2)
            if total_texts > 0
            else 0,
        }

        print(f"\n  Single embedding: {result['texts_per_sec']} texts/s, "
              f"avg={result['avg_latency_ms']}ms")

        return result

    def test_batch_embedding(self, embedding_service):
        """Measure throughput for batch embedding at various sizes."""
        results = []

        for batch_size in BATCH_SIZES:
            latencies: list[float] = []
            total_texts = 0

            for _ in range(ITERATIONS):
                texts = [random.choice(SAMPLE_TEXTS) for _ in range(batch_size)]
                t0 = time.perf_counter()
                embedding_service.embed_batch(texts)
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                total_texts += len(texts)

            total_time = sum(latencies)
            texts_per_sec = total_texts / total_time if total_time > 0 else 0

            result = {
                "mode": "batch",
                "batch_size": batch_size,
                "iterations": ITERATIONS,
                "total_texts": total_texts,
                "total_time_sec": round(total_time, 3),
                "texts_per_sec": round(texts_per_sec, 2),
                "avg_batch_latency_ms": round(
                    (total_time / ITERATIONS) * 1000, 2
                )
                if ITERATIONS > 0
                else 0,
            }
            results.append(result)

            print(
                f"\n  batch_size={batch_size}: "
                f"{result['texts_per_sec']} texts/s, "
                f"avg_batch={result['avg_batch_latency_ms']}ms"
            )

        # Write JSON report
        report_path = os.environ.get(
            "BENCH_REPORT_DIR", "/tmp/evergreen_benchmarks"
        )
        os.makedirs(report_path, exist_ok=True)
        with open(os.path.join(report_path, "embedding.json"), "w") as f:
            json.dump(results, f, indent=2)
