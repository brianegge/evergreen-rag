"""Query latency benchmark.

Measures p50/p95/p99 query latency over N queries at the current
collection size.

Usage:
    pytest tests/benchmarks/bench_search.py -v -s
"""

from __future__ import annotations

import json
import os
import random
import statistics
import time

from evergreen_rag.models.search import SearchQuery
from tests.benchmarks.conftest import requires_services

QUERY_TEXTS = [
    "machine learning algorithms for text classification",
    "history of library cataloging systems",
    "python programming best practices",
    "database indexing and query optimization",
    "digital preservation of archival materials",
    "natural language processing applications",
    "information retrieval models and techniques",
    "open source software development methodology",
    "metadata standards for digital libraries",
    "cloud computing architecture patterns",
    "data structures for search engines",
    "bibliographic record management",
    "distributed systems fault tolerance",
    "applied cryptography and security",
    "statistical analysis methods",
]

NUM_QUERIES = int(os.environ.get("BENCH_QUERY_COUNT", "100"))


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@requires_services
class TestSearchBenchmark:
    """Query latency benchmarks."""

    def test_query_latency(self, embedding_service, vector_search):
        """Measure p50/p95/p99 query latency over many queries."""
        latencies: list[float] = []

        for i in range(NUM_QUERIES):
            query_text = random.choice(QUERY_TEXTS)

            # Embed the query
            t0 = time.perf_counter()
            query_embedding = embedding_service.embed_text(query_text)
            embed_time = time.perf_counter() - t0

            # Search
            search_query = SearchQuery(query=query_text, limit=10)
            t1 = time.perf_counter()
            vector_search.similarity_search(query_embedding, search_query)
            search_time = time.perf_counter() - t1

            total_time = embed_time + search_time
            latencies.append(total_time)

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        avg = statistics.mean(latencies) if latencies else 0

        result = {
            "num_queries": NUM_QUERIES,
            "p50_ms": round(p50 * 1000, 2),
            "p95_ms": round(p95 * 1000, 2),
            "p99_ms": round(p99 * 1000, 2),
            "avg_ms": round(avg * 1000, 2),
            "min_ms": round(min(latencies) * 1000, 2) if latencies else 0,
            "max_ms": round(max(latencies) * 1000, 2) if latencies else 0,
            "queries_per_sec": round(
                len(latencies) / sum(latencies), 2
            )
            if sum(latencies) > 0
            else 0,
        }

        print(f"\n  Query latency ({NUM_QUERIES} queries):")
        print(f"    p50={result['p50_ms']}ms  p95={result['p95_ms']}ms  p99={result['p99_ms']}ms")
        print(f"    avg={result['avg_ms']}ms  qps={result['queries_per_sec']}")

        # Write JSON report
        report_path = os.environ.get(
            "BENCH_REPORT_DIR", "/tmp/evergreen_benchmarks"
        )
        os.makedirs(report_path, exist_ok=True)
        with open(os.path.join(report_path, "search.json"), "w") as f:
            json.dump(result, f, indent=2)
