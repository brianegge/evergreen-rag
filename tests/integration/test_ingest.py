"""Integration tests for the ingest pipeline."""

import psycopg
from testplan import test_plan
from testplan.testing.multitest import MultiTest, testcase, testsuite

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.ingest.pipeline import IngestPipeline
from evergreen_rag.models.search import SearchQuery
from evergreen_rag.search.vector_search import VectorSearch
from tests.integration.drivers import (
    EmbeddingServiceDriver,
    PostgresDriver,
    load_quality_queries,
)


@testsuite
class IngestSuite:
    """Tests for MARC record ingestion and embedding."""

    @testcase
    def test_marc_to_embedding(self, env, result):
        """Ingest a MARC record from sample_records.xml, verify embedding is stored."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        pipeline = IngestPipeline(
            db_url=db_url,
            embedding_service=embedding_svc,
            batch_size=10,
        )

        # Run ingest for all sample records
        stats = pipeline.run(full=True)

        result.log(
            "Ingest stats: total=%d extracted=%d embedded=%d failed=%d",
            stats.total,
            stats.extracted,
            stats.embedded,
            stats.failed,
        )

        # Verify records were processed
        result.gt(stats.total, 0, "Should have found records to process")
        result.gt(stats.embedded, 0, "Should have embedded at least one record")
        result.equal(stats.failed, 0, "Should have zero failures")

        # Verify embeddings exist in the database
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rag.biblio_embedding")
                count = cur.fetchone()[0]
                result.gt(count, 0, "rag.biblio_embedding should have rows")

                # Verify specific record (1001 = "How to be an antiracist")
                cur.execute(
                    "SELECT record, chunk_text, model_name "
                    "FROM rag.biblio_embedding WHERE record = 1001"
                )
                row = cur.fetchone()
                result.true(row is not None, "Record 1001 should have an embedding")
                result.equal(row[2], model, "Model name should match configured model")
                result.true(
                    len(row[1]) > 0, "chunk_text should not be empty"
                )

                # Verify ingest log
                cur.execute(
                    "SELECT COUNT(*) FROM rag.ingest_log WHERE status = 'complete'"
                )
                log_count = cur.fetchone()[0]
                result.gt(log_count, 0, "Ingest log should have completed entries")


@testsuite
class RetrievalQualitySuite:
    """Tests validating semantic search quality.

    Assumes IngestSuite has already run and populated embeddings.
    """

    @testcase
    def test_known_item_retrieval(self, env, result):
        """A query derived from a record's title/subject must return that record in top 5."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        search = VectorSearch(db_url=db_url)

        queries = load_quality_queries()
        # Use queries with single expected records for known-item tests
        known_item_queries = [
            q for q in queries
            if q.get("expected_records") and len(q["expected_records"]) == 1
        ]

        passed = 0
        total = len(known_item_queries)

        for q in known_item_queries:
            query_text = q["query"]
            expected_id = q["expected_records"][0]

            query_embedding = embedding_svc.embed_text(query_text)
            search_query = SearchQuery(query=query_text, limit=5)
            response = search.similarity_search(query_embedding, search_query)

            result_ids = [r.record_id for r in response.results]
            found = expected_id in result_ids

            if found:
                passed += 1
                result.log(
                    "PASS [%s]: record %d found in top 5 for '%s'",
                    q["id"], expected_id, query_text,
                )
            else:
                result.log(
                    "FAIL [%s]: record %d NOT in top 5 for '%s' (got %s)",
                    q["id"], expected_id, query_text, result_ids,
                )

        # At least 70% of known-item queries should succeed
        pass_rate = passed / total if total > 0 else 0
        result.ge(
            pass_rate, 0.7,
            f"Known-item retrieval pass rate {passed}/{total} = {pass_rate:.0%} "
            f"should be >= 70%",
        )

    @testcase
    def test_semantic_synonym_match(self, env, result):
        """Queries using synonyms of subject headings must return relevant records."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        search = VectorSearch(db_url=db_url)

        # These queries use synonyms not in the MARC records
        synonym_queries = [
            {
                "query": "how DNA and heredity shape who we are",
                "expected": 1003,
                "note": "DNA/heredity -> genetics/genes",
            },
            {
                "query": "impact of robots and computers on jobs",
                "expected": 1006,
                "note": "robots/computers/jobs -> automation/AI/labor",
            },
            {
                "query": "children's book about a wizard school",
                "expected": 1002,
                "note": "wizard school -> Hogwarts/witches",
            },
            {
                "query": "learning to cook with basic principles",
                "expected": 1004,
                "note": "cooking principles -> salt/fat/acid/heat",
            },
        ]

        passed = 0
        total = len(synonym_queries)

        for sq in synonym_queries:
            query_embedding = embedding_svc.embed_text(sq["query"])
            search_query = SearchQuery(query=sq["query"], limit=5)
            response = search.similarity_search(query_embedding, search_query)

            result_ids = [r.record_id for r in response.results]
            found = sq["expected"] in result_ids

            if found:
                passed += 1
                result.log(
                    "PASS: record %d found for synonym query '%s' (%s)",
                    sq["expected"], sq["query"], sq["note"],
                )
            else:
                result.log(
                    "FAIL: record %d NOT found for synonym query '%s' (got %s)",
                    sq["expected"], sq["query"], result_ids,
                )

        pass_rate = passed / total if total > 0 else 0
        result.ge(
            pass_rate, 0.75,
            f"Synonym match pass rate {passed}/{total} = {pass_rate:.0%} "
            f"should be >= 75%",
        )

    @testcase
    def test_negative_irrelevant_query(self, env, result):
        """Unrelated queries must not surface specific records above similarity threshold."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        search = VectorSearch(db_url=db_url)

        queries = load_quality_queries()
        negative_queries = [q for q in queries if "excluded_records" in q]

        for q in negative_queries:
            query_text = q["query"]
            threshold = q.get("min_similarity_threshold", 0.7)
            excluded = set(q["excluded_records"])

            query_embedding = embedding_svc.embed_text(query_text)
            search_query = SearchQuery(
                query=query_text, limit=10, min_similarity=0.0,
            )
            response = search.similarity_search(query_embedding, search_query)

            # No excluded record should appear above the similarity threshold
            high_sim_excluded = [
                r for r in response.results
                if r.record_id in excluded and r.similarity >= threshold
            ]

            result.equal(
                len(high_sim_excluded), 0,
                f"Query '{query_text}' should not match excluded records "
                f"above {threshold} similarity (found: "
                f"{[(r.record_id, round(r.similarity, 3)) for r in high_sim_excluded]})",
            )

            if response.results:
                top_sim = response.results[0].similarity
                result.log(
                    "Negative query '%s': top similarity = %.3f (threshold=%.2f)",
                    query_text, top_sim, threshold,
                )


def make_multitest():
    return MultiTest(
        name="Evergreen RAG Integration",
        suites=[IngestSuite(), RetrievalQualitySuite()],
        environment=[
            PostgresDriver(name="db"),
            EmbeddingServiceDriver(name="embedding"),
        ],
    )


@test_plan(name="Evergreen RAG Integration Tests")
def main(plan):
    plan.add(make_multitest())


if __name__ == "__main__":
    main()
