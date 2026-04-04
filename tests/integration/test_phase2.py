"""Integration tests for Phase 2 features: merged search, ingest hook, graceful degradation."""

from testplan import test_plan
from testplan.testing.multitest import MultiTest, testcase, testsuite

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.ingest.pipeline import IngestPipeline
from evergreen_rag.models.search import SearchQuery
from evergreen_rag.search.vector_search import VectorSearch
from tests.integration.drivers import EmbeddingServiceDriver, PostgresDriver


def _make_test_client(db_url, ollama_url, model):
    """Create a FastAPI TestClient with configured services."""
    from starlette.testclient import TestClient

    from evergreen_rag.api.main import create_app

    app = create_app()

    # Override the lifespan-managed services with test-configured ones
    embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
    vector_search = VectorSearch(db_url=db_url)
    vector_search.open()

    app.state.embedding_service = embedding_svc
    app.state.vector_search = vector_search

    client = TestClient(app, raise_server_exceptions=False)
    return client, vector_search


@testsuite
class SetupSuite:
    """Ensure embeddings exist before running Phase 2 tests."""

    @testcase
    def test_ingest_sample_records(self, env, result):
        """Run ingest so that semantic search has data to work with."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        pipeline = IngestPipeline(
            db_url=db_url,
            embedding_service=embedding_svc,
            batch_size=10,
        )

        stats = pipeline.run(full=True)
        result.log(
            f"Setup ingest: total={stats.total} embedded={stats.embedded} "
            f"failed={stats.failed}"
        )
        result.gt(stats.embedded, 0, "Must have embeddings for Phase 2 tests")


@testsuite
class MergedSearchSuite:
    """Tests for POST /search/merged endpoint."""

    @testcase
    def test_merged_semantic_and_keyword(self, env, result):
        """Merge semantic query with keyword result IDs via RRF."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        # Get real semantic results first
        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        search = VectorSearch(db_url=db_url)

        query_text = "books about racism and antiracism"
        query_embedding = embedding_svc.embed_text(query_text)
        search_query = SearchQuery(query=query_text, limit=5)
        sem_response = search.similarity_search(query_embedding, search_query)
        semantic_ids = [r.record_id for r in sem_response.results]

        result.log(f"Semantic search returned IDs: {semantic_ids}")
        result.gt(len(semantic_ids), 0, "Semantic search must return results")

        # Build keyword results: overlap some IDs, add some new ones
        keyword_ids = list(semantic_ids[:2]) + [999901, 999902, 999903]

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "query": query_text,
                    "keyword_results": keyword_ids,
                    "limit": 10,
                    "weights": {"semantic": 1.0, "keyword": 1.0},
                },
            )

            result.equal(resp.status_code, 200, "Merged search should return 200")
            data = resp.json()

            result.log(f"Merged results: {len(data['results'])} items")
            result.gt(data["total"], 0, "Merged search must return results")

            merged_ids = [r["record_id"] for r in data["results"]]

            # Records appearing in both lists should rank higher via RRF
            overlapping = set(semantic_ids[:2]) & set(keyword_ids[:2])
            if overlapping:
                for oid in overlapping:
                    if oid in merged_ids:
                        rank = merged_ids.index(oid) + 1
                        result.log(f"Overlapping record {oid} ranked at position {rank}")
                        result.le(
                            rank,
                            5,
                            f"Record {oid} (in both rankings) should be in top 5",
                        )

            # Keyword-only IDs (999901, etc.) should appear in results
            for kid in [999901, 999902, 999903]:
                result.true(
                    kid in merged_ids,
                    f"Keyword-only record {kid} should appear in merged results",
                )
        finally:
            vs.close()

    @testcase
    def test_merged_semantic_only(self, env, result):
        """Merged endpoint with only a semantic query (no keyword results)."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "query": "children's fantasy novel about wizards",
                    "keyword_results": [],
                    "limit": 5,
                },
            )

            result.equal(resp.status_code, 200, "Semantic-only merged should return 200")
            data = resp.json()
            result.gt(data["total"], 0, "Should return semantic results")
            result.log(f"Semantic-only merged: {data['total']} results")

            # All results should have chunk_text (from semantic search)
            for r in data["results"]:
                result.true(
                    len(r["chunk_text"]) > 0,
                    f"Record {r['record_id']} should have chunk_text from semantic search",
                )
        finally:
            vs.close()

    @testcase
    def test_merged_keyword_only(self, env, result):
        """Merged endpoint with only keyword results (no semantic query)."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        keyword_ids = [1001, 1002, 1003]
        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "keyword_results": keyword_ids,
                    "limit": 10,
                },
            )

            result.equal(resp.status_code, 200, "Keyword-only merged should return 200")
            data = resp.json()
            result.equal(data["total"], len(keyword_ids), "Should return all keyword IDs")

            merged_ids = [r["record_id"] for r in data["results"]]
            # Order should be preserved (single ranking, RRF preserves original order)
            result.equal(
                merged_ids,
                keyword_ids,
                "Keyword-only results should preserve original ranking order",
            )

            # chunk_text should be empty for keyword-only results (no semantic data)
            for r in data["results"]:
                result.equal(
                    r["chunk_text"],
                    "",
                    f"Record {r['record_id']} should have empty chunk_text (keyword only)",
                )
        finally:
            vs.close()

    @testcase
    def test_merged_rrf_ranking_order(self, env, result):
        """Verify RRF produces expected ranking when items appear in both lists."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "query": "racism and antiracism in America",
                    "keyword_results": [1001, 1003, 1005],
                    "limit": 10,
                    "weights": {"semantic": 1.0, "keyword": 1.0},
                },
            )

            result.equal(resp.status_code, 200, "RRF ranking test should return 200")
            data = resp.json()

            merged_ids = [r["record_id"] for r in data["results"]]
            result.log(f"RRF merged order: {merged_ids}")

            # Record 1001 (antiracism) should appear in results since it matches
            # both the semantic query and is first in keyword results
            if 1001 in merged_ids:
                rank = merged_ids.index(1001) + 1
                result.le(
                    rank,
                    3,
                    "Record 1001 (appears in both rankings) should be top 3",
                )
        finally:
            vs.close()

    @testcase
    def test_merged_empty_request_rejected(self, env, result):
        """Merged endpoint rejects requests with neither query nor keyword results."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "keyword_results": [],
                    "limit": 10,
                },
            )

            result.equal(
                resp.status_code,
                422,
                "Request with no query and no keyword_results should return 422",
            )
        finally:
            vs.close()

    @testcase
    def test_merged_custom_weights(self, env, result):
        """Merged search with custom weights favoring semantic over keyword."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        keyword_ids = [1005, 1006, 1007]
        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "query": "books about racism and antiracism",
                    "keyword_results": keyword_ids,
                    "limit": 10,
                    "weights": {"semantic": 2.0, "keyword": 0.5},
                },
            )

            result.equal(resp.status_code, 200, "Weighted merged should return 200")
            data = resp.json()
            result.gt(data["total"], 0, "Should return results")
            result.log(
                f"Weighted merge (semantic=2.0, keyword=0.5): "
                f"{[r['record_id'] for r in data['results']]}"
            )
        finally:
            vs.close()


@testsuite
class IngestHookSuite:
    """Tests for the ingest endpoint as called by the Perl ingest hook."""

    @testcase
    def test_ingest_specific_records(self, env, result):
        """POST /ingest with specific record IDs (simulating Perl hook call)."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/ingest",
                json={"record_ids": [1001, 1002]},
            )

            result.equal(resp.status_code, 200, "Ingest should return 200")
            data = resp.json()
            result.equal(data["status"], "completed", "Status should be 'completed'")
            result.gt(data["embedded"], 0, "Should have embedded at least one record")
            result.log(
                f"Ingest result: total={data['total']} embedded={data['embedded']} "
                f"failed={data['failed']}"
            )
        finally:
            vs.close()

    @testcase
    def test_ingest_single_record(self, env, result):
        """POST /ingest with a single record ID (most common hook scenario)."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/ingest",
                json={"record_ids": [1001]},
            )

            result.equal(resp.status_code, 200, "Single record ingest should return 200")
            data = resp.json()
            result.equal(data["status"], "completed", "Status should be 'completed'")
            result.log(f"Single record ingest: {data}")
        finally:
            vs.close()

    @testcase
    def test_ingest_response_format(self, env, result):
        """Verify ingest response matches the contract documented for Perl clients."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/ingest",
                json={"record_ids": [1001]},
            )

            data = resp.json()

            # Verify all expected fields are present
            for field in ["status", "message", "total", "embedded", "failed"]:
                result.true(
                    field in data,
                    f"Response must contain '{field}' field",
                )

            # Verify types
            result.true(isinstance(data["status"], str), "status should be a string")
            result.true(isinstance(data["message"], str), "message should be a string")
            result.true(isinstance(data["total"], int), "total should be an integer")
            result.true(isinstance(data["embedded"], int), "embedded should be an integer")
            result.true(isinstance(data["failed"], int), "failed should be an integer")
        finally:
            vs.close()


@testsuite
class GracefulDegradationSuite:
    """Tests for error handling and graceful degradation."""

    @testcase
    def test_search_merged_invalid_body(self, env, result):
        """Merged search with invalid request body returns 422."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={"limit": "not_a_number"},
            )

            result.equal(
                resp.status_code,
                422,
                "Invalid request body should return 422 validation error",
            )
        finally:
            vs.close()

    @testcase
    def test_search_merged_empty_query_with_results(self, env, result):
        """Merged search with empty string query but keyword results should work."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.post(
                "/search/merged",
                json={
                    "query": "",
                    "keyword_results": [1001, 1002],
                    "limit": 10,
                },
            )

            # Empty query string with keyword_results should still succeed
            result.equal(resp.status_code, 200, "Empty query with keyword IDs should work")
            data = resp.json()
            result.equal(data["total"], 2, "Should return the keyword results")
        finally:
            vs.close()

    @testcase
    def test_health_endpoint_accessible(self, env, result):
        """Health endpoint should always respond."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.get("/health")
            result.equal(resp.status_code, 200, "Health endpoint should return 200")
            data = resp.json()
            result.true("status" in data, "Health response must have 'status' field")
            result.true("checks" in data, "Health response must have 'checks' field")
            result.log(f"Health: {data}")
        finally:
            vs.close()

    @testcase
    def test_stats_endpoint_accessible(self, env, result):
        """Stats endpoint should respond with embedding statistics."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.get("/stats")
            result.equal(resp.status_code, 200, "Stats endpoint should return 200")
            data = resp.json()
            result.true(
                "total_embeddings" in data,
                "Stats must include total_embeddings",
            )
            result.log(f"Stats: {data}")
        finally:
            vs.close()

    @testcase
    def test_nonexistent_endpoint_returns_404(self, env, result):
        """Requesting a non-existent endpoint returns 404, not 500."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            resp = client.get("/nonexistent")
            result.equal(
                resp.status_code,
                404,
                "Non-existent endpoint should return 404",
            )
        finally:
            vs.close()

    @testcase
    def test_search_with_extreme_parameters(self, env, result):
        """Search with boundary parameters should return appropriate errors."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        client, vs = _make_test_client(db_url, ollama_url, model)
        try:
            # limit=0 should be rejected (min is 1)
            resp = client.post(
                "/search",
                json={"query": "test", "limit": 0},
            )
            result.equal(
                resp.status_code,
                422,
                "limit=0 should be rejected with 422",
            )

            # limit=101 should be rejected (max is 100)
            resp = client.post(
                "/search",
                json={"query": "test", "limit": 101},
            )
            result.equal(
                resp.status_code,
                422,
                "limit=101 should be rejected with 422",
            )
        finally:
            vs.close()


def make_multitest():
    return MultiTest(
        name="Phase 2 Integration",
        suites=[
            SetupSuite(),
            MergedSearchSuite(),
            IngestHookSuite(),
            GracefulDegradationSuite(),
        ],
        environment=[
            PostgresDriver(name="db"),
            EmbeddingServiceDriver(name="embedding"),
        ],
    )


@test_plan(name="Evergreen RAG Phase 2 Integration Tests")
def main(plan):
    plan.add(make_multitest())


if __name__ == "__main__":
    main()
