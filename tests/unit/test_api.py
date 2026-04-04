"""Unit tests for the FastAPI HTTP API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evergreen_rag.api.routes import reciprocal_rank_fusion, router
from evergreen_rag.models.search import SearchResponse, SearchResult

FAKE_EMBEDDING = [0.1] * 768


@pytest.fixture
def app():
    """Create a test app with mocked services (no lifespan)."""
    application = FastAPI()
    application.include_router(router)

    # Mock the services on app.state
    mock_embedding = MagicMock()
    mock_search = MagicMock()
    mock_generation = MagicMock()

    application.state.embedding_service = mock_embedding
    application.state.vector_search = mock_search
    application.state.generation_service = mock_generation

    return application


@pytest.fixture
def client(app):
    """Create a test client."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestSearchEndpoint:
    def test_successful_search(self, client, app):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test query",
            results=[
                SearchResult(record_id=1001, similarity=0.95, chunk_text="hello"),
            ],
            total=1,
            model="nomic-embed-text",
        )

        resp = client.post("/search", json={"query": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test query"
        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["record_id"] == 1001

    def test_search_with_params(self, client, app):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="cats", results=[], total=0, model="nomic-embed-text",
        )

        resp = client.post(
            "/search",
            json={"query": "cats", "limit": 5, "min_similarity": 0.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_search_embedding_service_unavailable(self, client, app):
        app.state.embedding_service.embed_text.side_effect = ConnectionError("down")

        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 503

    def test_search_missing_query(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 422

    def test_search_invalid_limit(self, client):
        resp = client.post("/search", json={"query": "test", "limit": 0})
        assert resp.status_code == 422

    def test_search_vector_store_unavailable(self, client, app):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.side_effect = Exception("db down")

        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 503
        assert "Search error" in resp.json()["detail"]

    def test_search_empty_results(self, client, app):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="obscure query", results=[], total=0, model="nomic-embed-text",
        )

        resp = client.post("/search", json={"query": "obscure query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_search_limit_over_max(self, client):
        resp = client.post("/search", json={"query": "test", "limit": 200})
        assert resp.status_code == 422

    def test_search_invalid_min_similarity(self, client):
        resp = client.post("/search", json={"query": "test", "min_similarity": 1.5})
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_healthy(self, client, app):
        app.state.embedding_service.health_check.return_value = True
        app.state.vector_search.health_check.return_value = True

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["checks"]["embedding_service"] is True
        assert data["checks"]["database"] is True

    def test_degraded(self, client, app):
        app.state.embedding_service.health_check.return_value = True
        app.state.vector_search.health_check.return_value = False

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"

    def test_all_unhealthy(self, client, app):
        app.state.embedding_service.health_check.return_value = False
        app.state.vector_search.health_check.return_value = False

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"

    def test_health_check_exception(self, client, app):
        """Health endpoint returns degraded when checks raise exceptions."""
        app.state.embedding_service.health_check.side_effect = ConnectionError("down")
        app.state.vector_search.health_check.side_effect = ConnectionError("down")

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["embedding_service"] is False
        assert data["checks"]["database"] is False


class TestStatsEndpoint:
    def test_stats(self, client, app):
        app.state.vector_search.get_stats.return_value = {
            "total_embeddings": 100,
            "unique_records": 80,
            "last_embedded_at": "2024-01-01T00:00:00",
            "model_name": "nomic-embed-text",
        }

        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_embeddings"] == 100
        assert data["unique_records"] == 80

    def test_stats_empty(self, client, app):
        app.state.vector_search.get_stats.return_value = {
            "total_embeddings": 0,
            "unique_records": 0,
            "last_embedded_at": None,
            "model_name": None,
        }

        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_embeddings"] == 0

    def test_stats_service_error(self, client, app):
        app.state.vector_search.get_stats.side_effect = Exception("db down")

        resp = client.get("/stats")
        assert resp.status_code == 503


class TestIngestEndpoint:
    def test_ingest_specific_records(self, client, app):
        with patch("evergreen_rag.ingest.pipeline.IngestPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_stats = MagicMock()
            mock_stats.total = 2
            mock_stats.embedded = 2
            mock_stats.failed = 0
            mock_pipeline.run.return_value = mock_stats
            mock_pipeline_cls.return_value = mock_pipeline

            resp = client.post("/ingest", json={"record_ids": [1001, 1002]})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"
            assert data["embedded"] == 2

    def test_ingest_all(self, client, app):
        with patch("evergreen_rag.ingest.pipeline.IngestPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_stats = MagicMock()
            mock_stats.total = 10
            mock_stats.embedded = 10
            mock_stats.failed = 0
            mock_pipeline.run.return_value = mock_stats
            mock_pipeline_cls.return_value = mock_pipeline

            resp = client.post("/ingest", json={"all": True})
            assert resp.status_code == 200

    def test_ingest_failure(self, client, app):
        with patch("evergreen_rag.ingest.pipeline.IngestPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run.side_effect = Exception("db error")

            resp = client.post("/ingest", json={"record_ids": [1]})
            assert resp.status_code == 500

    def test_ingest_incremental(self, client, app):
        """Default ingest (no record_ids, all=False) triggers incremental."""
        with patch("evergreen_rag.ingest.pipeline.IngestPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_stats = MagicMock()
            mock_stats.total = 5
            mock_stats.embedded = 5
            mock_stats.failed = 0
            mock_pipeline.run.return_value = mock_stats
            mock_pipeline_cls.return_value = mock_pipeline

            resp = client.post("/ingest", json={})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"
            mock_pipeline.run.assert_called_once_with(record_ids=None, full=False)

    def test_ingest_partial_failure(self, client, app):
        """Ingest with some records failing."""
        with patch("evergreen_rag.ingest.pipeline.IngestPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_stats = MagicMock()
            mock_stats.total = 10
            mock_stats.embedded = 7
            mock_stats.failed = 3
            mock_pipeline.run.return_value = mock_stats
            mock_pipeline_cls.return_value = mock_pipeline

            resp = client.post("/ingest", json={"record_ids": list(range(10))})
            assert resp.status_code == 200
            data = resp.json()
            assert data["embedded"] == 7
            assert data["failed"] == 3


class TestReciprocalRankFusion:
    """Unit tests for the RRF helper function."""

    def test_single_ranking(self):
        merged = reciprocal_rank_fusion([10, 20, 30])
        ids = [rid for rid, _ in merged]
        assert ids == [10, 20, 30]

    def test_two_rankings_overlap(self):
        # Record 20 appears in both rankings at good positions
        merged = reciprocal_rank_fusion([10, 20, 30], [20, 40, 10])
        ids = [rid for rid, _ in merged]
        # 20 appears rank 2 and rank 1 -> highest combined score
        assert ids[0] == 20
        # 10 appears rank 1 and rank 3
        assert 10 in ids

    def test_disjoint_rankings(self):
        merged = reciprocal_rank_fusion([1, 2], [3, 4])
        ids = [rid for rid, _ in merged]
        assert set(ids) == {1, 2, 3, 4}

    def test_weights(self):
        # Heavily weight the second ranking
        merged = reciprocal_rank_fusion(
            [10, 20], [30, 40], weights=[0.1, 10.0]
        )
        ids = [rid for rid, _ in merged]
        # Second ranking's top result should dominate
        assert ids[0] == 30

    def test_empty_ranking(self):
        merged = reciprocal_rank_fusion([])
        assert merged == []

    def test_scores_are_positive(self):
        merged = reciprocal_rank_fusion([1, 2, 3], [2, 3, 4])
        for _, score in merged:
            assert score > 0


class TestSearchWithGeneration:
    """Tests for POST /search with generate=true."""

    def test_search_with_generate_returns_summary(self, client, app):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="grief teens",
            results=[
                SearchResult(
                    record_id=1, similarity=0.9, chunk_text="grief book"
                ),
            ],
            total=1,
            model="nomic-embed-text",
        )
        app.state.generation_service.summarize.return_value = (
            "This result covers grief resources for teens."
        )

        resp = client.post(
            "/search",
            json={"query": "grief teens", "generate": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated_text"] == (
            "This result covers grief resources for teens."
        )
        assert data["total"] == 1

    def test_search_generate_false_no_summary(self, client, app):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test", results=[], total=0, model="nomic-embed-text",
        )

        resp = client.post(
            "/search", json={"query": "test", "generate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated_text"] is None
        app.state.generation_service.summarize.assert_not_called()

    def test_search_generate_service_unavailable(self, client, app):
        """Search still returns results when generation is None."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    record_id=1, similarity=0.9, chunk_text="book"
                ),
            ],
            total=1,
            model="nomic-embed-text",
        )
        app.state.generation_service = None

        resp = client.post(
            "/search", json={"query": "test", "generate": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["generated_text"] is None

    def test_search_generate_failure_still_returns_results(
        self, client, app
    ):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    record_id=1, similarity=0.9, chunk_text="book"
                ),
            ],
            total=1,
            model="nomic-embed-text",
        )
        app.state.generation_service.summarize.side_effect = Exception(
            "LLM down"
        )

        resp = client.post(
            "/search", json={"query": "test", "generate": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["generated_text"] is None

    def test_search_generate_empty_results_skips_generation(
        self, client, app
    ):
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test", results=[], total=0, model="nomic-embed-text",
        )

        resp = client.post(
            "/search", json={"query": "test", "generate": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated_text"] is None
        app.state.generation_service.summarize.assert_not_called()


class TestRecommendEndpoint:
    """Tests for POST /recommend."""

    def test_recommend_success(self, client, app):
        app.state.generation_service.recommend.return_value = (
            "Start with The Great Gatsby."
        )
        resp = client.post(
            "/recommend",
            json={
                "query": "fitzgerald",
                "results": [
                    {
                        "record_id": 1,
                        "similarity": 0.9,
                        "chunk_text": "The Great Gatsby",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == "Start with The Great Gatsby."

    def test_recommend_no_generation_service(self, client, app):
        app.state.generation_service = None
        resp = client.post(
            "/recommend",
            json={
                "query": "test",
                "results": [
                    {
                        "record_id": 1,
                        "similarity": 0.9,
                        "chunk_text": "book",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] is None

    def test_recommend_generation_failure(self, client, app):
        app.state.generation_service.recommend.side_effect = Exception("fail")
        resp = client.post(
            "/recommend",
            json={
                "query": "test",
                "results": [
                    {
                        "record_id": 1,
                        "similarity": 0.9,
                        "chunk_text": "book",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] is None


class TestRefineEndpoint:
    """Tests for POST /refine."""

    def test_refine_success(self, client, app):
        app.state.generation_service.refine.return_value = [
            "F. Scott Fitzgerald biography",
            "Jazz Age literature",
        ]
        resp = client.post(
            "/refine",
            json={
                "query": "fitzgerald",
                "results": [
                    {
                        "record_id": 1,
                        "similarity": 0.9,
                        "chunk_text": "The Great Gatsby",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) == 2
        assert "F. Scott Fitzgerald biography" in data["suggestions"]

    def test_refine_no_generation_service(self, client, app):
        app.state.generation_service = None
        resp = client.post(
            "/refine",
            json={
                "query": "test",
                "results": [
                    {
                        "record_id": 1,
                        "similarity": 0.9,
                        "chunk_text": "book",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["suggestions"] == []

    def test_refine_generation_failure(self, client, app):
        app.state.generation_service.refine.side_effect = Exception("fail")
        resp = client.post(
            "/refine",
            json={
                "query": "test",
                "results": [
                    {
                        "record_id": 1,
                        "similarity": 0.9,
                        "chunk_text": "book",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["suggestions"] == []


class TestMergedSearchEndpoint:
    """Tests for POST /search/merged."""

    def test_merged_search_both_inputs(self, client, app):
        """Merged search with both semantic query and keyword results."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="grief teenagers",
            results=[
                SearchResult(record_id=100, similarity=0.9, chunk_text="coping with grief"),
                SearchResult(record_id=200, similarity=0.8, chunk_text="teen support"),
            ],
            total=2,
            model="nomic-embed-text",
        )

        resp = client.post(
            "/search/merged",
            json={
                "query": "grief teenagers",
                "keyword_results": [200, 300, 400],
                "limit": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # 200 appears in both rankings, should be ranked high
        ids = [r["record_id"] for r in data["results"]]
        assert 200 in ids
        assert data["total"] <= 5

    def test_merged_search_semantic_only(self, client, app):
        """Merged search with query but no keyword results."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="python programming",
            results=[
                SearchResult(record_id=50, similarity=0.85, chunk_text="python book"),
            ],
            total=1,
            model="nomic-embed-text",
        )

        resp = client.post(
            "/search/merged",
            json={"query": "python programming", "keyword_results": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["record_id"] == 50
        assert data["results"][0]["chunk_text"] == "python book"

    def test_merged_search_keyword_only(self, client, app):
        """Merged search with keyword results but no query."""
        resp = client.post(
            "/search/merged",
            json={"keyword_results": [10, 20, 30]},
        )
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["record_id"] for r in data["results"]]
        assert ids == [10, 20, 30]
        # No semantic data, so similarity should be 0
        for r in data["results"]:
            assert r["similarity"] == 0.0

    def test_merged_search_rrf_ranking(self, client, app):
        """Verify RRF produces correct ordering when a record appears in both."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        # Semantic ranking: [1, 2, 3]
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(record_id=1, similarity=0.9, chunk_text="a"),
                SearchResult(record_id=2, similarity=0.8, chunk_text="b"),
                SearchResult(record_id=3, similarity=0.7, chunk_text="c"),
            ],
            total=3,
            model="nomic-embed-text",
        )

        # Keyword ranking: [2, 4, 1]
        resp = client.post(
            "/search/merged",
            json={
                "query": "test",
                "keyword_results": [2, 4, 1],
                "limit": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["record_id"] for r in data["results"]]
        # Record 2: semantic rank 2 + keyword rank 1 -> best combined
        # Record 1: semantic rank 1 + keyword rank 3 -> second best
        assert ids[0] == 2
        assert ids[1] == 1

    def test_merged_search_with_weights(self, client, app):
        """Custom weights shift ranking toward the weighted source."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(record_id=1, similarity=0.9, chunk_text="a"),
            ],
            total=1,
            model="nomic-embed-text",
        )

        resp = client.post(
            "/search/merged",
            json={
                "query": "test",
                "keyword_results": [2],
                "limit": 10,
                "weights": {"semantic": 0.1, "keyword": 10.0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["record_id"] for r in data["results"]]
        # Keyword heavily weighted, so record 2 should come first
        assert ids[0] == 2

    def test_merged_search_neither_input(self, client):
        """Error when neither query nor keyword_results are provided."""
        resp = client.post(
            "/search/merged",
            json={"keyword_results": []},
        )
        assert resp.status_code == 422

    def test_merged_search_embedding_error(self, client, app):
        """503 when embedding service fails."""
        app.state.embedding_service.embed_text.side_effect = ConnectionError("down")

        resp = client.post(
            "/search/merged",
            json={"query": "test", "keyword_results": [1]},
        )
        assert resp.status_code == 503
        assert "Embedding service error" in resp.json()["detail"]

    def test_merged_search_vector_store_error(self, client, app):
        """503 when vector search fails."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.side_effect = Exception("db down")

        resp = client.post(
            "/search/merged",
            json={"query": "test", "keyword_results": [1]},
        )
        assert resp.status_code == 503
        assert "Search error" in resp.json()["detail"]

    def test_merged_search_limit_applied(self, client, app):
        """Results are truncated to the requested limit."""
        app.state.embedding_service.embed_text.return_value = FAKE_EMBEDDING
        app.state.vector_search.similarity_search.return_value = SearchResponse(
            query="test",
            results=[
                SearchResult(record_id=i, similarity=0.5, chunk_text=f"r{i}")
                for i in range(1, 6)
            ],
            total=5,
            model="nomic-embed-text",
        )

        resp = client.post(
            "/search/merged",
            json={
                "query": "test",
                "keyword_results": list(range(6, 11)),
                "limit": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
