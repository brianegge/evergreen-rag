"""Unit tests for the FastAPI HTTP API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evergreen_rag.api.routes import router
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

    application.state.embedding_service = mock_embedding
    application.state.vector_search = mock_search

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
