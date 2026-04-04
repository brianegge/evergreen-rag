"""Unit tests for vector similarity search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from evergreen_rag.models.search import SearchQuery, SearchResponse, SearchResult
from evergreen_rag.search.vector_search import VectorSearch

FAKE_EMBEDDING = [0.1] * 768


class TestBuildSearchQuery:
    def test_basic_query(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        query = SearchQuery(query="test query", limit=5)
        sql, params = vs._build_search_query(FAKE_EMBEDDING, query)
        assert "rag.biblio_embedding" in sql
        assert "ORDER BY" in sql
        assert params["limit"] == 5

    def test_min_similarity_filter(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        query = SearchQuery(query="test", limit=10, min_similarity=0.5)
        sql, params = vs._build_search_query(FAKE_EMBEDDING, query)
        assert "WHERE" in sql
        assert params["min_similarity"] == 0.5

    def test_no_min_similarity(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        query = SearchQuery(query="test", limit=10, min_similarity=0.0)
        sql, params = vs._build_search_query(FAKE_EMBEDDING, query)
        assert "min_similarity" not in params

    def test_embedding_param_is_string(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        query = SearchQuery(query="test", limit=5)
        sql, params = vs._build_search_query(FAKE_EMBEDDING, query)
        assert params["embedding"] == str(FAKE_EMBEDDING)

    def test_cosine_distance_operator(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        query = SearchQuery(query="test", limit=5)
        sql, _params = vs._build_search_query(FAKE_EMBEDDING, query)
        assert "<=>" in sql


class TestSimilaritySearch:
    def test_search_returns_response(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"record": 1001, "similarity": 0.95, "chunk_text": "Hello world"},
            {"record": 1002, "similarity": 0.85, "chunk_text": "Another text"},
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            query = SearchQuery(query="test query", limit=10)
            response = vs.similarity_search(FAKE_EMBEDDING, query)

        assert isinstance(response, SearchResponse)
        assert response.total == 2
        assert response.query == "test query"
        assert len(response.results) == 2

    def test_search_empty_results(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            query = SearchQuery(query="no results", limit=10)
            response = vs.similarity_search(FAKE_EMBEDDING, query)

        assert response.total == 0
        assert response.results == []

    def test_search_result_mapping(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"record": 42, "similarity": 0.77, "chunk_text": "Mapped text"},
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            query = SearchQuery(query="map test", limit=5)
            response = vs.similarity_search(FAKE_EMBEDDING, query)

        result = response.results[0]
        assert result.record_id == 42
        assert result.similarity == 0.77
        assert result.chunk_text == "Mapped text"

    def test_search_result_types(self):
        result = SearchResult(record_id=1001, similarity=0.95, chunk_text="text")
        assert result.record_id == 1001
        assert result.similarity == 0.95
        assert result.chunk_text == "text"


class TestStoreEmbedding:
    def test_store_embedding_executes_insert(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            vs.store_embedding(
                record_id=100,
                embedding=FAKE_EMBEDDING,
                chunk_text="Test chunk",
                model_name="nomic-embed-text",
                chunk_index=0,
            )

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "ON CONFLICT (record, chunk_index, model_name)" in sql
        assert "chunk_index" in sql
        assert "created_at" in sql
        assert "embedded_at" not in sql
        assert params["record"] == 100
        assert params["chunk_index"] == 0
        assert params["chunk_text"] == "Test chunk"
        assert params["model_name"] == "nomic-embed-text"
        mock_conn.commit.assert_called_once()

    def test_store_embedding_default_chunk_index(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            vs.store_embedding(
                record_id=200,
                embedding=FAKE_EMBEDDING,
                chunk_text="Default chunk index",
            )

        params = mock_cursor.execute.call_args[0][1]
        assert params["chunk_index"] == 0

    def test_store_embedding_custom_chunk_index(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            vs.store_embedding(
                record_id=300,
                embedding=FAKE_EMBEDDING,
                chunk_text="Second chunk",
                chunk_index=1,
            )

        params = mock_cursor.execute.call_args[0][1]
        assert params["chunk_index"] == 1


class TestHasEmbedding:
    def test_has_embedding_true(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            assert vs.has_embedding(42) is True

    def test_has_embedding_false(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            assert vs.has_embedding(999) is False

    def test_has_embedding_custom_model(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            vs.has_embedding(42, model_name="custom-model")

        params = mock_cursor.execute.call_args[0][1]
        assert params["model_name"] == "custom-model"


class TestGetStats:
    def test_get_stats_with_data(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "total_embeddings": 100,
                "unique_records": 50,
                "last_embedded_at": "2026-01-01 00:00:00+00",
                "model_name": "nomic-embed-text",
            }
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            stats = vs.get_stats()

        assert stats["total_embeddings"] == 100
        assert stats["unique_records"] == 50
        assert stats["model_name"] == "nomic-embed-text"
        assert stats["last_embedded_at"] is not None

        # Verify SQL references created_at column (not embedded_at as source column)
        sql = mock_cursor.execute.call_args[0][0]
        assert "MAX(created_at)" in sql

    def test_get_stats_empty(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            stats = vs.get_stats()

        assert stats["total_embeddings"] == 0
        assert stats["unique_records"] == 0
        assert stats["last_embedded_at"] is None
        assert stats["model_name"] is None

    def test_get_stats_null_timestamp(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "total_embeddings": 5,
                "unique_records": 3,
                "last_embedded_at": None,
                "model_name": "nomic-embed-text",
            }
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            stats = vs.get_stats()

        assert stats["last_embedded_at"] is None


class TestHealthCheck:
    def test_health_check_success(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            assert vs.health_check() is True

    def test_health_check_no_pgvector(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            assert vs.health_check() is False

    def test_health_check_connection_error(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")

        with patch.object(vs, "_get_conn") as mock_get:
            mock_get.return_value.__enter__ = MagicMock(
                side_effect=Exception("Connection refused")
            )
            mock_get.return_value.__exit__ = MagicMock(return_value=False)

            assert vs.health_check() is False


class TestConnectionManagement:
    def test_open_creates_pool(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        with patch("evergreen_rag.search.vector_search.ConnectionPool") as mock_pool:
            vs.open()
            mock_pool.assert_called_once_with(
                vs.db_url,
                min_size=1,
                max_size=vs.pool_size,
                timeout=vs.pool_timeout,
            )

    def test_open_idempotent(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        vs._pool = MagicMock()
        with patch("evergreen_rag.search.vector_search.ConnectionPool") as mock_pool:
            vs.open()
            mock_pool.assert_not_called()

    def test_close_pool(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        mock_pool = MagicMock()
        vs._pool = mock_pool
        vs.close()
        mock_pool.close.assert_called_once()
        assert vs._pool is None

    def test_close_noop_when_not_open(self):
        vs = VectorSearch(db_url="postgresql://test:test@localhost/test")
        vs.close()  # should not raise


class TestConfiguration:
    def test_default_config(self):
        vs = VectorSearch()
        assert "postgresql" in vs.db_url
        assert vs.pool_size == 10

    def test_custom_config(self):
        vs = VectorSearch(
            db_url="postgresql://custom:pass@host/db",
            pool_size=20,
            pool_timeout=60.0,
        )
        assert vs.db_url == "postgresql://custom:pass@host/db"
        assert vs.pool_size == 20
        assert vs.pool_timeout == 60.0

    def test_env_var_config(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://envhost/envdb")
        vs = VectorSearch()
        assert vs.db_url == "postgresql://envhost/envdb"


class TestSearchQueryModel:
    def test_defaults(self):
        q = SearchQuery(query="books about cats")
        assert q.limit == 10
        assert q.min_similarity == 0.0
        assert q.org_unit is None
        assert q.format is None

    def test_custom_values(self):
        q = SearchQuery(query="test", limit=5, min_similarity=0.7, org_unit=1)
        assert q.limit == 5
        assert q.min_similarity == 0.7
        assert q.org_unit == 1


class TestSearchResponseModel:
    def test_response_structure(self):
        resp = SearchResponse(
            query="test",
            results=[
                SearchResult(record_id=1, similarity=0.9, chunk_text="text"),
            ],
            total=1,
            model="nomic-embed-text",
        )
        assert resp.query == "test"
        assert resp.total == 1
        assert resp.model == "nomic-embed-text"
