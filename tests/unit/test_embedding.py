"""Unit tests for the embedding service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.models.embedding import EmbeddingRequest, EmbeddingResponse

FAKE_EMBEDDING = [0.1] * 768


@pytest.fixture
def mock_ollama_client():
    """Patch the ollama.Client used by EmbeddingService."""
    with patch("evergreen_rag.embedding.service.ollama_pkg") as mock_pkg:
        mock_client = MagicMock()
        mock_pkg.Client.return_value = mock_client
        yield mock_client


@pytest.fixture
def service(mock_ollama_client):
    """Create an EmbeddingService with a mocked Ollama client."""
    return EmbeddingService(ollama_url="http://test:11434", model="nomic-embed-text")


class TestEmbedText:
    def test_single_text(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        result = service.embed_text("hello world")
        assert isinstance(result, list)
        assert len(result) == 768
        assert result == FAKE_EMBEDDING

    def test_empty_text(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        result = service.embed_text("")
        assert isinstance(result, list)
        assert len(result) == 768


class TestEmbedBatch:
    def test_batch_multiple_texts(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING, [0.2] * 768],
        }
        response = service.embed_batch(["text one", "text two"])
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 2
        assert response.model == "nomic-embed-text"
        assert response.dimensions == 768

    def test_batch_single_text(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        response = service.embed_batch(["only one"])
        assert len(response.embeddings) == 1

    def test_embed_request(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        request = EmbeddingRequest(texts=["test text"], model="nomic-embed-text")
        response = service.embed_request(request)
        assert isinstance(response, EmbeddingResponse)
        assert response.model == "nomic-embed-text"


class TestHealthCheck:
    def test_healthy(self, service, mock_ollama_client):
        mock_ollama_client.list.return_value = {"models": []}
        assert service.health_check() is True

    def test_unhealthy_both_fail(self, service, mock_ollama_client):
        mock_ollama_client.list.side_effect = ConnectionError("down")
        with patch("evergreen_rag.embedding.service.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = ConnectionError("also down")
            assert service.health_check() is False


class TestErrorHandling:
    def test_ollama_failure_falls_back_to_httpx(self, service, mock_ollama_client):
        """When ollama package fails, httpx fallback is attempted."""
        mock_ollama_client.embed.side_effect = Exception("ollama error")

        with patch("evergreen_rag.embedding.service.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"embeddings": [FAKE_EMBEDDING]}
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()

            result = service.embed_text("test")
            assert result == FAKE_EMBEDDING

    def test_both_backends_fail(self, service, mock_ollama_client):
        mock_ollama_client.embed.side_effect = Exception("ollama error")

        with patch("evergreen_rag.embedding.service.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("httpx error")
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()

            with pytest.raises(Exception):
                service.embed_text("test")


class TestConfiguration:
    def test_default_config(self):
        with patch("evergreen_rag.embedding.service.ollama_pkg"):
            svc = EmbeddingService()
            assert svc.model == "nomic-embed-text"
            assert svc.ollama_url == "http://localhost:11434"

    def test_custom_config(self):
        with patch("evergreen_rag.embedding.service.ollama_pkg"):
            svc = EmbeddingService(
                ollama_url="http://custom:1234",
                model="custom-model",
                timeout=120.0,
            )
            assert svc.ollama_url == "http://custom:1234"
            assert svc.model == "custom-model"
            assert svc.timeout == 120.0

    def test_env_var_config(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://envhost:9999")
        monkeypatch.setenv("EMBEDDING_MODEL", "env-model")
        with patch("evergreen_rag.embedding.service.ollama_pkg"):
            svc = EmbeddingService()
            assert svc.ollama_url == "http://envhost:9999"
            assert svc.model == "env-model"


class TestPullModel:
    def test_pull_success(self, service, mock_ollama_client):
        service.pull_model()
        mock_ollama_client.pull.assert_called_once_with("nomic-embed-text")

    def test_pull_failure(self, service, mock_ollama_client):
        mock_ollama_client.pull.side_effect = Exception("pull failed")
        with pytest.raises(Exception, match="pull failed"):
            service.pull_model()


class TestOllamaResponseFormats:
    def test_object_response(self, service, mock_ollama_client):
        """Ollama can return an object with an embeddings attribute."""
        result_obj = MagicMock()
        result_obj.embeddings = [FAKE_EMBEDDING]
        # Make isinstance(result, dict) return False
        mock_ollama_client.embed.return_value = result_obj
        response = service.embed_batch(["test"])
        assert response.embeddings == [FAKE_EMBEDDING]
        assert response.dimensions == 768

    def test_empty_embeddings_raises(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {"embeddings": []}
        with patch("evergreen_rag.embedding.service.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("httpx also fails")
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()
            with pytest.raises(Exception):
                service.embed_batch(["test"])


class TestHealthCheckFallback:
    def test_ollama_fails_httpx_succeeds(self, service, mock_ollama_client):
        mock_ollama_client.list.side_effect = ConnectionError("down")
        with patch("evergreen_rag.embedding.service.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            assert service.health_check() is True


class TestEmbedRequest:
    def test_custom_model_in_request(self, service, mock_ollama_client):
        mock_ollama_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        request = EmbeddingRequest(texts=["test"], model="custom-model")
        response = service.embed_request(request)
        assert response.model == "custom-model"
        mock_ollama_client.embed.assert_called_once_with(
            model="custom-model", input=["test"]
        )
