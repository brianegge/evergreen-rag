"""Unit tests for the generation service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evergreen_rag.generation.service import GenerationService
from evergreen_rag.models.search import SearchResult

FAKE_RESULTS = [
    SearchResult(
        record_id=1, similarity=0.95,
        chunk_text="The Great Gatsby by F. Scott Fitzgerald",
    ),
    SearchResult(
        record_id=2, similarity=0.87,
        chunk_text="Tender Is the Night by F. Scott Fitzgerald",
    ),
]


@pytest.fixture
def mock_ollama_client():
    """Patch the ollama.Client used by GenerationService."""
    with patch("evergreen_rag.generation.service.ollama_pkg") as mock_pkg:
        mock_client = MagicMock()
        mock_pkg.Client.return_value = mock_client
        yield mock_client


@pytest.fixture
def service(mock_ollama_client):
    """Create a GenerationService with a mocked Ollama client."""
    return GenerationService(
        ollama_url="http://test:11434",
        model="llama3.2",
    )


class TestSummarize:
    def test_returns_summary(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "These results include two novels by Fitzgerald."},
        }
        result = service.summarize("fitzgerald novels", FAKE_RESULTS)
        assert result == "These results include two novels by Fitzgerald."
        mock_ollama_client.chat.assert_called_once()

    def test_includes_query_in_prompt(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Summary here."},
        }
        service.summarize("science fiction classics", FAKE_RESULTS)
        call_args = mock_ollama_client.chat.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "science fiction classics" in prompt

    def test_includes_results_in_prompt(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Summary."},
        }
        service.summarize("query", FAKE_RESULTS)
        call_args = mock_ollama_client.chat.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "The Great Gatsby" in prompt
        assert "record #1" in prompt
        assert "0.95" in prompt


class TestRecommend:
    def test_returns_recommendations(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "I recommend starting with The Great Gatsby."},
        }
        result = service.recommend("fitzgerald", FAKE_RESULTS)
        assert result == "I recommend starting with The Great Gatsby."

    def test_uses_recommend_template(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Recommendations."},
        }
        service.recommend("query", FAKE_RESULTS)
        call_args = mock_ollama_client.chat.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "reading recommendations" in prompt


class TestRefine:
    def test_returns_list_of_queries(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {
                "content": "F. Scott Fitzgerald biography\n"
                "Jazz Age literature\n1920s American novels",
            },
        }
        result = service.refine("fitzgerald", FAKE_RESULTS)
        assert isinstance(result, list)
        assert len(result) == 3
        assert "F. Scott Fitzgerald biography" in result

    def test_filters_empty_lines(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "query one\n\nquery two\n\n"},
        }
        result = service.refine("test", FAKE_RESULTS)
        assert len(result) == 2

    def test_returns_empty_on_failure(self, service, mock_ollama_client):
        mock_ollama_client.chat.side_effect = Exception("down")
        with patch("evergreen_rag.generation.service.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("also down")
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()

            result = service.refine("test", FAKE_RESULTS)
            assert result == []


class TestHealthCheck:
    def test_healthy(self, service, mock_ollama_client):
        mock_ollama_client.list.return_value = {"models": []}
        assert service.health_check() is True

    def test_unhealthy_both_fail(self, service, mock_ollama_client):
        mock_ollama_client.list.side_effect = ConnectionError("down")
        with patch("evergreen_rag.generation.service.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = ConnectionError("also down")
            assert service.health_check() is False


class TestErrorHandling:
    def test_ollama_failure_falls_back_to_httpx(self, service, mock_ollama_client):
        mock_ollama_client.chat.side_effect = Exception("ollama error")

        with patch("evergreen_rag.generation.service.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "message": {"content": "httpx response"},
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()

            result = service.summarize("test", FAKE_RESULTS)
            assert result == "httpx response"

    def test_both_backends_fail_returns_none(self, service, mock_ollama_client):
        mock_ollama_client.chat.side_effect = Exception("ollama error")

        with patch("evergreen_rag.generation.service.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("httpx error")
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()

            result = service.summarize("test", FAKE_RESULTS)
            assert result is None

    def test_empty_response_falls_back(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {"message": {"content": ""}}

        with patch("evergreen_rag.generation.service.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "message": {"content": "fallback response"},
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.Timeout = MagicMock()

            result = service.summarize("test", FAKE_RESULTS)
            assert result == "fallback response"


class TestPromptTemplates:
    def test_summarize_template_exists(self):
        assert "summarize" in GenerationService.TEMPLATES

    def test_recommend_template_exists(self):
        assert "recommend" in GenerationService.TEMPLATES

    def test_refine_template_exists(self):
        assert "refine" in GenerationService.TEMPLATES

    def test_templates_have_placeholders(self):
        for name, template in GenerationService.TEMPLATES.items():
            assert "{query}" in template, f"{name} template missing {{query}}"
            assert "{results}" in template, f"{name} template missing {{results}}"

    def test_build_prompt(self, service):
        prompt = service._build_prompt("summarize", "test query", FAKE_RESULTS)
        assert "test query" in prompt
        assert "The Great Gatsby" in prompt
        assert "Result 1" in prompt
        assert "Result 2" in prompt

    def test_format_results(self, service):
        formatted = service._format_results(FAKE_RESULTS)
        assert "record #1" in formatted
        assert "record #2" in formatted
        assert "0.95" in formatted
        assert "0.87" in formatted


class TestConfiguration:
    def test_default_config(self):
        with patch("evergreen_rag.generation.service.ollama_pkg"):
            svc = GenerationService()
            assert svc.model == "llama3.2"
            assert svc.ollama_url == "http://localhost:11434"
            assert svc.temperature == 0.7
            assert svc.max_tokens == 1024

    def test_custom_config(self):
        with patch("evergreen_rag.generation.service.ollama_pkg"):
            svc = GenerationService(
                ollama_url="http://custom:1234",
                model="mistral",
                temperature=0.3,
                max_tokens=2048,
                timeout=60.0,
            )
            assert svc.ollama_url == "http://custom:1234"
            assert svc.model == "mistral"
            assert svc.temperature == 0.3
            assert svc.max_tokens == 2048
            assert svc.timeout == 60.0

    def test_env_var_config(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://envhost:9999")
        monkeypatch.setenv("GENERATION_MODEL", "phi3")
        with patch("evergreen_rag.generation.service.ollama_pkg"):
            svc = GenerationService()
            assert svc.ollama_url == "http://envhost:9999"
            assert svc.model == "phi3"


class TestOllamaResponseFormats:
    def test_object_response(self, service, mock_ollama_client):
        """Ollama can return an object with message attribute."""
        message_obj = MagicMock()
        message_obj.content = "Object-style response"
        result_obj = MagicMock(spec=[])  # no dict behavior
        result_obj.message = message_obj
        mock_ollama_client.chat.return_value = result_obj

        result = service.summarize("query", FAKE_RESULTS)
        assert result == "Object-style response"

    def test_passes_temperature_and_max_tokens(self, service, mock_ollama_client):
        mock_ollama_client.chat.return_value = {
            "message": {"content": "response"},
        }
        service.summarize("query", FAKE_RESULTS)
        call_args = mock_ollama_client.chat.call_args
        options = call_args[1]["options"]
        assert options["temperature"] == 0.7
        assert options["num_predict"] == 1024


class TestHealthCheckFallback:
    def test_ollama_fails_httpx_succeeds(self, service, mock_ollama_client):
        mock_ollama_client.list.side_effect = ConnectionError("down")
        with patch("evergreen_rag.generation.service.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_httpx.Client.return_value.__enter__ = MagicMock(
                return_value=mock_client
            )
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            assert service.health_check() is True
