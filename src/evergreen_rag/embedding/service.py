"""Embedding service client wrapping Ollama for vector generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
import ollama as ollama_pkg

from evergreen_rag.models.embedding import EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Client wrapper for Ollama's embedding API.

    Prefers the ``ollama`` Python package for simplicity and falls back
    to raw ``httpx`` requests when necessary.

    Supports per-language model selection via *model_map*.  When a map is
    configured, ``embed_text_with_language`` picks the model matching the
    given MARC language code, falling back to the ``"*"`` wildcard entry
    or the default single model.
    """

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        model_map: dict[str, str] | None = None,
    ) -> None:
        self.ollama_url = ollama_url or os.environ.get(
            "OLLAMA_URL", "http://localhost:11434"
        )
        self.model = model or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
        self.timeout = timeout
        self.max_retries = max_retries
        self._dimensions: int | None = None
        self.model_map: dict[str, str] = model_map or _load_model_map_from_env()

        # Ollama Python client
        self._client = ollama_pkg.Client(host=self.ollama_url, timeout=httpx.Timeout(self.timeout))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string and return the embedding vector."""
        response = self.embed_batch([text])
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> EmbeddingResponse:
        """Embed multiple texts and return an ``EmbeddingResponse``."""
        request = EmbeddingRequest(texts=texts, model=self.model)
        return self._embed_via_ollama(request)

    def embed_text_with_language(self, text: str, language: str) -> list[float]:
        """Embed *text* using the model configured for *language*.

        If no ``model_map`` is configured or the language has no mapping,
        falls back to the default model (backward-compatible).
        """
        model = self._resolve_model(language)
        request = EmbeddingRequest(texts=[text], model=model)
        response = self._embed_via_ollama(request)
        return response.embeddings[0]

    def embed_batch_with_language(
        self, texts: list[str], language: str
    ) -> EmbeddingResponse:
        """Embed a batch of texts using the model for *language*."""
        model = self._resolve_model(language)
        request = EmbeddingRequest(texts=texts, model=model)
        return self._embed_via_ollama(request)

    def embed_request(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Accept an ``EmbeddingRequest`` and return an ``EmbeddingResponse``."""
        return self._embed_via_ollama(request)

    def health_check(self) -> bool:
        """Return *True* if Ollama is reachable and the model is available."""
        try:
            self._client.list()
            return True
        except Exception:
            logger.debug("Ollama health check failed, trying httpx fallback")
            return self._health_check_httpx()

    def pull_model(self) -> None:
        """Pull the configured model if not already present."""
        try:
            self._client.pull(self.model)
            logger.info("Pulled model %s", self.model)
        except Exception:
            logger.exception("Failed to pull model %s", self.model)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_via_ollama(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using the ``ollama`` Python package."""
        try:
            return self._do_ollama_embed(request)
        except Exception:
            logger.debug(
                "ollama package embed failed, falling back to httpx",
                exc_info=True,
            )
            return self._embed_via_httpx(request)

    def _do_ollama_embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Call Ollama embed endpoint via the Python package."""
        result: Any = self._client.embed(model=request.model, input=request.texts)

        embeddings: list[list[float]]
        if isinstance(result, dict):
            embeddings = result.get("embeddings", [])
        else:
            embeddings = getattr(result, "embeddings", [])

        if not embeddings:
            raise ValueError("No embeddings returned from Ollama")

        dimensions = len(embeddings[0])
        if self._dimensions is None:
            self._dimensions = dimensions
            logger.info(
                "Embedding dimensions: %d (model=%s)", dimensions, request.model
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model,
            dimensions=dimensions,
        )

    def _embed_via_httpx(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Fallback: call Ollama embed endpoint via raw HTTP."""
        url = f"{self.ollama_url}/api/embed"
        payload = {"model": request.model, "input": request.texts}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()

        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("No embeddings in httpx response")

        dimensions = len(embeddings[0])
        if self._dimensions is None:
            self._dimensions = dimensions
            logger.info(
                "Embedding dimensions (httpx): %d (model=%s)",
                dimensions,
                request.model,
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model,
            dimensions=dimensions,
        )

    def _resolve_model(self, language: str) -> str:
        """Pick the embedding model for the given MARC language code."""
        if not self.model_map:
            return self.model
        lang = language.lower()
        if lang in self.model_map:
            return self.model_map[lang]
        if "*" in self.model_map:
            return self.model_map["*"]
        return self.model

    def _health_check_httpx(self) -> bool:
        """Check Ollama health via raw HTTP GET."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(self.ollama_url)
                return resp.status_code == 200
        except Exception:
            return False


def _load_model_map_from_env() -> dict[str, str]:
    """Load a language-to-model mapping from ``EMBEDDING_MODEL_MAP`` env var.

    Expected format is a JSON object, e.g.::

        EMBEDDING_MODEL_MAP='{"eng": "nomic-embed-text", "spa": "multi-e5"}'

    Returns an empty dict if the env var is unset or invalid.
    """
    raw = os.environ.get("EMBEDDING_MODEL_MAP", "")
    if not raw:
        return {}
    try:
        mapping = json.loads(raw)
        if isinstance(mapping, dict):
            return {k: str(v) for k, v in mapping.items()}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid EMBEDDING_MODEL_MAP env var, ignoring: %s", raw)
    return {}
