"""Generation service wrapping Ollama's chat API for producing natural language."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import ollama as ollama_pkg

from evergreen_rag.models.search import SearchResult

logger = logging.getLogger(__name__)


class GenerationService:
    """Client wrapper for Ollama's chat/generate API.

    Produces natural language summaries, recommendations, and refined queries
    from retrieved search results. Uses the ``ollama`` Python package with
    an ``httpx`` fallback, mirroring the embedding service pattern.
    """

    TEMPLATES: dict[str, str] = {
        "summarize": (
            "You are a friendly library assistant helping a patron. "
            "They searched for: \"{query}\"\n\n"
            "Here are the matching catalog records:\n\n{results}\n\n"
            "Write a brief, natural 2-3 sentence response. "
            "IMPORTANT: When you mention a catalog record, you MUST use the "
            "exact token ###N### where N is the numeric record ID from the results above. "
            "For example, if a result says 'record #247', write ###247### in your response. "
            "The display system will replace ###247### with the formatted linked title. "
            "Do NOT write out the actual title text yourself. "
            "Mention 2-3 of the most relevant records using their ###N### tokens. "
            "Keep it concise. Do NOT start with a greeting or filler phrase. "
            "Do NOT list every record. Do NOT repeat similarity scores."
        ),
        "recommend": (
            "You are a knowledgeable library reader's advisor. "
            "A patron searched for: \"{query}\"\n\n"
            "Here are the catalog records found:\n\n{results}\n\n"
            "Based on these results, provide personalized reading recommendations. "
            "Explain why each recommendation would interest the patron and suggest "
            "an order for reading if applicable."
        ),
        "refine": (
            "You are a search assistant for a library catalog. "
            "A patron searched for: \"{query}\"\n\n"
            "Here are the results found:\n\n{results}\n\n"
            "Suggest 3-5 refined or related search queries that could help the patron "
            "find additional relevant materials. Return ONLY the search queries, "
            "one per line, with no numbering or extra text."
        ),
    }

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.ollama_url = ollama_url or os.environ.get(
            "OLLAMA_URL", "http://localhost:11434"
        )
        self.model = model or os.environ.get("GENERATION_MODEL", "llama3.2")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Ollama Python client
        self._client = ollama_pkg.Client(
            host=self.ollama_url, timeout=httpx.Timeout(self.timeout)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, query: str, results: list[SearchResult]) -> str | None:
        """Generate a natural language summary of search results."""
        return self._generate("summarize", query, results)

    def recommend(self, query: str, results: list[SearchResult]) -> str | None:
        """Generate reading recommendations from search results."""
        return self._generate("recommend", query, results)

    def refine(self, query: str, results: list[SearchResult]) -> list[str]:
        """Suggest refined/related search queries."""
        response = self._generate("refine", query, results)
        if not response:
            return []
        return [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip()
        ]

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            models = self._client.list()
            # Check if our model is available
            model_names: list[str] = []
            if isinstance(models, dict):
                for m in models.get("models", []):
                    name = m.get("name", "") if isinstance(m, dict) else getattr(m, "name", "")
                    model_names.append(name)
            return True
        except Exception:
            logger.debug("Ollama health check failed, trying httpx fallback")
            return self._health_check_httpx()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_results(self, results: list[SearchResult]) -> str:
        """Format search results into a text block for prompt injection."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"Result {i} (record #{r.record_id}, similarity: {r.similarity:.2f}):\n"
                f"{r.chunk_text}"
            )
        return "\n\n".join(lines)

    def _build_prompt(self, mode: str, query: str, results: list[SearchResult]) -> str:
        """Build the full prompt from template, query, and results."""
        template = self.TEMPLATES[mode]
        return template.format(query=query, results=self._format_results(results))

    def _generate(self, mode: str, query: str, results: list[SearchResult]) -> str | None:
        """Generate text using Ollama chat API with httpx fallback."""
        prompt = self._build_prompt(mode, query, results)
        try:
            return self._chat_via_ollama(prompt)
        except Exception:
            logger.debug(
                "ollama package chat failed, falling back to httpx",
                exc_info=True,
            )
            try:
                return self._chat_via_httpx(prompt)
            except Exception:
                logger.exception("Both ollama and httpx chat failed")
                return None

    def _chat_via_ollama(self, prompt: str) -> str:
        """Call Ollama chat endpoint via the Python package."""
        result: Any = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        if isinstance(result, dict):
            message = result.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "")
        else:
            message = getattr(result, "message", None)
            content = getattr(message, "content", "") if message else ""

        if not content:
            raise ValueError("Empty response from Ollama chat")

        return content

    def stream_generate(
        self, mode: str, query: str, results: list[SearchResult]
    ):
        """Yield text chunks from Ollama streaming chat via httpx."""
        prompt = self._build_prompt(mode, query, results)
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": True,
        }
        import json as _json

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    content = (
                        chunk.get("message", {}).get("content", "")
                    )
                    if content:
                        yield content
                    if chunk.get("done"):
                        return

    def _chat_via_httpx(self, prompt: str) -> str:
        """Fallback: call Ollama chat endpoint via raw HTTP."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from httpx chat")

        return content

    def _health_check_httpx(self) -> bool:
        """Check Ollama health via raw HTTP GET."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(self.ollama_url)
                return resp.status_code == 200
        except Exception:
            return False
