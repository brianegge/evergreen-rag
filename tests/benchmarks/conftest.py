"""Shared fixtures for performance benchmarks."""

from __future__ import annotations

import os

import psycopg
import pytest

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.search.vector_search import VectorSearch

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://evergreen:evergreen@localhost:5432/evergreen"
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _check_postgres() -> bool:
    try:
        with psycopg.connect(DB_URL, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _check_ollama() -> bool:
    import httpx

    try:
        resp = httpx.get(OLLAMA_URL, timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _check_postgres(), reason="PostgreSQL not available"
)
requires_ollama = pytest.mark.skipif(
    not _check_ollama(), reason="Ollama not available"
)
requires_services = pytest.mark.skipif(
    not (_check_postgres() and _check_ollama()),
    reason="PostgreSQL and/or Ollama not available",
)


@pytest.fixture(scope="session")
def db_url() -> str:
    return DB_URL


@pytest.fixture(scope="session")
def db_conn(db_url: str) -> psycopg.Connection:  # type: ignore[type-arg]
    conn = psycopg.connect(db_url)
    yield conn  # type: ignore[misc]
    conn.close()


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    return EmbeddingService(ollama_url=OLLAMA_URL)


@pytest.fixture(scope="session")
def vector_search(db_url: str) -> VectorSearch:
    vs = VectorSearch(db_url=db_url)
    vs.open()
    yield vs  # type: ignore[misc]
    vs.close()
