"""HTTP endpoints for the RAG service."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from evergreen_rag.models.search import SearchQuery, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Request/response models for non-search endpoints
# ------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for the ingest endpoint."""

    record_ids: list[int] | None = None
    all: bool = False


class IngestResponse(BaseModel):
    """Response from the ingest endpoint."""

    status: str
    message: str
    total: int = 0
    embedded: int = 0
    failed: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    checks: dict[str, bool]


class StatsResponse(BaseModel):
    """Statistics response."""

    total_embeddings: int
    unique_records: int
    last_embedded_at: str | None
    model_name: str | None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery, request: Request) -> SearchResponse:
    """Perform semantic search against the catalog.

    Embeds the query text, then searches for similar records.
    """
    embedding_service = request.app.state.embedding_service
    vector_search = request.app.state.vector_search

    try:
        query_embedding = embedding_service.embed_text(query.query)
    except Exception as exc:
        logger.exception("Failed to embed query")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {exc}") from exc

    try:
        response = vector_search.similarity_search(query_embedding, query)
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=503, detail=f"Search error: {exc}") from exc

    return response


@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    """Trigger ingest for specified records or full re-ingest."""
    from evergreen_rag.ingest.pipeline import IngestPipeline

    embedding_service = request.app.state.embedding_service

    try:
        pipeline = IngestPipeline(embedding_service=embedding_service)
        stats = pipeline.run(record_ids=body.record_ids, full=body.all)
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}") from exc

    return IngestResponse(
        status="completed",
        message=f"Ingested {stats.embedded} records",
        total=stats.total,
        embedded=stats.embedded,
        failed=stats.failed,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Check service health: database, embedding service, pgvector."""
    embedding_ok = False
    db_ok = False

    try:
        embedding_ok = request.app.state.embedding_service.health_check()
    except Exception:
        logger.debug("Embedding health check failed", exc_info=True)

    try:
        db_ok = request.app.state.vector_search.health_check()
    except Exception:
        logger.debug("DB health check failed", exc_info=True)

    checks = {
        "embedding_service": embedding_ok,
        "database": db_ok,
    }

    status = "ok" if all(checks.values()) else "degraded"

    return HealthResponse(status=status, checks=checks)


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> Any:
    """Return statistics about embedded records."""
    try:
        data = request.app.state.vector_search.get_stats()
    except Exception as exc:
        logger.exception("Failed to get stats")
        raise HTTPException(status_code=503, detail=f"Stats error: {exc}") from exc

    return StatsResponse(**data)
