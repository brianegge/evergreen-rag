"""HTTP endpoints for the RAG service."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from evergreen_rag.models.search import SearchQuery, SearchResponse, SearchResult

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


class MergedSearchRequest(BaseModel):
    """Request body for merged keyword + semantic search."""

    query: str | None = None
    keyword_results: list[int] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    weights: dict[str, float] | None = None


# ------------------------------------------------------------------
# RRF helper
# ------------------------------------------------------------------


def reciprocal_rank_fusion(
    *rankings: list[int],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings:
        Each ranking is a list of record IDs in rank order.
    k:
        RRF constant (default 60).
    weights:
        Optional per-ranking weight multiplier. Defaults to equal weight.

    Returns
    -------
    List of (record_id, score) tuples sorted by descending score.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, record_id in enumerate(ranking, start=1):
            scores[record_id] += weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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


@router.post("/search/merged", response_model=SearchResponse)
async def search_merged(body: MergedSearchRequest, request: Request) -> SearchResponse:
    """Merge keyword and semantic search results using Reciprocal Rank Fusion.

    Accepts keyword result IDs from Evergreen's native search and an optional
    query string for semantic search.  Results are merged via RRF.
    """
    if not body.query and not body.keyword_results:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'query' or 'keyword_results' must be provided",
        )

    semantic_ids: list[int] = []
    semantic_chunks: dict[int, str] = {}
    semantic_similarities: dict[int, float] = {}

    # Run semantic search if a query is provided
    if body.query:
        embedding_service = request.app.state.embedding_service
        vector_search = request.app.state.vector_search

        try:
            query_embedding = embedding_service.embed_text(body.query)
        except Exception as exc:
            logger.exception("Failed to embed query")
            raise HTTPException(
                status_code=503, detail=f"Embedding service error: {exc}"
            ) from exc

        try:
            search_query = SearchQuery(query=body.query, limit=body.limit)
            sem_response = vector_search.similarity_search(query_embedding, search_query)
        except Exception as exc:
            logger.exception("Semantic search failed")
            raise HTTPException(
                status_code=503, detail=f"Search error: {exc}"
            ) from exc

        for result in sem_response.results:
            semantic_ids.append(result.record_id)
            semantic_chunks[result.record_id] = result.chunk_text
            semantic_similarities[result.record_id] = result.similarity

    # Build rankings for RRF
    rankings: list[list[int]] = []
    weight_list: list[float] = []

    semantic_weight = (body.weights or {}).get("semantic", 1.0)
    keyword_weight = (body.weights or {}).get("keyword", 1.0)

    if semantic_ids:
        rankings.append(semantic_ids)
        weight_list.append(semantic_weight)
    if body.keyword_results:
        rankings.append(body.keyword_results)
        weight_list.append(keyword_weight)

    if not rankings:
        return SearchResponse(
            query=body.query or "",
            results=[],
            total=0,
            model="nomic-embed-text",
        )

    merged = reciprocal_rank_fusion(*rankings, weights=weight_list)
    merged = merged[: body.limit]

    results = [
        SearchResult(
            record_id=record_id,
            similarity=semantic_similarities.get(record_id, 0.0),
            chunk_text=semantic_chunks.get(record_id, ""),
        )
        for record_id, _score in merged
    ]

    return SearchResponse(
        query=body.query or "",
        results=results,
        total=len(results),
        model="nomic-embed-text",
    )


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
