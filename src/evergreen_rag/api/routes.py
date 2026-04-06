"""HTTP endpoints for the RAG service."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
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


class GeneratedSearchRequest(BaseModel):
    """Request body for search with optional generation."""

    query: str
    limit: int = Field(default=10, ge=1, le=100)
    org_unit: int | None = None
    format: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    generate: bool = False


class GeneratedSearchResponse(BaseModel):
    """Search response with optional generated text."""

    query: str
    results: list[SearchResult]
    total: int
    model: str
    generated_text: str | None = None


class RecommendRequest(BaseModel):
    """Request body for reading recommendations."""

    query: str
    results: list[SearchResult]


class RecommendResponse(BaseModel):
    """Response with reading recommendations."""

    query: str
    recommendations: str | None = None


class RefineRequest(BaseModel):
    """Request body for search refinement suggestions."""

    query: str
    results: list[SearchResult]


class RefineResponse(BaseModel):
    """Response with refined query suggestions."""

    query: str
    suggestions: list[str]


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


@router.post("/search", response_model=GeneratedSearchResponse)
async def search(
    body: GeneratedSearchRequest, request: Request,
) -> GeneratedSearchResponse:
    """Perform semantic search against the catalog.

    Embeds the query text, then searches for similar records.
    If ``generate=true`` and a generation service is available,
    a natural language summary is appended to the response.
    """
    embedding_service = request.app.state.embedding_service
    vector_search = request.app.state.vector_search

    search_query = SearchQuery(
        query=body.query,
        limit=body.limit,
        org_unit=body.org_unit,
        format=body.format,
        min_similarity=body.min_similarity,
    )

    try:
        query_embedding = embedding_service.embed_text(body.query)
    except Exception as exc:
        logger.exception("Failed to embed query")
        raise HTTPException(
            status_code=503, detail=f"Embedding service error: {exc}"
        ) from exc

    try:
        response = vector_search.similarity_search(
            query_embedding, search_query
        )
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(
            status_code=503, detail=f"Search error: {exc}"
        ) from exc

    generated_text = None
    strong_results = [
        r for r in response.results if r.similarity >= 0.55
    ]
    if body.generate and strong_results:
        gen_service = getattr(
            request.app.state, "generation_service", None
        )
        if gen_service is not None:
            try:
                generated_text = gen_service.summarize(
                    body.query, strong_results
                )
            except Exception:
                logger.debug(
                    "Generation failed, returning results without summary",
                    exc_info=True,
                )

    return GeneratedSearchResponse(
        query=response.query,
        results=response.results,
        total=response.total,
        model=response.model,
        generated_text=generated_text,
    )


@router.post("/search/stream")
async def search_stream(
    body: GeneratedSearchRequest, request: Request,
) -> StreamingResponse:
    """Stream search results followed by token-by-token generation via SSE."""
    embedding_service = request.app.state.embedding_service
    vector_search = request.app.state.vector_search

    try:
        query_embedding = embedding_service.embed_text(body.query)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Embedding service error: {exc}"
        ) from exc

    try:
        search_query = SearchQuery(
            query=body.query,
            limit=body.limit,
            min_similarity=body.min_similarity,
        )
        response = vector_search.similarity_search(
            query_embedding, search_query
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Search error: {exc}"
        ) from exc

    def _build_record_lookup(results):
        """Build a dict mapping record_id to title/author for link expansion."""
        lookup = {}
        for r in results:
            lines = r.chunk_text.split("\n")
            title = lines[0].removeprefix("Title: ").strip() if lines else "Untitled"
            author = ""
            for line in lines[1:]:
                if line.startswith("by "):
                    author = line[3:].strip()
                    break
            lookup[str(r.record_id)] = (title, author)
        return lookup

    def _expand_token_buffer(buf, record_lookup, flush_all=False):
        """Expand \u00abN\u00bb tokens in buffer, return (expanded_html, remaining_buffer)."""
        import re
        expanded = ""
        while True:
            m = re.search(r"\u00ab(\d+)\u00bb", buf)
            if m:
                expanded += _esc(buf[:m.start()])
                rid = m.group(1)
                if rid in record_lookup:
                    title, author = record_lookup[rid]
                    link = (
                        f'<a href="/eg/opac/record/{rid}" '
                        f'style="color:#1565c0;text-decoration:underline">'
                        f'{_esc(title)}</a>'
                    )
                    if author:
                        link += f' <span style="color:#888;font-size:12px">by {_esc(author)}</span>'
                    expanded += link
                else:
                    expanded += _esc(m.group(0))
                buf = buf[m.end():]
            else:
                break
        if flush_all:
            expanded += _esc(buf)
            buf = ""
        return expanded, buf

    def _esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def event_stream():
        # Send search results first
        results_data = {
            "query": response.query,
            "results": [
                {
                    "record_id": r.record_id,
                    "similarity": r.similarity,
                    "chunk_text": r.chunk_text,
                }
                for r in response.results
            ],
            "total": response.total,
            "model": response.model,
        }
        yield f"event: results\ndata: {json.dumps(results_data)}\n\n"

        # Stream generation tokens, expanding record refs server-side
        # Filter to strong matches for LLM context (keep all for client)
        strong_results = [
            r for r in response.results if r.similarity >= 0.55
        ]
        if body.generate and strong_results:
            gen_service = getattr(
                request.app.state, "generation_service", None
            )
            if gen_service is not None:
                record_lookup = _build_record_lookup(response.results)
                buf = ""
                try:
                    for token in gen_service.stream_generate(
                        "summarize", body.query, strong_results
                    ):
                        buf += token
                        # Only flush when we're not mid-token (no open «)
                        if "\u00ab" not in buf or "\u00bb" in buf:
                            expanded, buf = _expand_token_buffer(buf, record_lookup)
                            if expanded:
                                yield (
                                    f"event: token\n"
                                    f"data: {json.dumps({'html': expanded})}\n\n"
                                )
                    # Flush remaining buffer
                    if buf:
                        expanded, _ = _expand_token_buffer(buf, record_lookup, flush_all=True)
                        if expanded:
                            yield (
                                f"event: token\n"
                                f"data: {json.dumps({'html': expanded})}\n\n"
                            )
                    yield "event: done\ndata: {}\n\n"
                except Exception:
                    logger.debug(
                        "Stream generation failed", exc_info=True
                    )
                    yield "event: done\ndata: {}\n\n"
            else:
                yield "event: done\ndata: {}\n\n"
        else:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    body: RecommendRequest, request: Request,
) -> RecommendResponse:
    """Generate reading recommendations from search results.

    Degrades gracefully: returns null recommendations if the
    generation service is unavailable.
    """
    gen_service = getattr(
        request.app.state, "generation_service", None
    )
    if gen_service is None:
        return RecommendResponse(query=body.query, recommendations=None)

    try:
        text = gen_service.recommend(body.query, body.results)
    except Exception:
        logger.debug("Recommend generation failed", exc_info=True)
        text = None

    return RecommendResponse(query=body.query, recommendations=text)


@router.post("/refine", response_model=RefineResponse)
async def refine(
    body: RefineRequest, request: Request,
) -> RefineResponse:
    """Suggest refined or related search queries.

    Degrades gracefully: returns an empty list if the generation
    service is unavailable.
    """
    gen_service = getattr(
        request.app.state, "generation_service", None
    )
    if gen_service is None:
        return RefineResponse(query=body.query, suggestions=[])

    try:
        suggestions = gen_service.refine(body.query, body.results)
    except Exception:
        logger.debug("Refine generation failed", exc_info=True)
        suggestions = []

    return RefineResponse(query=body.query, suggestions=suggestions)


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
    """Trigger ingest for specified records or full re-ingest.

    Restricted to local/private network requests only.
    """
    from ipaddress import ip_address, ip_network

    client_ip = request.client.host if request.client else "0.0.0.0"
    allowed_nets = [
        ip_network("127.0.0.0/8"),
        ip_network("10.0.0.0/8"),
        ip_network("192.168.0.0/16"),
    ]
    if not any(ip_address(client_ip) in net for net in allowed_nets):
        raise HTTPException(status_code=403, detail="Forbidden")

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

    generation_ok = False
    gen_service = getattr(request.app.state, "generation_service", None)
    if gen_service is not None:
        try:
            generation_ok = gen_service.health_check()
        except Exception:
            logger.debug("Generation health check failed", exc_info=True)

    checks = {
        "embedding_service": embedding_ok,
        "database": db_ok,
        "generation_service": generation_ok,
    }

    # Generation is optional — only core services determine degraded
    core_ok = embedding_ok and db_ok
    status = "ok" if core_ok else "degraded"

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
