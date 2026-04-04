"""Models for search queries and results."""

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A semantic search query against the catalog."""

    query: str
    limit: int = Field(default=10, ge=1, le=100)
    org_unit: int | None = None
    format: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """A single search result with similarity score."""

    record_id: int
    similarity: float
    chunk_text: str


class SearchResponse(BaseModel):
    """Response containing ranked search results."""

    query: str
    results: list[SearchResult]
    total: int
    model: str
