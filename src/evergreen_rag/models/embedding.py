"""Models for the embedding service interface."""

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """Request to generate embeddings for one or more texts."""

    texts: list[str]
    model: str = "nomic-embed-text"


class EmbeddingResponse(BaseModel):
    """Response containing generated embeddings."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
