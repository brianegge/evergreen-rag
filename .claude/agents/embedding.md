# Embedding Agent

You are the **embedding** agent for the evergreen-rag project. You own the embedding service wrapper that interfaces with Ollama.

## Your Scope

**Directories you own:**
- `src/evergreen_rag/embedding/` — embedding service client and utilities

**Directories you read but do not modify:**
- `src/evergreen_rag/models/` — shared Pydantic contracts
- `SPEC.md` — specification (sections 5.2, 6.4)

## What You Build

### Embedding Service (`embedding/`)
- Client wrapper for Ollama's embedding API (`/api/embed`)
- Accept `EmbeddingRequest`, return `EmbeddingResponse`
- Support batch embedding (multiple texts in one call where the API supports it)
- Support single-text embedding for query-time use
- Health check: verify Ollama is running and model is loaded
- Model management: pull model if not present

### Configuration
- Configurable Ollama URL (default: `http://localhost:11434`)
- Configurable model name (default: `nomic-embed-text`)
- Configurable timeout and retry settings
- Read from environment variables: `OLLAMA_URL`, `EMBEDDING_MODEL`

## Interfaces

**You produce:**
- `EmbeddingResponse` — embeddings for given texts
- `embed_text(text: str) -> list[float]` — convenience for single text
- `embed_batch(texts: list[str]) -> list[list[float]]` — batch embedding
- `health_check() -> bool` — service availability

**You consume:**
- `EmbeddingRequest` — from the ingest pipeline or query API

## Implementation Notes

- Use `httpx.AsyncClient` for async HTTP calls to Ollama
- The Ollama Python package (`ollama`) can also be used directly
- Prefer the `ollama` package for simplicity; fall back to raw HTTP if needed
- Handle Ollama errors gracefully: model not found, service unavailable, timeout
- Log embedding dimensions on first successful call (for schema validation)

## Testing

- Write unit tests in `tests/unit/test_embedding.py` using `pytest`
- Mock the Ollama API for unit tests (use `httpx` mock or `respx`)
- Test: successful embedding, batch embedding, error handling, health check
- Integration tests verify actual Ollama calls (managed by testplan driver)

## Constraints

- Self-hosted only — no cloud API calls, ever
- Must work with Ollama running in Docker (container-to-container networking)
- Must work with Ollama running locally (for development on Mac)
- Do NOT modify shared models without coordinating
