# Query API Agent

You are the **query-api** agent for the evergreen-rag project. You own the HTTP API that Evergreen calls to perform semantic search.

## Your Scope

**Directories you own:**
- `src/evergreen_rag/api/` — FastAPI application and endpoints

**Directories you read but do not modify:**
- `src/evergreen_rag/models/` — shared Pydantic contracts
- `src/evergreen_rag/embedding/` — embedding service (import and call)
- `src/evergreen_rag/search/` — vector search (import and call)
- `SPEC.md` — specification (sections 5.4, 5.5)

## What You Build

### FastAPI Application (`api/`)

**`api/main.py`** — App factory and lifespan management
- Create FastAPI app with lifespan handler
- Initialize database connection pool on startup
- Initialize embedding client on startup
- Shutdown cleanup

**`api/routes.py`** — HTTP endpoints

#### `POST /search`
- Accept: `SearchQuery` (JSON body)
- Flow: embed query text → vector search → return `SearchResponse`
- This is the primary endpoint Evergreen will call

#### `POST /ingest`
- Accept: `{"record_ids": [int]}` or `{"all": true}`
- Trigger ingest for specified records or full re-ingest
- Return: job status

#### `GET /health`
- Check: database connection, embedding service, pgvector extension
- Return: `{"status": "ok", "checks": {...}}`

#### `GET /stats`
- Return: total records embedded, last ingest time, model info, index stats

### Result Merging (optional, Phase 1 stretch)
- Accept both semantic results and keyword result IDs
- Merge using reciprocal rank fusion
- Endpoint: `POST /search/merged`

## Interfaces

**You produce:**
- HTTP API consumed by Evergreen (Perl) via HTTP requests
- `SearchResponse` JSON responses

**You consume:**
- `embedding.embed_text()` — to embed the query
- `search.similarity_search()` — to find matching records
- `ingest.run_ingest()` — to trigger ingest jobs

## Testing

- Write unit tests in `tests/unit/test_api.py` using `pytest` and FastAPI's `TestClient`
- Test each endpoint: valid input, invalid input, service unavailable
- Integration tests verify full query flow (embed → search → response)

## Constraints

- Use FastAPI with Pydantic models for request/response validation
- Use `uvicorn` as the ASGI server
- All endpoints must return JSON
- Error responses must use standard HTTP status codes with detail messages
- Do NOT modify shared models without coordinating
