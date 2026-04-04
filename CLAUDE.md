# evergreen-rag

Retrieval-Augmented Generation for Evergreen ILS Search. See `SPEC.md` for the full specification.

## Quick Reference

- **Language:** Python 3.11+
- **Deployment:** Sidecar container, communicates with Evergreen via HTTP API
- **Vector store:** pgvector on PostgreSQL 14+
- **Embedding:** Self-hosted only (Ollama), no cloud APIs

## Build & Run

```bash
# Start dev environment (PostgreSQL+pgvector, Ollama, RAG service)
docker compose up -d

# Install locally for development
pip install -e ".[dev]"
```

## Test Commands

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires running docker compose services)
python tests/integration/test_ingest.py

# Lint
ruff check src/ tests/
```

## Project Structure

```
src/evergreen_rag/
├── api/          # FastAPI HTTP endpoints (query-api team)
├── embedding/    # Embedding service wrapper (embedding team)
├── extractor/    # MARC-XML text extraction (data-pipeline team)
├── ingest/       # Batch ingest orchestration (data-pipeline team)
├── models/       # Shared Pydantic models (contracts between components)
└── search/       # pgvector similarity search (vector-store team)
```

## Architecture

Evergreen (Perl) → HTTP → RAG sidecar (Python/FastAPI) → PostgreSQL (pgvector)

The RAG service is a standalone Python container. It:
1. Reads MARC records from `biblio.record_entry` in PostgreSQL
2. Extracts text and generates embeddings via Ollama
3. Stores vectors in `rag.biblio_embedding` (pgvector)
4. Serves semantic search queries via HTTP API

## Shared Interfaces

All component contracts are defined as Pydantic models in `src/evergreen_rag/models/`:
- `ExtractedRecord` — MARC text extraction output
- `EmbeddingRequest/Response` — embedding service interface
- `SearchQuery/Result/Response` — search API interface

Agent teams must use these models at component boundaries. Changes to models require updating all consumers.

## Agent Team Ownership

| Team | Owns | Directories |
|------|------|-------------|
| data-pipeline | MARC extraction, ingest | `extractor/`, `ingest/` |
| vector-store | Schema, search queries | `search/`, `scripts/init-db.sql` |
| embedding | Embedding service | `embedding/` |
| query-api | HTTP API | `api/` |
| integration | Evergreen hooks, OPAC | (Phase 2) |

## Agent Teams

This project uses Claude Code's experimental agent teams feature. Teammate definitions live in `.claude/agents/` as subagent definitions that teammates inherit when spawned.

| Teammate | Definition | Role |
|----------|-----------|------|
| data-pipeline | `.claude/agents/data-pipeline.md` | MARC extraction + batch ingest |
| vector-store | `.claude/agents/vector-store.md` | pgvector schema + similarity search |
| embedding | `.claude/agents/embedding.md` | Ollama embedding wrapper |
| query-api | `.claude/agents/query-api.md` | FastAPI HTTP endpoints |
| integration | `.claude/agents/integration.md` | Evergreen integration (Phase 2) |

### How to Launch

The team lead (your main session) creates the team. Example:

```
Create an agent team for Phase 1. Spawn teammates:
- One using the embedding agent type to build the Ollama wrapper
- One using the data-pipeline agent type to build MARC extraction and ingest
Start with these two in parallel. When both finish, spawn vector-store,
then query-api. Require plan approval before implementation.
```

### Phase 1 Dependency Order

1. `embedding` + `data-pipeline` (independent, parallel)
2. `vector-store` (needs embedding interface confirmed)
3. `query-api` (needs embedding + search modules)

### Coordination Rules

- Shared models in `models/` are the contract — teammates must not change them unilaterally
- Each teammate writes tests for its own component
- Teammates must not modify files outside their owned directories
- If a teammate needs an interface change, message the lead with the request
- Teammates should message each other when their work affects a shared boundary

## Conventions

- All database access uses `psycopg` (async where possible)
- HTTP client calls use `httpx`
- Test fixtures go in `tests/fixtures/`
- Integration tests use `morganstanley/testplan` with custom drivers
- Unit tests use `pytest`
