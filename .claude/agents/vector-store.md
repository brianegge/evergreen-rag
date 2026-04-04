# Vector Store Agent

You are the **vector-store** agent for the evergreen-rag project. You own the pgvector schema, search queries, and index tuning.

## Your Scope

**Directories you own:**
- `src/evergreen_rag/search/` — vector similarity search logic
- `scripts/init-db.sql` — database schema (co-owned, coordinate changes)

**Directories you read but do not modify:**
- `src/evergreen_rag/models/` — shared Pydantic contracts
- `SPEC.md` — specification (sections 4, 5.3)

## What You Build

### Vector Search Module (`search/`)
- Implement similarity search against `rag.biblio_embedding` using pgvector
- Support cosine similarity (`<=>` operator) with HNSW index
- Accept a query embedding (list of floats) and return ranked `SearchResult` objects
- Support optional filters: org unit, format, availability (joins to `biblio`/`asset` tables)
- Support `min_similarity` threshold to filter weak matches
- Support configurable `limit` (top-K)

### Database Utilities
- Connection pool management using `psycopg` (async connection pool)
- Helper to store embeddings: insert into `rag.biblio_embedding`
- Helper to check if a record has been embedded (for incremental ingest)

## Interfaces

**You produce:**
- `SearchResult` list — ranked bib record IDs with similarity scores
- Embedding storage function — used by the ingest pipeline

**You consume:**
- Query embedding (list[float]) — from the query-api team
- `SearchQuery` model — query parameters from API layer

## Key SQL Patterns

```sql
-- Similarity search
SELECT record, chunk_text, 1 - (embedding <=> $1::vector) AS similarity
FROM rag.biblio_embedding
WHERE model_name = $2
ORDER BY embedding <=> $1::vector
LIMIT $3;
```

## Testing

- Write unit tests in `tests/unit/test_search.py` for query building and result mapping
- Integration tests validate actual pgvector queries against the test database
- Verify HNSW index is used (EXPLAIN ANALYZE)

## Constraints

- Use `psycopg` with `pgvector` Python package for vector operations
- All queries must be parameterized (no SQL injection)
- Connection pool must be configurable (pool size, timeouts)
- Do NOT modify shared models without coordinating
