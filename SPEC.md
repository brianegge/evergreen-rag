# evergreen-rag: Retrieval-Augmented Generation for Evergreen ILS Search

**Status:** Draft
**Version:** 0.1.0
**Last Updated:** 2026-04-03

---

## 1. Problem Statement

Evergreen ILS search relies entirely on PostgreSQL full-text search (`tsvector`/`ts_rank`) against pre-extracted MARC field indexes. This approach has well-documented limitations:

- **No semantic understanding** — queries must match indexed terms; conceptually related results are missed
- **No natural language support** — patrons cannot ask questions like "books about coping with grief for teenagers"
- **Limited relevance ranking** — `ts_rank` lacks BM25, learned ranking, or field-boosting beyond manual weights
- **No spellcheck, synonyms, or "did you mean"** functionality
- **Faceting is post-query computed**, impacting performance on large result sets

## 2. Proposed Solution

Add a **Retrieval-Augmented Generation (RAG)** layer that:

1. **Embeds** bibliographic records (MARC data) as vectors for semantic search
2. **Retrieves** semantically relevant records given a natural language query
3. **Augments** Evergreen's existing search — not replaces it — by offering a complementary search mode
4. **Optionally generates** natural language summaries, reading recommendations, or faceted refinements via LLM

## 3. Architecture Overview

```
+----------------------------------------------------------+
|                     Evergreen Host                        |
|                                                          |
|  +------------------+                                    |
|  |   Evergreen OPAC |----+                               |
|  |  / Staff Client  |    |                               |
|  +------------------+    |                               |
|                          | HTTP                          |
|  +------------------+    |     +-----------------------+ |
|  | Evergreen Native |    |     | RAG Sidecar Container | |
|  | Search (existing)|    +---->|                       | |
|  +------------------+          |  RAG Query API        | |
|                                |  Embedding Service    | |
|                                |  Ingest Pipeline      | |
|                                +-----------+-----------+ |
|                                            |             |
|                                     SQL    |             |
|                                            v             |
|                                +-----------+-----------+ |
|                                |     PostgreSQL        | |
|                                |  biblio.record_entry  | |
|                                |  rag.biblio_embedding | |
|                                |  (pgvector)           | |
|                                +-----------------------+ |
+----------------------------------------------------------+
```

### 3.1 Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | **pgvector** (PostgreSQL extension) | Evergreen already runs on PostgreSQL; no new infrastructure required. Lowers adoption barrier. |
| Embedding model | TBD (candidate: `nomic-embed-text`, `bge-large`) | Self-hosted only; must run in the sidecar container |
| Deployment | **Sidecar container** (Python) | Decoupled from Evergreen's Perl stack; communicates via HTTP API; connects directly to PostgreSQL |
| Integration point | HTTP API from Evergreen to sidecar | Evergreen sends HTTP requests; no Python dependency in Evergreen codebase |
| LLM for generation | Optional, pluggable | Libraries may not want/allow LLM generation; RAG retrieval must work standalone |

### 3.2 Data Flow

#### Ingest Pipeline (offline/batch)
1. Read MARC records from `biblio.record_entry`
2. Extract searchable text: title, author, subjects, summary/abstract (MARC 520), ToC (MARC 505), notes
3. Chunk records (one record = one or more chunks depending on content length)
4. Generate embeddings via configured model
5. Store vectors in `rag.biblio_embedding` table (pgvector)

#### Query Pipeline (online)
1. Patron/staff submits natural language query
2. Query is embedded using the same model
3. Vector similarity search (`cosine` or `inner product`) returns top-N candidate bib record IDs
4. (Optional) Results are re-ranked or merged with native Evergreen search results
5. (Optional) LLM generates a summary/recommendation from retrieved records
6. Record IDs are returned to Evergreen's existing display pipeline

## 4. Database Schema

```sql
-- New schema to keep RAG components isolated
CREATE SCHEMA rag;

-- Embedding storage
CREATE TABLE rag.biblio_embedding (
    id          BIGSERIAL PRIMARY KEY,
    record      BIGINT NOT NULL REFERENCES biblio.record_entry(id) ON DELETE CASCADE,
    chunk_index SMALLINT NOT NULL DEFAULT 0,
    chunk_text  TEXT NOT NULL,
    embedding   vector NOT NULL,  -- pgvector type; dimension set by model
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(record, chunk_index, model_name)
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_biblio_embedding_vec
    ON rag.biblio_embedding
    USING hnsw (embedding vector_cosine_ops);

-- Configuration
CREATE TABLE rag.config (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ingest tracking
CREATE TABLE rag.ingest_log (
    id          BIGSERIAL PRIMARY KEY,
    record      BIGINT NOT NULL REFERENCES biblio.record_entry(id),
    status      TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'error')),
    error_msg   TEXT,
    started_at  TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);
```

## 5. Components

### 5.1 MARC Text Extractor
- **Input:** MARC-XML from `biblio.record_entry.marc`
- **Output:** Structured text suitable for embedding
- **Extracts:** Title (245), Author (100/110/700/710), Subjects (6XX), Summary (520), ToC (505), Notes (500), Series (490/830), Edition (250), Publisher (264/260)
- **Language:** Perl (to align with Evergreen's ingest pipeline) or Python (for ML ecosystem access)

### 5.2 Embedding Service
- **Input:** Text chunks from extractor
- **Output:** Dense vectors
- **Requirements:**
  - Must support self-hosted models (library data privacy)
  - Must be configurable (model, dimension, batch size)
  - Should support GPU acceleration where available
- **Candidates:** Ollama, vLLM, HuggingFace TEI, or direct `sentence-transformers`
- **Deployment:** Self-hosted only; no patron query data sent to external APIs

### 5.3 Vector Search Service
- **Input:** Query embedding + optional filters (org unit, copy status, format)
- **Output:** Ranked list of `biblio.record_entry` IDs with similarity scores
- **Implementation:** SQL queries against pgvector with optional joins to `asset.copy` / `metabib.record_attr` for filtering

### 5.4 RAG Query API
- **Input:** Natural language query string + search parameters
- **Output:** Search results (bib record IDs + scores) and optional generated text
- **Integration:** OpenSRF service (`open-ils.rag`) registered with the Evergreen router
- **Fallback:** Standalone HTTP API for testing/development without full Evergreen stack

### 5.5 Result Merger (optional)
- **Merges** traditional keyword search results with semantic search results
- **Strategies:** reciprocal rank fusion, weighted score combination, or cascading (semantic first, keyword fallback)

### 5.6 LLM Generation Layer (optional)
- **Input:** Retrieved record data + original query
- **Output:** Natural language summary, reading recommendation, or search refinement suggestions
- **Requirements:** Must be optional; core retrieval works without it
- **Privacy:** Self-hosted models only; no patron query data sent to external APIs

## 6. Hardware Requirements

### 6.1 Development / Small Library (up to 100K bib records)
- **CPU:** Apple Silicon M1+ or x86-64 with AVX2
- **RAM:** 16GB minimum
- **Disk:** 10GB free (PostgreSQL + vectors + model weights)
- **GPU:** Not required; Apple Metal or integrated GPU sufficient
- **Embedding throughput:** ~50-100 records/sec (batch ingest)
- **Reference system:** Mac Mini M4, 16GB RAM, 10-core GPU

### 6.2 Medium Library (100K-1M bib records)
- **RAM:** 32GB recommended (HNSW index grows with collection size)
- **Disk:** 50GB+ (vector storage scales ~3KB per record at 768 dimensions)
- **GPU:** NVIDIA GPU with 8GB+ VRAM recommended for batch ingest speed

### 6.3 Large Library / Consortium (1M+ bib records)
- **RAM:** 64GB+ recommended
- **Disk:** SSD required; 100GB+ for vector indexes
- **GPU:** NVIDIA GPU with 16GB+ VRAM for reasonable ingest times
- **Note:** Consider partitioning `rag.biblio_embedding` by org unit or record source

### 6.4 Embedding Model Sizing

| Model | Parameters | RAM Usage | Dimensions | Quality |
|-------|-----------|-----------|------------|---------|
| `all-MiniLM-L6-v2` | 22M | ~90MB | 384 | Good |
| `nomic-embed-text` | 137M | ~550MB | 768 | Very good |
| `bge-large-en-v1.5` | 335M | ~1.3GB | 1024 | Excellent |

All models run on CPU. Apple Metal acceleration available via Ollama on macOS.

## 7. Integration with Evergreen

### 7.1 Minimal Integration (Phase 1)
- Standalone HTTP API that accepts queries and returns bib record IDs
- OPAC integration via JavaScript: add "Semantic Search" tab/toggle
- No OpenSRF dependency; can run as a sidecar service

### 7.2 Full Integration (Phase 2)
- OpenSRF service registered as `open-ils.rag`
- Ingest hooks triggered by `open-ils.ingest` on new/updated records
- Staff client catalog integration in Angular `eg2`
- SRU/Z39.50 gateway support

## 8. Non-Goals (v0.1)

- Replacing Evergreen's existing search
- Real-time streaming/chat interface
- Multi-language embedding (initially English; extensible later)
- Patron data or circulation history in the RAG context
- Fine-tuning embedding models on library data

## 9. Agent Team Structure

This project uses a multi-agent development approach. Each agent team owns a component and works against this spec.

| Agent Team | Component | Key Deliverables |
|------------|-----------|-----------------|
| **data-pipeline** | MARC Text Extractor + Ingest Pipeline | MARC-to-text extraction, chunking strategy, batch ingest orchestration |
| **vector-store** | pgvector Schema + Search | Schema migrations, HNSW index tuning, similarity search queries, filter integration |
| **embedding** | Embedding Service | Model evaluation, embedding API wrapper, batch embedding pipeline |
| **query-api** | RAG Query API | HTTP API, query embedding, result ranking, OpenSRF service stub |
| **integration** | Evergreen Integration | OPAC JavaScript, staff client Angular, ingest hooks, result merger |

### 9.1 Agent Team Interfaces

Each team communicates through well-defined interfaces:

```
data-pipeline --[text chunks]--> embedding --[vectors]--> vector-store
                                                              |
query-api --[query embedding]---> vector-store --[bib IDs]--> query-api
                                                              |
integration <--[search results]-- query-api
```

## 10. Testing Strategy

### 10.1 Framework

Integration and quality tests use **[testplan](https://github.com/morganstanley/testplan)** (`morganstanley/testplan`), a Python multi-testing framework designed for functional and integration testing with managed service lifecycles.

### 10.2 Custom Test Drivers

| Driver | Manages | Lifecycle |
|--------|---------|-----------|
| `PostgresDriver` | PostgreSQL instance with pgvector extension | Start/stop, schema migration, seed data |
| `EmbeddingServiceDriver` | Ollama or embedding API process | Start/stop, model loading, health check |
| `IngestDriver` | Batch ingest pipeline | Runs ingest against test MARC records |

### 10.3 Test Suites

#### Unit Tests (pytest, no services required)
- MARC text extraction logic
- Chunking/tokenization
- Query parsing and parameter validation

#### Integration Tests (testplan, requires running services)
- End-to-end ingest: MARC record -> text -> embedding -> pgvector storage
- Query pipeline: natural language query -> embedding -> vector search -> ranked results
- Filter integration: org unit, format, availability filters applied correctly
- API contract: HTTP endpoints return expected shapes

#### Retrieval Quality Tests (testplan, requires running services)
- **Known-item retrieval:** Given a query derived from a known record's title/subject, that record must appear in the top-K results (e.g., top 5)
- **Semantic relevance:** Queries using synonyms or natural language paraphrases of subject headings must return relevant records
- **Negative cases:** Unrelated queries must NOT surface specific records
- **Quality threshold:** A curated test set of query/expected-result pairs; overall recall@K must meet a minimum threshold (e.g., 80% recall@10)

### 10.4 Test Data
- A curated MARC dataset (~1000 records) covering diverse formats, subjects, and field lengths
- A query test set (50-100 query/expected-result pairs) maintained as a fixture file
- Test data committed to the repo under `tests/fixtures/`

## 11. Development Phases

### Phase 0: Foundation
- [ ] Project scaffolding and CI
- [ ] pgvector installation/configuration guide
- [ ] Embedding model evaluation and selection
- [ ] Sample MARC dataset for development
- [ ] Testplan setup with custom drivers (Postgres, Embedding Service)
- [ ] Curate initial query/expected-result test set

### Phase 1: Core Pipeline
- [ ] MARC text extractor
- [ ] Embedding service wrapper
- [ ] Batch ingest pipeline
- [ ] Vector similarity search
- [ ] Basic HTTP query API
- [ ] Integration tests (ingest + query end-to-end)
- [ ] Retrieval quality tests (known-item, semantic, negative)

### Phase 2: Evergreen Integration
- [ ] OPAC semantic search UI
- [ ] Result merging with native search
- [ ] Ingest hook integration
- [ ] Staff client search integration

### Phase 3: Enhancement
- [ ] LLM generation layer
- [ ] Search refinement suggestions
- [ ] Multi-language support
- [ ] Performance optimization and benchmarking

## 11. Open Questions

1. ~~**Embedding model:** Self-hosted only, or allow optional cloud API?~~ **Resolved:** Self-hosted only. No patron query data leaves the network. Cloud API may be considered in a future version with explicit opt-in.
2. ~~**Chunking strategy:** One embedding per bib record, or split long records (505/520 fields)?~~ **Resolved:** One embedding per bib record for v0.1. Schema supports chunking (`chunk_index`) if retrieval quality tests show degradation on long records (505/520). Revisit based on quality test results.
3. ~~**pgvector version:** Requires PostgreSQL 12+ with pgvector 0.5+. What PG versions do active Evergreen sites run?~~ **Resolved:** Require PostgreSQL 14+ with pgvector 0.7+. PG 14 is a reasonable floor; pgvector 0.7+ provides mature HNSW indexes.
4. ~~**Perl vs Python:** Extractor/ingest in Perl (Evergreen-native) or Python (ML ecosystem)? Or bridge?~~ **Resolved:** Python service with HTTP API, deployed as a sidecar container. Connects directly to PostgreSQL for vector operations. Evergreen communicates with it via HTTP. No Python dependency in the Evergreen codebase.
5. ~~**Community input:** Should this go through the Evergreen RFC process before implementation?~~ **Resolved:** Prototype first, RFC later. Build a working Phase 1 demo, then submit to the Evergreen community with a tangible proof of concept.
