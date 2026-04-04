-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- RAG schema
CREATE SCHEMA IF NOT EXISTS rag;

-- Minimal biblio schema for development (mirrors Evergreen's structure)
CREATE SCHEMA IF NOT EXISTS biblio;

CREATE TABLE IF NOT EXISTS biblio.record_entry (
    id          BIGSERIAL PRIMARY KEY,
    marc        TEXT NOT NULL,
    quality     INTEGER,
    tcn_source  TEXT,
    tcn_value   TEXT,
    create_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    edit_date   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    deleted     BOOLEAN NOT NULL DEFAULT FALSE
);

-- Embedding storage
CREATE TABLE IF NOT EXISTS rag.biblio_embedding (
    id          BIGSERIAL PRIMARY KEY,
    record      BIGINT NOT NULL REFERENCES biblio.record_entry(id) ON DELETE CASCADE,
    chunk_index SMALLINT NOT NULL DEFAULT 0,
    chunk_text  TEXT NOT NULL,
    embedding   vector NOT NULL,
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(record, chunk_index, model_name)
);

-- HNSW index for approximate nearest neighbor search
-- Dimension will be set by the first insert; this uses cosine distance
CREATE INDEX IF NOT EXISTS idx_biblio_embedding_vec
    ON rag.biblio_embedding
    USING hnsw (embedding vector_cosine_ops);

-- Configuration
CREATE TABLE IF NOT EXISTS rag.config (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ingest tracking
CREATE TABLE IF NOT EXISTS rag.ingest_log (
    id              BIGSERIAL PRIMARY KEY,
    record          BIGINT NOT NULL REFERENCES biblio.record_entry(id),
    status          TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'error')),
    error_msg       TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);
