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
    embedding   vector(768) NOT NULL,
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(record, chunk_index, model_name)
);

-- HNSW index for approximate nearest neighbor search (768 = nomic-embed-text dims)
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

-- Notify the RAG service when bib records change.
-- Payload format: "upsert:<id>" or "delete:<id>"
CREATE OR REPLACE FUNCTION biblio.notify_rag_ingest()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('rag_ingest', 'delete:' || OLD.id::text);
        RETURN OLD;
    END IF;

    -- Treat soft-deleted or NULL-marc records as deletes
    IF NEW.deleted OR NEW.marc IS NULL THEN
        PERFORM pg_notify('rag_ingest', 'delete:' || NEW.id::text);
    ELSE
        PERFORM pg_notify('rag_ingest', 'upsert:' || NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_rag_ingest ON biblio.record_entry;
CREATE TRIGGER trg_rag_ingest
    AFTER INSERT OR UPDATE OF marc, deleted ON biblio.record_entry
    FOR EACH ROW
    EXECUTE FUNCTION biblio.notify_rag_ingest();

DROP TRIGGER IF EXISTS trg_rag_ingest_delete ON biblio.record_entry;
CREATE TRIGGER trg_rag_ingest_delete
    AFTER DELETE ON biblio.record_entry
    FOR EACH ROW
    EXECUTE FUNCTION biblio.notify_rag_ingest();
