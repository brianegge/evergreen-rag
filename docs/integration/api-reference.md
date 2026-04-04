# Evergreen RAG API Reference

Base URL: `http://<rag-host>:8000`

All endpoints accept and return JSON. Set `Content-Type: application/json` for POST requests.

---

## POST /search

Perform a semantic search against the catalog using natural language.

### Request Body (`SearchQuery`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | | Natural language search query |
| `limit` | integer | no | 10 | Max results to return (1-100) |
| `org_unit` | integer | no | null | Filter by Evergreen org unit ID |
| `format` | string | no | null | Filter by item format (e.g., "book", "dvd") |
| `min_similarity` | float | no | 0.0 | Minimum cosine similarity threshold (0.0-1.0) |

### Response Body (`SearchResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The original query text |
| `results` | array | List of `SearchResult` objects |
| `total` | integer | Number of results returned |
| `model` | string | Embedding model used |

Each `SearchResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | integer | Evergreen `biblio.record_entry` ID |
| `similarity` | float | Cosine similarity score (0.0-1.0) |
| `chunk_text` | string | The matched text chunk |

### Example

**Request:**

```json
POST /search
Content-Type: application/json

{
  "query": "books about coping with grief for teenagers",
  "limit": 5,
  "org_unit": 1,
  "min_similarity": 0.3
}
```

**Response:**

```json
{
  "query": "books about coping with grief for teenagers",
  "results": [
    {
      "record_id": 54321,
      "similarity": 0.87,
      "chunk_text": "The Grieving Teen: A Guide for Teenagers and Their Friends\nby Helen Fitzgerald\nSubjects: Grief in adolescence; Teenagers; Bereavement"
    },
    {
      "record_id": 12045,
      "similarity": 0.79,
      "chunk_text": "Healing Your Grieving Heart for Teens: 100 Practical Ideas\nby Alan D. Wolfelt\nSubjects: Grief in adolescence; Loss (Psychology); Youth"
    }
  ],
  "total": 2,
  "model": "nomic-embed-text"
}
```

### Error Responses

| Status | Meaning |
|--------|---------|
| 422 | Invalid request body (e.g., limit out of range) |
| 503 | Embedding service or database unavailable |

---

## POST /search/merged

Merge semantic search results with Evergreen's native keyword search results using reciprocal rank fusion or weighted score combination.

> **Note:** This endpoint is planned for Phase 2. The contract below describes the expected interface.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | | Natural language search query |
| `keyword_result_ids` | array[integer] | yes | | Record IDs from Evergreen's native keyword search, in ranked order |
| `limit` | integer | no | 10 | Max merged results to return (1-100) |
| `org_unit` | integer | no | null | Filter by Evergreen org unit ID |
| `format` | string | no | null | Filter by item format |
| `semantic_weight` | float | no | 0.5 | Weight for semantic results in fusion (0.0-1.0) |

### Response Body (`SearchResponse`)

Same as `POST /search` — returns a `SearchResponse` with merged, re-ranked results.

### Example

**Request:**

```json
POST /search/merged
Content-Type: application/json

{
  "query": "books about coping with grief for teenagers",
  "keyword_result_ids": [12045, 67890, 11111, 22222, 33333],
  "limit": 10,
  "semantic_weight": 0.6
}
```

**Response:**

```json
{
  "query": "books about coping with grief for teenagers",
  "results": [
    {
      "record_id": 12045,
      "similarity": 0.92,
      "chunk_text": "Healing Your Grieving Heart for Teens..."
    },
    {
      "record_id": 54321,
      "similarity": 0.85,
      "chunk_text": "The Grieving Teen: A Guide..."
    },
    {
      "record_id": 67890,
      "similarity": 0.71,
      "chunk_text": "When Someone Dies: A Practical Guide..."
    }
  ],
  "total": 3,
  "model": "nomic-embed-text"
}
```

---

## POST /ingest

Trigger embedding ingest for specified records or a full re-ingest of all bibliographic records.

### Request Body (`IngestRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `record_ids` | array[integer] | no | null | Specific `biblio.record_entry` IDs to ingest |
| `all` | boolean | no | false | Set to true for full re-ingest of all records |

At least one of `record_ids` or `all: true` should be provided.

### Response Body (`IngestResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "completed" on success |
| `message` | string | Human-readable summary |
| `total` | integer | Total records processed |
| `embedded` | integer | Records successfully embedded |
| `failed` | integer | Records that failed to embed |

### Example: Ingest Specific Records

**Request:**

```json
POST /ingest
Content-Type: application/json

{
  "record_ids": [100, 200, 300]
}
```

**Response:**

```json
{
  "status": "completed",
  "message": "Ingested 3 records",
  "total": 3,
  "embedded": 3,
  "failed": 0
}
```

### Example: Full Re-Ingest

**Request:**

```json
POST /ingest
Content-Type: application/json

{
  "all": true
}
```

**Response:**

```json
{
  "status": "completed",
  "message": "Ingested 84523 records",
  "total": 84523,
  "embedded": 84501,
  "failed": 22
}
```

### Error Responses

| Status | Meaning |
|--------|---------|
| 422 | Invalid request body |
| 500 | Ingest pipeline error |

---

## GET /health

Check service health including database and embedding service connectivity.

### Response Body (`HealthResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "ok" if all checks pass, "degraded" otherwise |
| `checks` | object | Individual check results |

The `checks` object contains:

| Key | Type | Description |
|-----|------|-------------|
| `embedding_service` | boolean | Ollama/embedding service reachable |
| `database` | boolean | PostgreSQL/pgvector reachable |

### Example

**Request:**

```
GET /health
```

**Response (healthy):**

```json
{
  "status": "ok",
  "checks": {
    "embedding_service": true,
    "database": true
  }
}
```

**Response (degraded):**

```json
{
  "status": "degraded",
  "checks": {
    "embedding_service": false,
    "database": true
  }
}
```

---

## GET /stats

Return statistics about embedded records in the vector store.

### Response Body (`StatsResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `total_embeddings` | integer | Total embedding rows in `rag.biblio_embedding` |
| `unique_records` | integer | Distinct `biblio.record_entry` IDs with embeddings |
| `last_embedded_at` | string or null | ISO 8601 timestamp of most recent embedding |
| `model_name` | string or null | Name of the embedding model in use |

### Example

**Request:**

```
GET /stats
```

**Response:**

```json
{
  "total_embeddings": 84501,
  "unique_records": 84501,
  "last_embedded_at": "2026-04-03T14:22:00Z",
  "model_name": "nomic-embed-text"
}
```

### Error Responses

| Status | Meaning |
|--------|---------|
| 503 | Database unavailable |
