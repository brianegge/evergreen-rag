# Evergreen RAG API -- curl Examples

These examples assume the RAG sidecar is running at `http://localhost:8000`. Adjust the host and port for your deployment.

---

## Semantic Search

Basic natural language search:

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "books about coping with grief for teenagers",
    "limit": 5
  }' | python3 -m json.tool
```

Search with filters:

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "introductory calculus textbook",
    "limit": 10,
    "org_unit": 1,
    "format": "book",
    "min_similarity": 0.3
  }' | python3 -m json.tool
```

Extract just the record IDs (useful for piping to Evergreen):

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "history of jazz music", "limit": 20}' \
  | python3 -c "import sys,json; [print(r['record_id']) for r in json.load(sys.stdin)['results']]"
```

---

## Merged Search (Phase 2)

Merge semantic results with Evergreen keyword results:

```bash
curl -s -X POST http://localhost:8000/search/merged \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "books about coping with grief for teenagers",
    "keyword_result_ids": [12045, 67890, 11111, 22222, 33333],
    "limit": 10,
    "semantic_weight": 0.6
  }' | python3 -m json.tool
```

---

## Ingest

Ingest specific records by ID:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "record_ids": [100, 200, 300]
  }' | python3 -m json.tool
```

Full re-ingest of all bibliographic records:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "all": true
  }' | python3 -m json.tool
```

---

## Health Check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Quick one-liner for monitoring scripts (exits non-zero if degraded):

```bash
curl -sf http://localhost:8000/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data['status'] != 'ok':
    print('DEGRADED:', data['checks'], file=sys.stderr)
    sys.exit(1)
print('OK')
"
```

---

## Stats

```bash
curl -s http://localhost:8000/stats | python3 -m json.tool
```
