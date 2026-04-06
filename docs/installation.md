# Installation Guide

## Overview

The Evergreen RAG service has three components:

1. **PostgreSQL + pgvector** — stores MARC records and vector embeddings
2. **Ollama** — runs AI models locally (no cloud APIs, no data leaves your network)
3. **RAG service** — Python API that connects everything

All three can run on a single server, or be distributed across machines.

## Hardware Requirements

### Minimum (small catalog, < 10k records)
- 4 CPU cores
- 8 GB RAM
- 20 GB disk
- Any modern CPU (no GPU required, but slow)

### Recommended (up to 100k records)
- 8 CPU cores
- 16 GB RAM (or 16 GB unified memory on Apple Silicon)
- 50 GB disk
- GPU with 8+ GB VRAM, or Apple Silicon M1/M2/M4

### GPU Recommendations

Ollama runs models on GPU when available, which dramatically improves speed:

| GPU | VRAM | Embedding Speed | Generation Speed | Notes |
|-----|------|----------------|-----------------|-------|
| Apple M4 (16GB) | 11.8 GB shared | ~650 records/min | ~30 tokens/sec | Tested. Excellent for dev/small production |
| Apple M1/M2 | 8-16 GB shared | ~400 records/min | ~20 tokens/sec | Good for development |
| NVIDIA RTX 3060 | 12 GB | ~800 records/min | ~40 tokens/sec | Good budget option |
| NVIDIA RTX 4090 | 24 GB | ~2000 records/min | ~80 tokens/sec | Can run larger models |
| CPU only | N/A | ~50 records/min | ~5 tokens/sec | Functional but slow |

Initial embedding of 100k records takes about **2.5 hours** on Apple M4 with `mxbai-embed-large`.

## Model Selection

### Embedding Models

The embedding model converts text into vectors for similarity search. This is the most
important model choice — it determines search quality.

| Model | Dimensions | Size | Quality | Speed | Recommendation |
|-------|-----------|------|---------|-------|---------------|
| `mxbai-embed-large` | 1024 | 669 MB | Best | Slower | **Recommended** — best semantic understanding |
| `nomic-embed-text` | 768 | 274 MB | Good | Fast | Good for limited hardware or very large catalogs |
| `all-minilm` | 384 | 45 MB | Basic | Fastest | Only if resources are very constrained |

We tested both `mxbai-embed-large` and `nomic-embed-text` on 100k SFPL records:
- `mxbai-embed-large` found "Ready Player Two" for a "video games" query (rank #5)
- `nomic-embed-text` did not find it in the top 200 results
- For natural language queries, `mxbai-embed-large` consistently returned more relevant results

**Note:** Changing the embedding model requires re-embedding all records and updating the
`vector()` column dimension in the database schema.

### Generation Models (LLM)

The generation model writes the AI summaries shown to patrons. It's optional — the
semantic search works without it, but the summaries help patrons decide which titles
to explore.

| Model | Parameters | Size | Quality | Speed | VRAM Needed |
|-------|-----------|------|---------|-------|-------------|
| `qwen2.5:7b` | 7.6B | 4.7 GB | Good | ~30 tok/s on M4 | 6 GB |
| `qwen2.5:3b` | 3.1B | 1.9 GB | Decent | ~50 tok/s on M4 | 3 GB |
| `llama3.1:8b` | 8B | 4.7 GB | Good | ~25 tok/s on M4 | 6 GB |
| `mistral:7b` | 7B | 4.1 GB | Good | ~30 tok/s on M4 | 5 GB |
| `gemma2:9b` | 9B | 5.4 GB | Very good | ~20 tok/s on M4 | 7 GB |

We use `qwen2.5:7b` — it follows prompt instructions well, generates concise summaries,
and reliably uses the record reference tokens needed for linked titles.

**Both models run simultaneously:** The embedding model loads when a query arrives and
unloads after idle timeout. The generation model loads for the summary step. On 16 GB
systems, both fit comfortably.

## Step-by-Step Installation

### 1. Install PostgreSQL + pgvector

#### Option A: Container (recommended)

```bash
# Using podman or docker
podman run -d --name pgvector \
  -e POSTGRES_USER=evergreen \
  -e POSTGRES_PASSWORD=evergreen \
  -p 5432:5432 \
  -v /path/to/pgdata:/var/lib/postgresql/data:Z \
  --restart=always \
  pgvector/pgvector:pg16
```

#### Option B: Native install

```bash
# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# Enable the extension
sudo -u postgres psql -c "CREATE EXTENSION vector;"
```

#### Create the database and schema

```bash
# Create database
psql -U evergreen -c "CREATE DATABASE evergreen_rag;"

# Apply schema
psql -U evergreen evergreen_rag < scripts/init-db.sql
```

### 2. Install Ollama

```bash
# macOS
brew install ollama
brew services start ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull mxbai-embed-large
ollama pull qwen2.5:7b
```

If the RAG service runs on a different machine than Ollama, configure Ollama to
listen on all interfaces:

```bash
# macOS (homebrew)
# Add to ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist EnvironmentVariables:
#   OLLAMA_HOST = 0.0.0.0:11434

# Linux
# Add to /etc/systemd/system/ollama.service:
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

### 3. Install the RAG Service

#### Option A: Container (recommended for production)

```bash
docker run -d --name evergreen-rag \
  -e DATABASE_URL=postgresql://evergreen:evergreen@db-host:5432/evergreen_rag \
  -e OLLAMA_URL=http://ollama-host:11434 \
  -e EMBEDDING_MODEL=mxbai-embed-large \
  -e GENERATION_MODEL=qwen2.5:7b \
  -p 8000:8000 \
  --restart=always \
  ghcr.io/brianegge/evergreen-rag:latest
```

That's it. No Python, no virtual environments, no dependencies to install.

#### Option B: From source (for development)

```bash
git clone https://github.com/brianegge/evergreen-rag.git
cd evergreen-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Configure

For container deployments, pass environment variables with `-e` flags (see above).

For source installs, create a `.env` file:

```bash
DATABASE_URL=postgresql://evergreen:evergreen@localhost:5432/evergreen_rag
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=mxbai-embed-large
GENERATION_MODEL=qwen2.5:7b
```

Adjust hostnames if services are on different machines.

### 5. Load MARC Records

If your Evergreen database is on a separate server, the RAG service needs a copy
of `biblio.record_entry` in the `evergreen_rag` database. The schema includes a
minimal `biblio.record_entry` table for this purpose.

```bash
# Option A: Copy from Evergreen database
pg_dump -h evergreen-db-host -U evergreen -t biblio.record_entry --data-only evergreen | \
  psql -U evergreen evergreen_rag

# Option B: Load from MARC files
python scripts/load_marc_bulk.py --source ia --limit 100000
```

### 6. Build Embeddings

```bash
# Start the service
source .env
uvicorn evergreen_rag.api.main:app --host 0.0.0.0 --port 8000

# Trigger full ingest (runs in background)
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"all": true}'
```

Monitor progress:
```bash
psql -U evergreen evergreen_rag -c "SELECT COUNT(*) FROM rag.biblio_embedding;"
```

### 7. Integrate with Evergreen OPAC (optional)

Add a reverse proxy in Evergreen's Apache config:

```apache
ProxyPass /rag http://rag-service-host:8000
ProxyPassReverse /rag http://rag-service-host:8000
```

Add the streaming search template to the OPAC results page and the "no results"
page:

```
# In opac/parts/result/table.tt2 (before the results row):
[% INCLUDE "opac/parts/result/rag_streaming.tt2" %]

# In opac/parts/result/lowhits.tt2 (at the top):
[% INCLUDE "opac/parts/result/rag_streaming.tt2" %]
```

Copy `src/evergreen_rag/static/opac/rag_streaming.tt2` into the Evergreen
templates directory.

### 8. Run as a Service (production)

```bash
# systemd unit file
sudo tee /etc/systemd/system/evergreen-rag.service << EOF
[Unit]
Description=Evergreen RAG Service
After=network.target postgresql.service

[Service]
Type=simple
User=evergreen
WorkingDirectory=/path/to/evergreen-rag
EnvironmentFile=/path/to/evergreen-rag/.env
ExecStart=/path/to/evergreen-rag/.venv/bin/uvicorn \
  evergreen_rag.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now evergreen-rag
```

## Upgrading the Embedding Model

If you switch embedding models (e.g., from `nomic-embed-text` to `mxbai-embed-large`):

1. Check the new model's embedding dimensions:
   ```bash
   curl http://localhost:11434/api/embed -d '{"model":"mxbai-embed-large","input":"test"}' | \
     python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))"
   ```

2. Update the database schema:
   ```sql
   DROP INDEX rag.idx_biblio_embedding_vec;
   TRUNCATE rag.biblio_embedding;
   ALTER TABLE rag.biblio_embedding DROP COLUMN embedding;
   ALTER TABLE rag.biblio_embedding ADD COLUMN embedding vector(1024) NOT NULL;
   CREATE INDEX idx_biblio_embedding_vec
     ON rag.biblio_embedding USING hnsw (embedding vector_cosine_ops);
   ```

3. Update `.env` with the new model name

4. Re-ingest all records:
   ```bash
   curl -X POST http://localhost:8000/ingest -d '{"all": true}' \
     -H 'Content-Type: application/json'
   ```

## Troubleshooting

**Search returns no results:** Check that embeddings exist: `SELECT COUNT(*) FROM rag.biblio_embedding;`

**Slow searches:** Ollama may be busy with ingest. Searches queue behind embedding requests. Wait for ingest to complete, or run Ollama on a separate machine.

**"Embedding service error":** Ollama is unreachable. Check `curl http://ollama-host:11434/api/tags`.

**OPAC shows raw tokens like `305`:** The server-side token expansion isn't working. Ensure the RAG service is running and the Apache proxy is configured.
