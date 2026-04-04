# Evergreen RAG Sidecar Deployment Guide

This guide covers deploying the RAG sidecar container alongside an existing Evergreen ILS installation.

---

## Architecture

```
+--------------------+         +--------------------+
|   Evergreen Host   |  HTTP   | RAG Sidecar        |
|   (OPAC, staff,    |-------->| (Python/FastAPI)    |
|    OpenSRF)        |  :8000  |                     |
+--------------------+         +----------+----------+
                                          |
                                    SQL   |
                                          v
                               +----------+----------+
                               |    PostgreSQL        |
                               |  (shared with        |
                               |   Evergreen)         |
                               +----------------------+
```

The RAG sidecar connects directly to the same PostgreSQL instance that Evergreen uses. It reads from `biblio.record_entry` and writes to the `rag` schema. Evergreen communicates with the sidecar over HTTP.

---

## Docker Compose Setup

### Production Compose File

Create a `docker-compose.rag.yml` alongside your Evergreen deployment:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:11434/api/tags || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
    # Uncomment for NVIDIA GPU support:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  rag:
    image: evergreen-rag:latest
    build:
      context: .
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      DATABASE_URL: postgresql://evergreen:${DB_PASSWORD}@${DB_HOST}:5432/evergreen
      OLLAMA_URL: http://ollama:11434
      EMBEDDING_MODEL: nomic-embed-text
    depends_on:
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 30s

volumes:
  ollama_data:
```

### Starting the Services

```bash
# Pull the embedding model (first time only)
docker compose -f docker-compose.rag.yml up -d ollama
docker compose -f docker-compose.rag.yml exec ollama ollama pull nomic-embed-text

# Start the RAG sidecar
docker compose -f docker-compose.rag.yml up -d

# Verify health
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Using Podman

The same compose file works with `podman-compose`:

```bash
podman-compose -f docker-compose.rag.yml up -d
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | yes | | PostgreSQL connection string. Must point to the Evergreen database. Format: `postgresql://user:password@host:port/dbname` |
| `OLLAMA_URL` | yes | | URL of the Ollama embedding service. Use `http://ollama:11434` when running in the same compose network. |
| `EMBEDDING_MODEL` | no | `nomic-embed-text` | Ollama model name for generating embeddings. Must be pulled into Ollama before use. |

### Example `.env` File

Create a `.env` file next to your compose file:

```bash
# Database connection (use the same credentials as Evergreen)
DB_HOST=db.example.org
DB_PASSWORD=your_evergreen_db_password

# Embedding model (must be pulled into Ollama)
EMBEDDING_MODEL=nomic-embed-text
```

---

## Database Setup

The RAG sidecar requires the `rag` schema and `pgvector` extension in the Evergreen database. Run the initialization script once:

```bash
# From the evergreen-rag project directory
psql -h $DB_HOST -U evergreen -d evergreen -f scripts/init-db.sql
```

This creates:
- `pgvector` extension (if not already installed)
- `rag` schema
- `rag.biblio_embedding` table with HNSW index
- `rag.config` table
- `rag.ingest_log` table

The `rag` schema is isolated from Evergreen's existing schemas and does not modify any Evergreen tables.

---

## Health Check Configuration

The `/health` endpoint reports the status of both the embedding service and database connections.

### Docker Health Check

Already configured in the compose file above. Docker will mark the container as unhealthy if the health endpoint fails 3 consecutive checks.

### Monitoring Integration

For Nagios, Icinga, or similar monitoring:

```bash
#!/bin/bash
# check_rag_health.sh -- Nagios-compatible health check
RESPONSE=$(curl -sf http://localhost:8000/health 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "CRITICAL: RAG service unreachable"
    exit 2
fi

STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
if [ "$STATUS" = "ok" ]; then
    echo "OK: RAG service healthy"
    exit 0
else
    echo "WARNING: RAG service degraded -- $RESPONSE"
    exit 1
fi
```

### Systemd Watchdog (Alternative to Docker)

If running the RAG service directly with systemd instead of Docker:

```ini
[Unit]
Description=Evergreen RAG Sidecar
After=network.target postgresql.service

[Service]
Type=simple
User=opensrf
Environment=DATABASE_URL=postgresql://evergreen:password@localhost:5432/evergreen
Environment=OLLAMA_URL=http://localhost:11434
Environment=EMBEDDING_MODEL=nomic-embed-text
ExecStart=/usr/bin/uvicorn evergreen_rag.api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## Network Configuration

### Binding the RAG Service

By default, the compose file binds port 8000 to `127.0.0.1` only:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

This ensures the RAG API is only accessible from the Evergreen host itself. If Evergreen and the RAG sidecar run on different hosts, adjust the bind address or use a reverse proxy.

### Firewall Rules

The RAG sidecar needs:

| Direction | Port | Purpose |
|-----------|------|---------|
| Outbound | 5432 | PostgreSQL (Evergreen database) |
| Outbound | 11434 | Ollama (embedding service) |
| Inbound | 8000 | RAG API (from Evergreen only) |

Restrict inbound access to port 8000 to only the Evergreen application servers:

```bash
# iptables example: allow only Evergreen app server
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.5 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### Reverse Proxy (Optional)

If you want to put the RAG API behind Apache (which Evergreen already uses):

```apache
# In your Apache config
ProxyPass /rag http://127.0.0.1:8000
ProxyPassReverse /rag http://127.0.0.1:8000

# Restrict access to staff network
<Location /rag>
    Require ip 10.0.0.0/8
</Location>
```

Evergreen would then call `http://localhost/rag/search` instead of `http://localhost:8000/search`.

### Docker Networking

When both Ollama and the RAG sidecar run in Docker Compose, they share a default bridge network. The RAG service reaches Ollama at `http://ollama:11434` using Docker's internal DNS.

If the PostgreSQL server runs outside Docker (on the host or a separate server), use the host's IP or `host.docker.internal` (Docker Desktop) / `172.17.0.1` (Linux Docker default gateway):

```yaml
environment:
  # When PostgreSQL runs on the Docker host
  DATABASE_URL: postgresql://evergreen:password@host.docker.internal:5432/evergreen
```

---

## Initial Data Ingest

After deployment, run the initial ingest to embed all existing bibliographic records:

```bash
# Trigger full ingest via the API
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"all": true}'

# Monitor progress via stats
watch -n 5 'curl -s http://localhost:8000/stats | python3 -m json.tool'
```

Ingest time depends on collection size and hardware. Rough estimates:

| Collection Size | CPU-Only | With GPU |
|----------------|----------|----------|
| 10,000 records | ~2 min | ~30 sec |
| 100,000 records | ~20 min | ~5 min |
| 1,000,000 records | ~3 hours | ~45 min |

---

## Troubleshooting

### RAG service won't start

Check that the database is reachable and the `rag` schema exists:

```bash
docker compose -f docker-compose.rag.yml logs rag
psql -h $DB_HOST -U evergreen -d evergreen -c "SELECT 1 FROM rag.config LIMIT 1;"
```

### Embedding service errors

Verify the Ollama model is pulled and responding:

```bash
docker compose -f docker-compose.rag.yml exec ollama ollama list
curl http://localhost:11434/api/tags
```

### Search returns no results

Check that embeddings exist in the database:

```bash
curl -s http://localhost:8000/stats | python3 -m json.tool
```

If `total_embeddings` is 0, run the ingest pipeline first.
