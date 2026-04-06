# Dev Environment Reorganization Plan

## Context

The current dev environment runs everything on a single Mac Mini (`openclaw.home`) in the DMZ — PostgreSQL (pgvector), Evergreen ILS (with its own internal Postgres), Ollama, and the RAG service. With 100k+ MARC records, the 40GB podman VM disk filled up. The Mac has only 228GB total with limited free space. Moving the databases to the internal Ubuntu 24 server (`ubuntu24`) frees up disk on the Mac and puts storage-heavy services on a machine with more room, while keeping GPU-dependent services (Ollama) and web-facing services (Evergreen OPAC, RAG API) on the DMZ Mac.

## Target Architecture

```
DMZ (openclaw.home - Mac Mini)          Internal LAN (ubuntu24)
+----------------------------------+    +----------------------------------+
|  Ollama (native, port 11434)     |    |  PostgreSQL 16 + pgvector        |
|    - nomic-embed-text            |    |    - DB: evergreen_rag           |
|    - qwen2.5:3b / 7b            |    |      (rag schema + biblio)       |
|                                  |    |    - DB: evergreen               |
|  RAG Service (port 8000)         |    |      (full Evergreen schema)     |
|    - FastAPI/Uvicorn             |    |    - Port: 5432                  |
|    - Connects to ubuntu24:5432   |    +----------------------------------+
|    - Connects to localhost:11434 |
|                                  |
|  Evergreen ILS (podman)          |
|    - Apache/OPAC (ports 80/443)  |
|    - OpenSRF services            |
|    - Connects to ubuntu24:5432   |
|    - No internal PostgreSQL      |
+----------------------------------+

EdgeRouter X: Allow openclaw -> ubuntu24:5432 (TCP)
```

## Step 1: Set up PostgreSQL 16 + pgvector on ubuntu24

Run pgvector as a container with persistent storage:

```bash
# On ubuntu24 — bind-mount to /mnt/storage (7.2TB free) instead of
# a named volume, which would land on / (only 32GB free).
mkdir -p /mnt/storage/pgdata

podman run -d --name pgvector \
  -e POSTGRES_USER=evergreen \
  -e POSTGRES_PASSWORD=<strong-password> \
  -p 5432:5432 \
  -v /mnt/storage/pgdata:/var/lib/postgresql/data:Z \
  --restart=always \
  pgvector/pgvector:pg16
```

Create both databases:

```bash
podman exec -it pgvector psql -U evergreen -c "CREATE DATABASE evergreen_rag;"
```

Apply the RAG schema to `evergreen_rag`:

```bash
# Copy init-db.sql from the evergreen-rag project
psql -h localhost -U evergreen evergreen_rag < scripts/init-db.sql
```

The `evergreen` database will be created during the Evergreen DB migration (Step 4).

## Step 2: Configure pg_hba.conf and postgresql.conf

Allow remote connections from openclaw:

```bash
# Find the pg_hba.conf inside the container
podman exec pgvector bash -c "echo 'host all evergreen <openclaw-ip>/32 scram-sha-256' >> /var/lib/postgresql/data/pg_hba.conf"

# Ensure postgresql.conf listens on all interfaces
podman exec pgvector bash -c "echo \"listen_addresses = '*'\" >> /var/lib/postgresql/data/postgresql.conf"

# Restart to apply
podman restart pgvector
```

## Step 3: EdgeRouter X firewall rule

Allow TCP from openclaw (DMZ) to ubuntu24 (internal LAN) on port 5432 only:

```
configure
set firewall name DMZ_TO_INTERNAL rule 10 action accept
set firewall name DMZ_TO_INTERNAL rule 10 protocol tcp
set firewall name DMZ_TO_INTERNAL rule 10 destination address <ubuntu24-ip>
set firewall name DMZ_TO_INTERNAL rule 10 destination port 5432
set firewall name DMZ_TO_INTERNAL rule 10 source address <openclaw-ip>
set firewall name DMZ_TO_INTERNAL rule 10 description "Allow PostgreSQL from openclaw to ubuntu24"
commit
save
```

Verify from openclaw:

```bash
psql -h <ubuntu24-ip> -U evergreen evergreen_rag -c "SELECT 1"
```

## Step 4: Migrate existing data

### RAG database (100k records + 100k embeddings)

From openclaw:

```bash
# Dump from the local pgvector container
pg_dump -h localhost -U evergreen -d evergreen | \
  psql -h <ubuntu24-ip> -U evergreen -d evergreen_rag
```

Verify:

```bash
psql -h <ubuntu24-ip> -U evergreen evergreen_rag -c "
  SELECT COUNT(*) AS records FROM biblio.record_entry;
  SELECT COUNT(*) AS embeddings FROM rag.biblio_embedding;
"
```

Expected: ~100,129 records, ~100,029 embeddings.

### Evergreen database

From openclaw, inside the Evergreen container:

```bash
# Dump Evergreen's internal database
podman exec evergreen-dev bash -c \
  "su - evergreen -c 'pg_dump evergreen'" > /tmp/evergreen_dump.sql

# Restore to ubuntu24
# First create the evergreen DB on ubuntu24 if not exists
psql -h <ubuntu24-ip> -U evergreen postgres -c "CREATE DATABASE evergreen OWNER evergreen;"

# Load the dump
psql -h <ubuntu24-ip> -U evergreen evergreen < /tmp/evergreen_dump.sql
```

This is the full Evergreen schema with 100k+ bib records and all search indexes.

## Step 5: Reconfigure Evergreen container for external PostgreSQL

The eg-docker restart playbook does too much at startup — it tries to recompile Evergreen
from source, rebuild Angular, and recreate the database. For external DB deployment, we
need a patched playbook that skips the build section (lines 125-298).

### Files created on openclaw

1. **`vars-runtime.yml`** — copy of `vars.yml` with external DB settings:
   ```yaml
   database_host: 192.168.254.35
   database_password: evergreen
   ```

2. **`evergreen_restart_services_patched.yml`** — the restart playbook with the
   build/compile section (lines 125-298) removed. This skips autoreconf, npm ci,
   ng build, and eg_db_config which would otherwise try to recreate the DB schema.

### Build the image

The image must be built from `rel_3_13` branch to match the 3.13.4 database schema.
The build `vars.yml` must keep `database_host: localhost` (uses internal DB during build).
The podman VM needs 8GB RAM for the Angular build.

```bash
podman machine stop
podman machine set --memory 8192
podman machine start

cd ~/dev/eg-docker/generic-dockerhub-dev
# vars.yml should have: evergreen_git_branch: rel_3_13, database_host: localhost
podman build -t evergreen-ils:3.13-arm64 .
```

### Start the container

```bash
# vars-runtime.yml has database_host: 192.168.254.35, database_password: evergreen
# The patched playbook skips the build/compile section and symlink-back-to-repo steps
podman run -d --name evergreen-dev \
  --hostname localhost \
  -p 8080:7080 -p 8443:7443 -p 2210:210 -p 6001:6001 \
  -v ~/dev/eg-docker/generic-dockerhub-dev/vars-runtime.yml:/egconfigs/vars.yml:ro \
  -v ~/dev/eg-docker/generic-dockerhub-dev/evergreen_restart_services_patched.yml:/egconfigs/evergreen_restart_services.yml:ro \
  localhost/evergreen-ils:3.13-arm64
```

After the playbook completes (~3 min), fix the DB host and restart services:

```bash
podman exec evergreen-dev bash -c '
  perl -pi -e "s|<host>localhost</host>|<host>192.168.254.35</host>|g" /openils/conf/opensrf.xml
  perl -pi -e "s|<pw>databasepassword</pw>|<pw>evergreen</pw>|g" /openils/conf/opensrf.xml
  su - opensrf -c "/openils/bin/osrf_control --localhost --restart-all"
  sleep 5
  su - opensrf -c "/openils/bin/autogen.sh"
'
podman exec evergreen-dev /etc/init.d/apache2 restart
```

### Key details

- **Port mapping**: Apache listens on 7080/7443 inside, mapped to 8080/8443 outside
- **`--hostname localhost`**: Required so OpenSRF settings server can resolve its own host config
- **Patched playbook**: Removes lines 125-298 (build section) and 522-643 (symlink-to-repo section).
  These sections try to recompile Evergreen from source on every startup, which fails because
  the git repo is cleaned after the initial build. The installed binaries/modules from the image are sufficient.
- **DB host fix**: The patched playbook skips `eg_db_config`, so opensrf.xml still has `localhost`.
  Must be fixed manually after startup with the sed commands above.
- **Build vars.yml** must keep `database_host: localhost` and `evergreen_git_branch: rel_3_13`

## Step 6: Update evergreen-rag configuration

### Option A: Run RAG service natively (recommended — avoids podman overhead)

Create a `.env` file in the project root:

```bash
# /Users/claw/dev/evergreen-rag/.env
DATABASE_URL=postgresql://evergreen:<strong-password>@<ubuntu24-ip>:5432/evergreen_rag
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
GENERATION_MODEL=qwen2.5:3b
```

Run directly:

```bash
cd ~/dev/evergreen-rag
source .env
.venv/bin/uvicorn evergreen_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Simplified docker-compose.yml

```yaml
services:
  rag:
    build:
      context: .
      target: dev
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    command: >
      uvicorn evergreen_rag.api.main:app
      --host 0.0.0.0 --port 8000 --reload
```

No `db` or `ollama` services needed.

### Files to update

- `docker-compose.yml` — remove `db` and `ollama` services, remove `pgdata` and `ollama_data` volumes
- `.env` — add with connection strings (already in `.gitignore`)
- `scripts/load_marc_bulk.py` — update `DATABASE_URL_DEFAULT` to match new location
- `scripts/init-db.sql` — no changes needed, just run against ubuntu24

## Step 7: Shrink podman VM

With databases gone, the podman VM only needs to host the Evergreen container:

```bash
# On openclaw
podman machine stop
podman machine set --disk-size 60
podman machine start
```

Clean up old volumes:

```bash
podman volume rm ollama_data
# Note: pgdata is now a bind-mount on ubuntu24:/mnt/storage/pgdata, not a podman volume
```

## Step 8: Security hardening

- [ ] Use a strong password for PostgreSQL (not `evergreen` or `databasepassword`)
- [ ] Store password in `.env` files only (never in committed config)
- [ ] EdgeRouter rule is narrow: only port 5432, only from openclaw's IP
- [ ] Consider TLS for PostgreSQL: set `sslmode=require` in connection strings
- [ ] The `.env` file is already in `.gitignore`

## Verification checklist

1. **DB connectivity from openclaw:**
   ```bash
   psql -h <ubuntu24-ip> -U evergreen evergreen_rag -c "SELECT COUNT(*) FROM rag.biblio_embedding"
   ```

2. **RAG service health:**
   ```bash
   curl http://localhost:8000/health
   # Should show: embedding_service: true, database: true, generation_service: true
   ```

3. **Semantic search:**
   ```bash
   curl -X POST http://localhost:8000/search \
     -H 'Content-Type: application/json' \
     -d '{"query": "science fiction", "limit": 5, "generate": true}'
   ```

4. **Auto-ingest (LISTEN/NOTIFY):**
   ```bash
   psql -h <ubuntu24-ip> -U evergreen evergreen_rag -c \
     "INSERT INTO biblio.record_entry (marc) VALUES ('<record><controlfield tag=\"001\">999999</controlfield></record>')"
   # Check RAG logs — should see "Auto-ingest: embedding 1 records"
   ```

5. **Evergreen OPAC:**
   - Browse to `https://openclaw.home:8443/eg/opac/home`
   - Search for a title from the SFPL records
   - Verify results appear

6. **Disk space:**
   ```bash
   # On openclaw
   df -h /
   podman machine ssh -- df -h /
   # Both should show comfortable free space
   ```

## Network diagram (with ports)

```
Internet
    |
    v
[EdgeRouter X]
    |
    +-- DMZ --------------------------+
    |   openclaw.home                 |
    |   :80/443  -> Evergreen OPAC    |
    |   :8000    -> RAG API           |
    |   :11434   -> Ollama (local)    |
    +------|---------------------------+
           | TCP 5432 (firewall rule)
    +------|---------------------------+
    |   ubuntu24                      |
    |   :5432 -> PostgreSQL 16        |
    |            + pgvector           |
    |            DB: evergreen_rag    |
    |            DB: evergreen        |
    +----------------------------------+
    Internal LAN
```
