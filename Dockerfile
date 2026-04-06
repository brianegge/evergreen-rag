FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .


FROM python:3.11-slim

LABEL org.opencontainers.image.title="Evergreen RAG"
LABEL org.opencontainers.image.description="Retrieval-Augmented Generation for Evergreen ILS Search"
LABEL org.opencontainers.image.source="https://github.com/brianegge/evergreen-rag"
LABEL org.opencontainers.image.licenses="GPL-2.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false rag

COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY src/ src/
COPY scripts/init-db.sql scripts/init-db.sql

USER rag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "evergreen_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
