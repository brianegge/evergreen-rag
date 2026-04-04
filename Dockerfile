FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

FROM base AS dev
RUN pip install --no-cache-dir ".[dev]"
COPY . .

FROM base AS prod
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "evergreen_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
