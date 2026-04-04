"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from evergreen_rag.api.routes import router

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.generation.service import GenerationService
from evergreen_rag.search.vector_search import VectorSearch

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting up RAG service")
    app.state.embedding_service = EmbeddingService()
    app.state.vector_search = VectorSearch()
    app.state.vector_search.open()

    # Generation service is optional
    try:
        gen_service = GenerationService()
        if gen_service.health_check():
            app.state.generation_service = gen_service
            logger.info(
                "Generation service initialized (model=%s)",
                gen_service.model,
            )
        else:
            app.state.generation_service = None
            logger.info("Generation service unavailable, running without LLM")
    except Exception:
        app.state.generation_service = None
        logger.info(
            "Generation service not configured, running without LLM",
            exc_info=True,
        )

    logger.info("RAG service started")

    yield

    # Shutdown
    logger.info("Shutting down RAG service")
    app.state.vector_search.close()
    logger.info("RAG service stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Evergreen RAG",
        description="Retrieval-Augmented Generation for Evergreen ILS Search",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    # Serve static files (OPAC JS, staff client, search UI)
    if STATIC_DIR.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()
