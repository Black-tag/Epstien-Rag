"""
FastAPI application factory for the Epstein RAG API.

Usage
-----
From the project root:

    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Or via the dedicated entry-point script:

    python api_server.py
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models.responses import HealthResponse
from api.routers import ingest, jobs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

_TITLE = "Epstein RAG API"
_DESCRIPTION = """
## Overview

REST API for the **Epstein RAG** ingestion and embedding pipeline.

A single `POST /api/ingest` call drives the **entire pipeline**:

1. **ETL** – scan the document repository, parse every file, chunk and deduplicate
2. **Postgres** – upsert documents and chunks (embedding column starts as `NULL`)
3. **Ollama** – generate vector embeddings for every inserted chunk
4. **pgvector** – `UPDATE chunks SET embedding = …::vector` for each chunk

No external vector store is required – embeddings live in the `chunks` table
alongside the text, queryable via pgvector's HNSW cosine-similarity index.

All operations run as **background jobs** so every endpoint returns immediately
with a `job_id`.  Poll `GET /api/jobs/{job_id}` to track progress.

---

## Endpoints

| Method   | Path                   | Description                                              |
|----------|------------------------|----------------------------------------------------------|
| `POST`   | `/api/ingest`          | Run the full ETL + embedding pipeline                    |
| `GET`    | `/api/jobs`            | List all submitted jobs with their current status        |
| `GET`    | `/api/jobs/{job_id}`   | Poll a specific job – status, result, or error           |
| `DELETE` | `/api/jobs/{job_id}`   | Remove a finished job record from the in-memory store    |
| `GET`    | `/health`              | Liveness check                                           |

---

## Job Lifecycle

```
POST /api/ingest  ──►  202 { job_id, status: "pending" }
                                │
                                ▼  (background task starts)
GET /api/jobs/{id}  ──►  { status: "running" }
                                │
                                ▼
GET /api/jobs/{id}  ──►  { status: "completed",
                            result: {
                              documents_count: N,
                              chunks_count: N,
                              chunks_embedded: N
                            }}
                        or { status: "failed", error: "…" }
```

All request-body fields are **optional** – omitted values fall back to
`config/properties.yaml`, environment variables, or built-in defaults.
"""

_VERSION = "0.1.0"
_CONTACT = {
    "name": "Epstein RAG",
}
_LICENSE = {
    "name": "MIT",
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    Separated into a factory function so the app can be instantiated in
    tests or alternative entry-points without executing module-level side
    effects.
    """
    app = FastAPI(
        title=_TITLE,
        description=_DESCRIPTION,
        version=_VERSION,
        contact=_CONTACT,
        license_info=_LICENSE,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    # CORS – allow all origins by default (tighten in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request / response logging middleware
    @app.middleware("http")
    async def _log_requests(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "Unhandled exception during %s %s: %s",
                request.method,
                request.url.path,
                exc,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error."},
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    app.include_router(ingest.router)
    app.include_router(jobs.router)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Liveness check",
        description=(
            "Returns `200 OK` with `{ status: 'ok' }` when the API server is "
            "reachable and the application has started successfully.\n\n"
            "Does **not** probe Postgres, Ollama, or pgvector – use it purely "
            "as a liveness indicator."
        ),
    )
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=_VERSION)

    # ------------------------------------------------------------------
    # Startup / shutdown hooks
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def _on_startup() -> None:  # pragma: no cover
        logger.info(
            "Epstein RAG API v%s started.  Docs: /docs  OpenAPI: /openapi.json",
            _VERSION,
        )

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:  # pragma: no cover
        logger.info("Epstein RAG API shutting down.")

    return app


# ---------------------------------------------------------------------------
# Module-level app instance
# ---------------------------------------------------------------------------
# Allows:  uvicorn api.app:app --reload
# as well as the explicit entry-point in api_server.py.
# ---------------------------------------------------------------------------

app = create_app()
