"""
api_server.py – Entry-point for the Epstein RAG REST API server.

Run with:
    python api_server.py

Or directly via uvicorn (auto-reload for development):
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Or via the project script (if installed with pip / uv):
    epstein-rag-api
"""

from __future__ import annotations

import logging
import os

import uvicorn

from api.app import app  # re-export so uvicorn can find it as "api_server:app"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

__all__ = ["app"]


def main() -> None:
    """
    Start the uvicorn server with configuration pulled from environment
    variables or sensible defaults.

    Environment variables
    ---------------------
    API_HOST        Bind host   (default: 0.0.0.0)
    API_PORT        Bind port   (default: 8000)
    API_RELOAD      Enable hot-reload – set to "true" for development
                    (default: false)
    API_LOG_LEVEL   Uvicorn log level: debug | info | warning | error
                    (default: info)
    API_WORKERS     Number of worker processes – ignored when reload=True
                    (default: 1)
    """
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() in ("1", "true", "yes")
    log_level: str = os.getenv("API_LOG_LEVEL", "info").lower()
    workers: int = int(os.getenv("API_WORKERS", "1"))

    logger.info(
        "Starting Epstein RAG API on %s:%d  (reload=%s, workers=%d, log_level=%s)",
        host,
        port,
        reload,
        workers,
        log_level,
    )

    uvicorn.run(
        # Use the import-string form so reload mode can re-import the module.
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        # workers > 1 is incompatible with reload; uvicorn ignores it when
        # reload=True so we pass it unconditionally.
        workers=workers,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
