"""
API package – FastAPI application and routers for the Epstein RAG pipeline.

Exposes:
- create_app: factory function that builds the configured FastAPI instance.
- app: module-level application instance for use with uvicorn / gunicorn.
"""

from .app import app, create_app

__all__ = [
    "app",
    "create_app",
]
