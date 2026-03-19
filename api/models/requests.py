"""
Pydantic request models for the Epstein RAG REST API.

All fields are optional and fall back to:
1. Explicit API request value
2. config/properties.yaml
3. Environment variables
4. Hard-coded defaults
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """
    Request body for POST /api/ingest.

    Triggers the full pipeline: ETL → Postgres → Ollama embeddings → pgvector.

    Every field is optional – omitted fields are resolved at runtime from
    config/properties.yaml, environment variables, or built-in defaults.
    """

    repository: Optional[str] = Field(
        default=None,
        description=(
            "Absolute or relative path to the documents repository. "
            "Overrides EPSTEIN_DOCS_PATH env var and ingestion.repository_path "
            "in properties.yaml. Defaults to ./epstein-documents."
        ),
        examples=["./epstein-documents", "/data/epstein-docs"],
    )
    state_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to the ingestion-state JSON file used for incremental processing. "
            "Defaults to ./data/ingestion_state.json."
        ),
        examples=["./data/ingestion_state.json"],
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of characters per chunk when splitting documents. "
            "Defaults to the value in properties.yaml (800)."
        ),
        examples=[800],
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Number of overlapping characters between consecutive chunks. "
            "Defaults to the value in properties.yaml (200)."
        ),
        examples=[200],
    )

    # ------------------------------------------------------------------
    # Postgres connection overrides
    # ------------------------------------------------------------------
    db_host: Optional[str] = Field(
        default=None,
        description="Postgres host. Defaults to localhost.",
        examples=["localhost"],
    )
    db_port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Postgres port. Defaults to 5432.",
        examples=[5432],
    )
    db_user: Optional[str] = Field(
        default=None,
        description="Postgres user. Defaults to postgres.",
        examples=["postgres"],
    )
    db_password: Optional[str] = Field(
        default=None,
        description="Postgres password. Defaults to postgres.",
        examples=["postgres"],
    )
    db_name: Optional[str] = Field(
        default=None,
        description="Postgres database name. Defaults to epstien_files_db.",
        examples=["epstien_files_db"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "repository": "./epstein-documents",
                    "chunk_size": 800,
                    "chunk_overlap": 200,
                    "db_host": "localhost",
                    "db_port": 5432,
                    "db_user": "postgres",
                    "db_password": "postgres",
                    "db_name": "epstien_files_db",
                }
            ]
        }
    }
