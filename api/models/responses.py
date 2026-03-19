"""
Pydantic response models for the Epstein RAG REST API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from api.services.job_manager import JobStatus


class JobResponse(BaseModel):
    """
    Immediate response returned when a job is accepted (HTTP 202).
    Use the job_id to poll GET /api/jobs/{job_id} for the final result.
    """

    job_id: str = Field(description="Unique identifier for the submitted job.")
    status: JobStatus = Field(description="Current lifecycle status of the job.")
    message: str = Field(description="Human-readable summary of the acceptance.")
    result: Optional[Any] = Field(
        default=None,
        description="Final result payload once the job reaches COMPLETED status.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the job reached FAILED status.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "status": "pending",
                    "message": "Ingestion job accepted and queued.",
                    "result": None,
                    "error": None,
                }
            ]
        }
    }


class IngestResult(BaseModel):
    """
    Payload stored in Job.result upon successful completion of the full
    ingestion pipeline (ETL → Postgres → Ollama → pgvector).
    """

    documents_count: int = Field(
        description="Number of documents parsed and upserted to Postgres."
    )
    chunks_count: int = Field(
        description="Number of text chunks inserted into Postgres."
    )
    chunks_embedded: int = Field(
        description=(
            "Number of chunks whose embedding vector was written to "
            "the chunks.embedding column via pgvector."
        )
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents_count": 42,
                    "chunks_count": 1380,
                    "chunks_embedded": 1380,
                }
            ]
        }
    }


class JobDetail(BaseModel):
    """
    Full detail of a tracked job, returned by GET /api/jobs and
    GET /api/jobs/{job_id}.
    """

    job_id: str = Field(description="Unique job identifier (UUID).")
    job_type: str = Field(
        description="Type of job – currently always 'ingest'.",
        examples=["ingest"],
    )
    status: JobStatus = Field(description="Current lifecycle status.")
    created_at: datetime = Field(description="UTC timestamp when the job was created.")
    updated_at: datetime = Field(
        description="UTC timestamp of the last status transition."
    )
    result: Optional[Any] = Field(
        default=None,
        description=(
            "Result payload once the job is COMPLETED. "
            "Shape matches IngestResult: documents_count, chunks_count, chunks_embedded."
        ),
    )
    error: Optional[str] = Field(
        default=None,
        description="Error detail if the job failed.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "job_type": "ingest",
                    "status": "completed",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:35:42Z",
                    "result": {
                        "documents_count": 42,
                        "chunks_count": 1380,
                        "chunks_embedded": 1380,
                    },
                    "error": None,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """
    Response model for GET /health.
    """

    status: str = Field(description="Service health status.", examples=["ok"])
    version: str = Field(
        description="API version string.", examples=["0.1.0"]
    )

    model_config = {
        "json_schema_extra": {"examples": [{"status": "ok", "version": "0.1.0"}]}
    }
