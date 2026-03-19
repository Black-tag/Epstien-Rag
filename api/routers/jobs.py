"""
Jobs router – GET /api/jobs and GET /api/jobs/{job_id}

Provides polling endpoints so callers can track the lifecycle and result
of any background job (ingest or embed) that was previously submitted.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from api.models.responses import JobDetail
from api.services.job_manager import Job, JobStatus, job_manager

router = APIRouter(prefix="/api", tags=["Jobs"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_to_detail(job: Job) -> JobDetail:
    """
    Convert the internal Job dataclass into the Pydantic response model.
    """
    return JobDetail(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/jobs",
    response_model=List[JobDetail],
    summary="List all submitted jobs",
    description=(
        "Returns every job that has been submitted since the server started, "
        "in submission order. Each entry includes the current `status`, "
        "timestamps, and – once the job finishes – the `result` or `error` "
        "payload.\n\n"
        "Jobs are kept in-memory for the lifetime of the server process. "
        "Restarting the server clears the job list."
    ),
    responses={
        200: {
            "description": "List of all jobs (may be empty).",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "job_type": "ingest",
                            "status": "completed",
                            "created_at": "2024-01-15T10:30:00Z",
                            "updated_at": "2024-01-15T10:35:42Z",
                            "result": {"documents_count": 42, "chunks_count": 1380},
                            "error": None,
                        },
                        {
                            "job_id": "7cb92a11-1234-4321-abcd-9f8e7d6c5b4a",
                            "job_type": "embed",
                            "status": "running",
                            "created_at": "2024-01-15T10:36:00Z",
                            "updated_at": "2024-01-15T10:36:05Z",
                            "result": None,
                            "error": None,
                        },
                    ]
                }
            },
        }
    },
)
def list_jobs() -> List[JobDetail]:
    """
    Return all tracked jobs ordered by submission time.
    """
    jobs = job_manager.list_jobs()
    logger.debug("list_jobs: returning %d job(s).", len(jobs))
    return [_job_to_detail(j) for j in jobs]


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetail,
    summary="Get a single job's status and result",
    description=(
        "Poll this endpoint with the `job_id` returned by `POST /api/ingest` "
        "or `POST /api/embed` to check whether the job is still running and – "
        "once it finishes – to retrieve the result or error details.\n\n"
        "**Lifecycle transitions:**\n"
        "- `pending` → job queued, not yet started\n"
        "- `running` → job is actively processing\n"
        "- `completed` → job finished successfully; check `result` for counts\n"
        "- `failed` → job encountered an unrecoverable error; check `error` for details"
    ),
    responses={
        200: {
            "description": "Job found – returns current detail.",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "job_type": "ingest",
                        "status": "completed",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:35:42Z",
                        "result": {"documents_count": 42, "chunks_count": 1380},
                        "error": None,
                    }
                }
            },
        },
        404: {
            "description": "No job with the given job_id exists.",
            "content": {
                "application/json": {
                    "example": {"detail": "Job '3fa85f64-5717-4562-b3fc-2c963f66afa6' not found."}
                }
            },
        },
    },
)
def get_job(job_id: str) -> JobDetail:
    """
    Return the full detail for a single job, or 404 if it does not exist.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        logger.warning("get_job: job_id=%r not found.", job_id)
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    logger.debug("get_job: job_id=%r status=%s.", job_id, job.status.value)
    return _job_to_detail(job)


@router.delete(
    "/jobs/{job_id}",
    status_code=204,
    summary="Delete a job record",
    description=(
        "Remove a job entry from the in-memory store. "
        "This does **not** cancel a running job – it only removes the tracking record. "
        "Returns `204 No Content` on success, or `404` if the job does not exist."
    ),
    responses={
        204: {"description": "Job record deleted successfully."},
        404: {
            "description": "No job with the given job_id exists.",
            "content": {
                "application/json": {
                    "example": {"detail": "Job '3fa85f64-5717-4562-b3fc-2c963f66afa6' not found."}
                }
            },
        },
    },
)
def delete_job(job_id: str) -> None:
    """
    Delete a job record from the in-memory store.
    """
    deleted = job_manager.delete_job(job_id)
    if not deleted:
        logger.warning("delete_job: job_id=%r not found.", job_id)
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )
    logger.info("delete_job: job_id=%r removed.", job_id)
