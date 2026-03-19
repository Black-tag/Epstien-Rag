"""
Job Manager – in-memory tracking for background ingestion and embedding jobs.

Each job goes through the following lifecycle:
    PENDING → RUNNING → COMPLETED | FAILED

Jobs are keyed by a UUID and live for the lifetime of the server process.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """
    Represents a single background job (ingest or embed).
    """

    def __init__(self, job_id: str, job_type: str) -> None:
        self.job_id: str = job_id
        self.job_type: str = job_type
        self.status: JobStatus = JobStatus.PENDING
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = datetime.now(timezone.utc)
        self.result: Optional[Any] = None
        self.error: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Job id={self.job_id!r} type={self.job_type!r} status={self.status.value!r}>"
        )


class JobManager:
    """
    Thread-safe (GIL-protected for CPython) in-memory store for Job objects.

    All mutating operations go through this class so there is a single
    authoritative source of truth for job state.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(self, job_type: str) -> Job:
        """
        Allocate a new job with a fresh UUID and return it.
        The job starts in PENDING state.
        """
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, job_type=job_type)
        self._jobs[job_id] = job
        logger.debug("Created job %s (type=%s)", job_id, job_type)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Return the Job for the given id, or None if it does not exist.
        """
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the status (and optionally the result / error) of an existing job.

        Unknown job_ids are silently ignored so that background tasks
        that outlive the job store do not crash the server.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(
                "update_job called for unknown job_id=%s; ignoring.", job_id
            )
            return

        job.status = status
        job.updated_at = datetime.now(timezone.utc)

        if result is not None:
            job.result = result

        if error is not None:
            job.error = error

        logger.debug(
            "Job %s updated → status=%s result=%s error=%s",
            job_id,
            status.value,
            result,
            error,
        )

    def list_jobs(self) -> List[Job]:
        """
        Return all tracked jobs in insertion order (Python 3.7+ dict guarantee).
        """
        return list(self._jobs.values())

    def delete_job(self, job_id: str) -> bool:
        """
        Remove a job from the store.  Returns True if it existed.
        Useful for future cleanup endpoints.
        """
        existed = job_id in self._jobs
        self._jobs.pop(job_id, None)
        return existed

    def __len__(self) -> int:  # pragma: no cover
        return len(self._jobs)


# ---------------------------------------------------------------------------
# Module-level singleton – import this everywhere instead of instantiating
# ---------------------------------------------------------------------------
job_manager = JobManager()
