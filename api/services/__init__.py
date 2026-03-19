"""
Services package – shared singletons and utilities for the API layer.
"""

from .job_manager import Job, JobManager, JobStatus, job_manager

__all__ = [
    "Job",
    "JobManager",
    "JobStatus",
    "job_manager",
]
