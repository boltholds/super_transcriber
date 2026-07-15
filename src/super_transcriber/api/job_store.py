from __future__ import annotations

from pathlib import Path
from threading import RLock

from super_transcriber.api.models import (
    JobArtifact,
    JobStatus,
    TranscriptionJob,
    TranscriptionOptions,
    utc_now,
)


class JobNotFoundError(KeyError):
    pass


class InMemoryJobStore:
    """Process-local MVP store behind a replaceable interface."""

    def __init__(self) -> None:
        self._jobs: dict[str, TranscriptionJob] = {}
        self._lock = RLock()

    def create(
        self,
        *,
        source_filename: str,
        input_path: Path,
        work_dir: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionJob:
        job = TranscriptionJob(
            source_filename=source_filename,
            input_path=input_path,
            work_dir=work_dir,
            options=options,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> TranscriptionJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.model_copy(deep=True)

    def mark_processing(self, job_id: str) -> TranscriptionJob:
        with self._lock:
            job = self._require(job_id)
            job.status = JobStatus.PROCESSING
            job.started_at = utc_now()
            job.error = None
            return job.model_copy(deep=True)

    def set_paths(
        self, job_id: str, *, input_path: Path, work_dir: Path
    ) -> TranscriptionJob:
        with self._lock:
            job = self._require(job_id)
            if job.status != JobStatus.QUEUED:
                raise RuntimeError("Job paths can only be changed while queued.")
            job.input_path = input_path
            job.work_dir = work_dir
            return job.model_copy(deep=True)

    def mark_completed(
        self, job_id: str, artifacts: list[JobArtifact]
    ) -> TranscriptionJob:
        with self._lock:
            job = self._require(job_id)
            job.status = JobStatus.COMPLETED
            job.completed_at = utc_now()
            job.artifacts = artifacts
            job.error = None
            return job.model_copy(deep=True)

    def mark_failed(self, job_id: str, error: str) -> TranscriptionJob:
        with self._lock:
            job = self._require(job_id)
            job.status = JobStatus.FAILED
            job.completed_at = utc_now()
            job.error = error
            return job.model_copy(deep=True)

    def _require(self, job_id: str) -> TranscriptionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job
