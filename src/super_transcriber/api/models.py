from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactKind(StrEnum):
    TRANSCRIPT_JSON = "transcript_json"
    TRANSCRIPT_MARKDOWN = "transcript_markdown"
    ANALYSIS_JSON = "analysis_json"
    ANALYSIS_MARKDOWN = "analysis_markdown"


class TranscriptionOptions(BaseModel):
    language: str = "ru"
    model: str | None = None
    speakers: int | None = Field(default=None, ge=1, le=50)
    min_speakers: int | None = Field(default=None, ge=1, le=50)
    max_speakers: int | None = Field(default=None, ge=1, le=50)
    batch_size: int = Field(default=8, ge=1, le=128)
    analyze: bool = False

    def validate_speaker_bounds(self) -> None:
        if self.speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError(
                "Use either speakers or min_speakers/max_speakers, not both."
            )
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers must not exceed max_speakers.")


class JobArtifact(BaseModel):
    kind: ArtifactKind
    filename: str
    download_url: str


class TranscriptionJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.QUEUED
    source_filename: str
    options: TranscriptionOptions
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    artifacts: list[JobArtifact] = Field(default_factory=list)
    input_path: Path = Field(exclude=True)
    work_dir: Path = Field(exclude=True)


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    source_filename: str
    options: TranscriptionOptions
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    artifacts: list[JobArtifact]

    @classmethod
    def from_job(cls, job: TranscriptionJob) -> "JobResponse":
        return cls.model_validate(job.model_dump())
