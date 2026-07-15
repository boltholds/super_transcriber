from pathlib import Path

import pytest

from super_transcriber.api.job_store import InMemoryJobStore, JobNotFoundError
from super_transcriber.api.models import JobStatus, TranscriptionOptions


def test_job_lifecycle(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    job = store.create(
        source_filename="meeting.wav",
        input_path=tmp_path / "meeting.wav",
        work_dir=tmp_path,
        options=TranscriptionOptions(),
    )

    assert job.status == JobStatus.QUEUED
    assert store.mark_processing(job.id).status == JobStatus.PROCESSING
    assert store.mark_completed(job.id, []).status == JobStatus.COMPLETED


def test_missing_job_raises() -> None:
    with pytest.raises(JobNotFoundError):
        InMemoryJobStore().get("missing")


def test_options_reject_ambiguous_speaker_configuration() -> None:
    options = TranscriptionOptions(speakers=2, min_speakers=1)
    with pytest.raises(ValueError, match="either speakers"):
        options.validate_speaker_bounds()
