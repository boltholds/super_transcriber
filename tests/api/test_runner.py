from pathlib import Path
from types import SimpleNamespace

from super_transcriber.api.job_store import InMemoryJobStore
from super_transcriber.api.models import ArtifactKind, JobStatus, TranscriptionOptions
from super_transcriber.api.runner import TranscriptionJobRunner
from super_transcriber.models import Transcript


class FakeTranscriber:
    def transcribe(self, **kwargs: object) -> Transcript:
        return Transcript(
            audio_path=str(kwargs["audio_path"]),
            language=str(kwargs["language"]),
            model="fake",
            device="cpu",
            segments=[],
        )


def test_runner_exports_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake")
    store = InMemoryJobStore()
    job = store.create(
        source_filename=audio.name,
        input_path=audio,
        work_dir=tmp_path,
        options=TranscriptionOptions(),
    )
    settings = SimpleNamespace(
        hf_token="",
        whisperx_model="small",
        whisperx_device="cpu",
        whisperx_compute_type="int8",
        llm_base_url="http://localhost",
        llm_api_key="local",
        llm_model="fake",
    )
    runner = TranscriptionJobRunner(
        settings=settings,
        store=store,
        transcriber_factory=lambda _: FakeTranscriber(),
    )

    runner.run(job.id)

    completed = store.get(job.id)
    assert completed.status == JobStatus.COMPLETED
    assert {item.kind for item in completed.artifacts} == {
        ArtifactKind.TRANSCRIPT_JSON,
        ArtifactKind.TRANSCRIPT_MARKDOWN,
    }
    assert (tmp_path / "meeting.transcript.json").is_file()
