from fastapi.testclient import TestClient

from super_transcriber.api.app import _safe_filename, create_app
from super_transcriber.api.job_store import InMemoryJobStore
from super_transcriber.settings import Settings


class FakeRunner:
    def __init__(self) -> None:
        self.jobs: list[str] = []

    def run(self, job_id: str) -> None:
        self.jobs.append(job_id)


def test_upload_creates_authenticated_job(tmp_path) -> None:
    store = InMemoryJobStore()
    runner = FakeRunner()
    app = create_app(
        settings=Settings(api_key="secret", api_data_dir=tmp_path),
        store=store,
        runner=runner,
    )
    client = TestClient(app)

    unauthorized = client.post(
        "/v1/transcriptions",
        files={"audio": ("meeting.wav", b"audio", "audio/wav")},
    )
    assert unauthorized.status_code == 401

    response = client.post(
        "/v1/transcriptions",
        headers={"X-API-Key": "secret"},
        files={"audio": ("meeting.wav", b"audio", "audio/wav")},
        data={"language": "ru", "min_speakers": "1", "max_speakers": "4"},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_filename"] == "meeting.wav"
    assert runner.jobs == [payload["id"]]


def test_rejects_empty_audio(tmp_path) -> None:
    app = create_app(
        settings=Settings(api_data_dir=tmp_path),
        store=InMemoryJobStore(),
        runner=FakeRunner(),
    )
    response = TestClient(app).post(
        "/v1/transcriptions",
        files={"audio": ("meeting.wav", b"", "audio/wav")},
    )
    assert response.status_code == 422


def test_sanitizes_dot_filename() -> None:
    assert _safe_filename("..") == "meeting.wav"
