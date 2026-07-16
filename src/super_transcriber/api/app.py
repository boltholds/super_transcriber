from __future__ import annotations

import re
from pathlib import Path
from secrets import compare_digest
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from super_transcriber.api.job_store import InMemoryJobStore, JobNotFoundError
from super_transcriber.api.models import ArtifactKind, JobResponse, TranscriptionOptions
from super_transcriber.api.runner import TranscriptionJobRunner
from super_transcriber.settings import Settings, get_settings

SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(filename: str | None) -> str:
    cleaned = SAFE_FILENAME.sub("_", Path(filename or "meeting.wav").name)
    cleaned = cleaned[:180]
    if cleaned in {"", ".", ".."}:
        return "meeting.wav"
    return cleaned


def create_app(
    *,
    settings: Settings | None = None,
    store: InMemoryJobStore | None = None,
    runner: TranscriptionJobRunner | None = None,
) -> FastAPI:
    config = settings or get_settings()
    job_store = store or InMemoryJobStore()
    job_runner = runner or TranscriptionJobRunner(settings=config, store=job_store)

    app = FastAPI(title="Super Transcriber API", version="0.1.0")
    app.state.settings = config
    app.state.job_store = job_store
    app.state.job_runner = job_runner

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        valid = bool(x_api_key) and compare_digest(x_api_key or "", config.api_key)
        if config.api_key and not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/transcriptions",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_transcription(
        background_tasks: BackgroundTasks,
        audio: UploadFile = File(...),
        language: str = Form("ru"),
        model: str | None = Form(None),
        speakers: int | None = Form(None),
        min_speakers: int | None = Form(None),
        max_speakers: int | None = Form(None),
        batch_size: int = Form(8),
        analyze: bool = Form(False),
    ) -> JobResponse:
        options = TranscriptionOptions(
            language=language,
            model=model,
            speakers=speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            batch_size=batch_size,
            analyze=analyze,
        )
        try:
            options.validate_speaker_bounds()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        filename = _safe_filename(audio.filename)
        pending_dir = config.api_data_dir / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = pending_dir / f"{uuid4()}-{filename}"
        written = 0

        try:
            with temporary_path.open("wb") as destination:
                while chunk := await audio.read(1024 * 1024):
                    written += len(chunk)
                    if written > config.api_max_upload_mb * 1024 * 1024:
                        raise HTTPException(
                            status_code=413, detail="Audio file is too large."
                        )
                    destination.write(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            await audio.close()

        if written == 0:
            temporary_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="Audio file is empty.")

        job = job_store.create(
            source_filename=filename,
            input_path=temporary_path,
            work_dir=pending_dir,
            options=options,
        )
        work_dir = config.api_data_dir / "jobs" / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / filename
        temporary_path.replace(input_path)

        job_store.set_paths(job.id, input_path=input_path, work_dir=work_dir)

        background_tasks.add_task(job_runner.run, job.id)
        return JobResponse.from_job(job_store.get(job.id))

    @app.get(
        "/v1/transcriptions/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_transcription(job_id: str) -> JobResponse:
        try:
            return JobResponse.from_job(job_store.get(job_id))
        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Transcription job not found."
            ) from exc

    @app.get(
        "/v1/transcriptions/{job_id}/artifacts/{kind}",
        dependencies=[Depends(require_api_key)],
    )
    def get_artifact(job_id: str, kind: ArtifactKind) -> FileResponse:
        try:
            job = job_store.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Transcription job not found."
            ) from exc

        artifact = next((item for item in job.artifacts if item.kind == kind), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact is not available.")

        path = job.work_dir / artifact.filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file is missing.")
        return FileResponse(path, filename=artifact.filename)

    return app


app = create_app()
