from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

from super_transcriber.api.job_store import InMemoryJobStore
from super_transcriber.api.models import ArtifactKind, JobArtifact, TranscriptionJob
from super_transcriber.services.export_service import ExportService

if TYPE_CHECKING:
    from super_transcriber.services.analysis_service import TranscriptAnalysisService
    from super_transcriber.services.whisperx_service import WhisperXService
    from super_transcriber.settings import Settings

logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    def transcribe(self, **kwargs: Any) -> Any: ...


class Analyzer(Protocol):
    def analyze_file(self, transcript_path: Path) -> Any: ...


TranscriberFactory = Callable[[TranscriptionJob], Transcriber]
AnalyzerFactory = Callable[[TranscriptionJob], Analyzer]


class TranscriptionJobRunner:
    """Runs one GPU-heavy job at a time inside the MVP API process."""

    def __init__(
        self,
        *,
        settings: "Settings",
        store: InMemoryJobStore,
        transcriber_factory: TranscriberFactory | None = None,
        analyzer_factory: AnalyzerFactory | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self._gpu_lock = Lock()
        self._transcriber_factory = transcriber_factory or self._make_transcriber
        self._analyzer_factory = analyzer_factory or self._make_analyzer

    def run(self, job_id: str) -> None:
        try:
            with self._gpu_lock:
                self.store.mark_processing(job_id)
                job = self.store.get(job_id)
                artifacts = self._execute(job)
        except Exception as exc:
            logger.exception("Transcription job %s failed", job_id)
            self.store.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
            return

        self.store.mark_completed(job_id, artifacts)

    def _execute(self, job: TranscriptionJob) -> list[JobArtifact]:
        options = job.options
        transcriber = self._transcriber_factory(job)
        transcript = transcriber.transcribe(
            audio_path=job.input_path,
            language=options.language,
            num_speakers=options.speakers,
            min_speakers=options.min_speakers,
            max_speakers=options.max_speakers,
            batch_size=options.batch_size,
        )

        exporter = ExportService()
        stem = job.input_path.stem
        transcript_json = job.work_dir / f"{stem}.transcript.json"
        transcript_md = job.work_dir / f"{stem}.transcript.md"
        exporter.export_json(transcript, transcript_json)
        exporter.export_markdown(transcript, transcript_md)

        paths: list[tuple[ArtifactKind, Path]] = [
            (ArtifactKind.TRANSCRIPT_JSON, transcript_json),
            (ArtifactKind.TRANSCRIPT_MARKDOWN, transcript_md),
        ]

        if options.analyze:
            analyzer = self._analyzer_factory(job)
            analysis = analyzer.analyze_file(transcript_json)
            analysis_json = job.work_dir / f"{stem}.analysis.json"
            analysis_md = job.work_dir / f"{stem}.analysis.md"
            exporter.export_analysis_json(analysis, analysis_json)
            exporter.export_analysis_markdown(analysis, analysis_md)
            paths.extend(
                [
                    (ArtifactKind.ANALYSIS_JSON, analysis_json),
                    (ArtifactKind.ANALYSIS_MARKDOWN, analysis_md),
                ]
            )

        return [
            JobArtifact(
                kind=kind,
                filename=path.name,
                download_url=f"/v1/transcriptions/{job.id}/artifacts/{kind.value}",
            )
            for kind, path in paths
        ]

    def _make_transcriber(self, job: TranscriptionJob) -> "WhisperXService":
        from super_transcriber.services.whisperx_service import WhisperXService

        return WhisperXService(
            hf_token=self.settings.hf_token,
            model_name=job.options.model or self.settings.whisperx_model,
            device=self.settings.whisperx_device,
            compute_type=self.settings.whisperx_compute_type,
        )

    def _make_analyzer(self, job: TranscriptionJob) -> "TranscriptAnalysisService":
        from super_transcriber.services.analysis_service import (
            TranscriptAnalysisService,
        )

        return TranscriptAnalysisService(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
        )
