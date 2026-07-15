from __future__ import annotations

import weakref
from pathlib import Path
from unittest.mock import Mock

import pytest

from super_transcriber.services import whisperx_service as service_module
from super_transcriber.services.whisperx_service import WhisperXService


class FakeAsrModel:
    def transcribe(self, *_args: object, **_kwargs: object) -> dict:
        return {"segments": []}


class FakeDiarizationModel:
    def __call__(self, *_args: object, **_kwargs: object) -> list:
        return []


def test_transcribe_releases_gpu_models_before_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_refs: list[weakref.ReferenceType[object]] = []

    def asr_factory(*_args: object, **_kwargs: object) -> FakeAsrModel:
        model = FakeAsrModel()
        model_refs.append(weakref.ref(model))
        return model

    def align_factory(*_args: object, **_kwargs: object) -> tuple[object, dict]:
        model = FakeAsrModel()
        model_refs.append(weakref.ref(model))
        return model, {}

    def diarization_factory(*_args: object, **_kwargs: object) -> FakeDiarizationModel:
        model = FakeDiarizationModel()
        model_refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(service_module.whisperx, "load_audio", lambda _path: [])
    monkeypatch.setattr(service_module.whisperx, "load_model", asr_factory)
    monkeypatch.setattr(service_module.whisperx, "load_align_model", align_factory)
    monkeypatch.setattr(
        service_module.whisperx,
        "align",
        lambda *_args, **_kwargs: {"segments": []},
    )
    monkeypatch.setattr(
        service_module,
        "DiarizationPipeline",
        diarization_factory,
    )
    monkeypatch.setattr(
        service_module.whisperx,
        "assign_word_speakers",
        lambda _diarization, result: result,
    )

    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake")
    service = WhisperXService(hf_token="", device="cpu")
    service._cleanup_cuda = Mock()

    transcript = service.transcribe(audio_path)

    assert transcript.segments == []
    assert all(reference() is None for reference in model_refs)
    assert service._cleanup_cuda.call_count == 4


def test_asr_model_is_released_when_transcription_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_ref: weakref.ReferenceType[object] | None = None

    class FailingAsrModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> dict:
            raise RuntimeError("ASR failed")

    def asr_factory(*_args: object, **_kwargs: object) -> FailingAsrModel:
        nonlocal model_ref
        model = FailingAsrModel()
        model_ref = weakref.ref(model)
        return model

    monkeypatch.setattr(service_module.whisperx, "load_audio", lambda _path: [])
    monkeypatch.setattr(service_module.whisperx, "load_model", asr_factory)

    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake")
    service = WhisperXService(hf_token="", device="cpu")
    service._cleanup_cuda = Mock()

    with pytest.raises(RuntimeError, match="ASR failed"):
        service.transcribe(audio_path)

    assert model_ref is not None
    assert model_ref() is None
    service._cleanup_cuda.assert_called_once()
