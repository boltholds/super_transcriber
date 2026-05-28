import gc
from pathlib import Path

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from super_transcriber.models import Transcript, TranscriptSegment, TranscriptWord


class WhisperXService:
    def __init__(
        self,
        hf_token: str,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.hf_token = hf_token
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.compute_type = self._resolve_compute_type(compute_type)

    def transcribe(
        self,
        audio_path: Path,
        language: str = "ru",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 8,
    ) -> Transcript:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = whisperx.load_audio(str(audio_path))

        print(f"Device: {self.device}")
        print(f"Compute type: {self.compute_type}")
        print(f"Model: {self.model_name}")
        print("Loading WhisperX ASR model...")

        model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=language,
        )

        print("Transcribing audio...")

        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
        )

        self._cleanup(model)

        print("Loading alignment model...")

        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )

        print("Aligning words...")

        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        self._cleanup(align_model)

        print("Running diarization...")

        diarize_model = DiarizationPipeline(
            token=self.hf_token,
            device=self.device,
        )

        diarize_kwargs: dict = {}

        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers

        diarize_segments = diarize_model(
            audio,
            **diarize_kwargs,
        )

        print("Assigning speakers...")

        result = whisperx.assign_word_speakers(
            diarize_segments,
            result,
        )

        segments = self._parse_segments(result)

        return Transcript(
            audio_path=str(audio_path),
            language=language,
            model=self.model_name,
            device=self.device,
            segments=segments,
        )

    def _parse_segments(self, raw_result: dict) -> list[TranscriptSegment]:
        parsed: list[TranscriptSegment] = []

        for segment in raw_result.get("segments", []):
            words = []

            for word in segment.get("words", []):
                words.append(
                    TranscriptWord(
                        word=word.get("word", ""),
                        start=word.get("start"),
                        end=word.get("end"),
                        speaker=word.get("speaker"),
                        score=word.get("score"),
                    )
                )

            parsed.append(
                TranscriptSegment(
                    start=segment.get("start", 0.0),
                    end=segment.get("end", 0.0),
                    speaker=segment.get("speaker", "UNKNOWN"),
                    text=segment.get("text", "").strip(),
                    words=words,
                )
            )

        return parsed

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_compute_type(self, compute_type: str) -> str:
        if compute_type != "auto":
            return compute_type

        if self.device == "cuda":
            return "float16"

        return "int8"

    def _cleanup(self, obj: object) -> None:
        del obj
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()