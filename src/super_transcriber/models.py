from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    word: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    score: float | None = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    speaker: str = "UNKNOWN"
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)


class Transcript(BaseModel):
    audio_path: str
    language: str
    model: str
    device: str
    segments: list[TranscriptSegment]