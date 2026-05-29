from typing import TypedDict

from super_transcriber.analysis.schemas import (
    ChunkAnalysisSchema,
    ChunkPromptInput,
    FinalAnalysisSchema,
)
from super_transcriber.models import Transcript


class AnalysisGraphState(TypedDict, total=False):
    transcript_path: str
    speaker_map: dict[str, str]
    transcript: Transcript
    chunks: list[ChunkPromptInput]
    chunk_analyses: list[ChunkAnalysisSchema]
    final_analysis: FinalAnalysisSchema