import json
from pathlib import Path

from super_transcriber.analysis.graph import TranscriptAnalysisGraph
from super_transcriber.models import (
    ActionItem,
    ConversationAnalysis,
    Participant,
    TimelineItem,
)


class TranscriptAnalysisService:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_chunk_chars: int = 8_000,
        temperature: float = 0.1,
    ) -> None:
        self.graph = TranscriptAnalysisGraph(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_chunk_chars=max_chunk_chars,
            temperature=temperature,
        )

    def analyze_file(
        self,
        transcript_path: Path,
        speaker_map_path: Path | None = None,
    ) -> ConversationAnalysis:
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        speaker_map = self._load_speaker_map(speaker_map_path)

        final = self.graph.invoke(
            transcript_path=transcript_path,
            speaker_map=speaker_map,
        )

        return ConversationAnalysis(
            transcript_path=str(transcript_path),
            summary=final.summary,
            participants=[
                Participant(
                    speaker=participant.speaker,
                    assumed_name=participant.assumed_name,
                    role=participant.role,
                    notes=participant.notes,
                )
                for participant in final.participants
            ],
            topics=final.topics,
            decisions=final.decisions,
            action_items=[
                ActionItem(
                    owner=item.owner,
                    task=item.task,
                    deadline=item.deadline,
                    evidence=item.evidence,
                )
                for item in final.action_items
            ],
            open_questions=final.open_questions,
            risks=final.risks,
            timeline=[
                TimelineItem(
                    start=item.start,
                    end=item.end,
                    title=item.title,
                    description=item.description,
                )
                for item in final.timeline
            ],
        )

    def _load_speaker_map(
        self,
        speaker_map_path: Path | None,
    ) -> dict[str, str]:
        if speaker_map_path is None:
            return {}

        if not speaker_map_path.exists():
            raise FileNotFoundError(f"Speaker map file not found: {speaker_map_path}")

        data = json.loads(speaker_map_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError("Speaker map must be a JSON object")

        result: dict[str, str] = {}

        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Speaker map keys and values must be strings")

            result[key] = value

        return result