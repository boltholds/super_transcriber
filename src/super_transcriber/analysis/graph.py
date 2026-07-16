import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from super_transcriber.analysis.prompt_renderer import PromptRenderer
from super_transcriber.analysis.schemas import (
    ChunkAnalysisSchema,
    ChunkPromptInput,
    FinalAnalysisSchema,
    MergePromptInput,
    TranscriptLine,
)
from super_transcriber.analysis.state import AnalysisGraphState
from super_transcriber.models import Transcript


class TranscriptAnalysisGraph:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_chunk_chars: int = 12_000,
        temperature: float = 0.1,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.renderer = PromptRenderer()

        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

        self.chunk_llm = self.llm.with_structured_output(ChunkAnalysisSchema)
        self.merge_llm = self.llm.with_structured_output(FinalAnalysisSchema)

        self.graph = self._build_graph()

    def invoke(
        self,
        transcript_path: Path,
        speaker_map: dict[str, str] | None = None,
    ) -> FinalAnalysisSchema:
        result = self.graph.invoke(
            {
                "transcript_path": str(transcript_path),
                "speaker_map": speaker_map or {},
            }
        )

        return result["final_analysis"]

    def _build_graph(self):
        builder = StateGraph(AnalysisGraphState)

        builder.add_node("load_transcript", self._load_transcript)
        builder.add_node("split_chunks", self._split_chunks)
        builder.add_node("analyze_chunks", self._analyze_chunks)
        builder.add_node("merge_analyses", self._merge_analyses)

        builder.add_edge(START, "load_transcript")
        builder.add_edge("load_transcript", "split_chunks")
        builder.add_edge("split_chunks", "analyze_chunks")
        builder.add_edge("analyze_chunks", "merge_analyses")
        builder.add_edge("merge_analyses", END)

        return builder.compile()

    def _load_transcript(self, state: AnalysisGraphState) -> AnalysisGraphState:
        transcript_path = Path(state["transcript_path"])

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        transcript = Transcript.model_validate_json(
            transcript_path.read_text(encoding="utf-8")
        )

        return {
            "transcript": transcript,
        }

    def _split_chunks(self, state: AnalysisGraphState) -> AnalysisGraphState:
        transcript = state["transcript"]
        speaker_map = state.get("speaker_map", {})

        return {
            "chunks": self._split_transcript(
                transcript=transcript,
                speaker_map=speaker_map,
            ),
        }

    def _analyze_chunks(self, state: AnalysisGraphState) -> AnalysisGraphState:
        chunks = state["chunks"]
        results: list[ChunkAnalysisSchema] = []

        for chunk in chunks:
            print(
                f"Analyzing chunk {chunk.chunk_index + 1}/{len(chunks)} "
                f"[{chunk.start} - {chunk.end}]"
            )

            system_prompt = self.renderer.render(
                "chunk_analysis.system.j2",
                {},
            )

            user_prompt = self.renderer.render(
                "chunk_analysis.user.j2",
                chunk,
            )

            result = self.chunk_llm.invoke(
                [
                    ("system", system_prompt),
                    ("user", user_prompt),
                ]
            )

            results.append(result)

        return {
            "chunk_analyses": results,
        }

    def _merge_analyses(self, state: AnalysisGraphState) -> AnalysisGraphState:
        chunk_analyses = state["chunk_analyses"]
        speaker_map = state.get("speaker_map", {})

        payload = [
            item.model_dump(mode="json")
            for item in chunk_analyses
        ]

        prompt_input = MergePromptInput(
            chunk_analyses_json=json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            speaker_map=speaker_map,
        )

        system_prompt = self.renderer.render(
            "merge_analysis.system.j2",
            {},
        )

        user_prompt = self.renderer.render(
            "merge_analysis.user.j2",
            prompt_input,
        )

        final_analysis = self.merge_llm.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

        return {
            "final_analysis": final_analysis,
        }

    def _split_transcript(
        self,
        transcript: Transcript,
        speaker_map: dict[str, str],
    ) -> list[ChunkPromptInput]:
        chunks: list[ChunkPromptInput] = []

        current_lines: list[TranscriptLine] = []
        current_chars = 0
        current_start: str | None = None
        current_end: str | None = None

        for segment in transcript.segments:
            text = segment.text.strip()
            if not text:
                continue

            start = self._format_time(segment.start)
            end = self._format_time(segment.end)

            line = TranscriptLine(
                speaker=segment.speaker,
                start=start,
                end=end,
                text=text,
            )

            rendered_line_length = len(
                f"{line.speaker} [{line.start} - {line.end}]: {line.text}"
            )

            if current_start is None:
                current_start = start

            if (
                current_chars + rendered_line_length > self.max_chunk_chars
                and current_lines
            ):
                chunks.append(
                    ChunkPromptInput(
                        chunk_index=len(chunks),
                        start=current_start or "00:00:00",
                        end=current_end or current_start or "00:00:00",
                        lines=current_lines,
                        speaker_map=speaker_map,
                    )
                )

                current_lines = []
                current_chars = 0
                current_start = start

            current_lines.append(line)
            current_chars += rendered_line_length
            current_end = end

        if current_lines:
            chunks.append(
                ChunkPromptInput(
                    chunk_index=len(chunks),
                    start=current_start or "00:00:00",
                    end=current_end or current_start or "00:00:00",
                    lines=current_lines,
                    speaker_map=speaker_map,
                )
            )

        return chunks

    def _format_time(self, seconds: float | None) -> str:
        if seconds is None:
            return "00:00:00"

        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"