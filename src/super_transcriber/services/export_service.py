from pathlib import Path

from super_transcriber.models import ConversationAnalysis, Transcript


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ExportService:
    def export_json(self, transcript: Transcript, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            transcript.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return output_path

    def export_markdown(self, transcript: Transcript, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []

        lines.append("# Transcript")
        lines.append("")
        lines.append(f"Audio: `{transcript.audio_path}`")
        lines.append(f"Language: `{transcript.language}`")
        lines.append(f"Model: `{transcript.model}`")
        lines.append(f"Device: `{transcript.device}`")
        lines.append("")
        lines.append("## Dialogue")
        lines.append("")

        for segment in transcript.segments:
            start = format_time(segment.start)
            end = format_time(segment.end)

            lines.append(f"**{segment.speaker} [{start} - {end}]**")
            lines.append("")
            lines.append(segment.text.strip())
            lines.append("")

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return output_path

    def export_analysis_json(
        self,
        analysis: ConversationAnalysis,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            analysis.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return output_path

    def export_analysis_markdown(
        self,
        analysis: ConversationAnalysis,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []

        lines.append("# Conversation Analysis")
        lines.append("")
        lines.append(f"Transcript: `{analysis.transcript_path}`")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(analysis.summary.strip())
        lines.append("")

        lines.append("## Participants")
        lines.append("")

        if analysis.participants:
            for participant in analysis.participants:
                name = participant.assumed_name or "unknown"
                role = participant.role or "unknown role"
                notes = participant.notes or ""

                lines.append(
                    f"- **{participant.speaker}** — {name}; {role}. {notes}".strip()
                )
        else:
            lines.append("- No participants extracted.")

        lines.append("")

        lines.append("## Topics")
        lines.append("")
        self._append_list(lines, analysis.topics)

        lines.append("## Decisions")
        lines.append("")
        self._append_list(lines, analysis.decisions)

        lines.append("## Action Items")
        lines.append("")

        if analysis.action_items:
            for item in analysis.action_items:
                owner = item.owner or "unknown"
                deadline = item.deadline or "no deadline"
                evidence = f" Evidence: {item.evidence}" if item.evidence else ""

                lines.append(
                    f"- **{owner}**: {item.task} "
                    f"Deadline: {deadline}.{evidence}"
                )
        else:
            lines.append("- No action items extracted.")

        lines.append("")

        lines.append("## Open Questions")
        lines.append("")
        self._append_list(lines, analysis.open_questions)

        lines.append("## Risks")
        lines.append("")
        self._append_list(lines, analysis.risks)

        lines.append("## Timeline")
        lines.append("")

        if analysis.timeline:
            for item in analysis.timeline:
                start = item.start or "?"
                end = item.end or "?"

                lines.append(f"- **[{start} - {end}] {item.title}**")
                lines.append(f"  {item.description}")
        else:
            lines.append("- No timeline extracted.")

        lines.append("")

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return output_path

    def _append_list(self, lines: list[str], items: list[str]) -> None:
        if not items:
            lines.append("- Nothing extracted.")
            lines.append("")
            return

        for item in items:
            lines.append(f"- {item}")

        lines.append("")