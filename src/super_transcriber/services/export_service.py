import json
from pathlib import Path

from super_transcriber.models import Transcript


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