from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from super_transcriber.services.analysis_service import TranscriptAnalysisService
from super_transcriber.services.export_service import ExportService, format_time
from super_transcriber.services.whisperx_service import WhisperXService
from super_transcriber.settings import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Super Transcriber: audio transcription, diarization and conversation analysis.",
)

console = Console()


@app.command("transcribe")
def transcribe(
    audio_path: Path = typer.Argument(
        ...,
        help="Path to audio file.",
    ),
    language: str = typer.Option(
        "ru",
        "--language",
        "-l",
        help="Audio language code.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="WhisperX model name. If not provided, uses WHISPERX_MODEL.",
    ),
    speakers: int | None = typer.Option(
        None,
        "--speakers",
        help="Exact number of speakers.",
    ),
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers.",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        help="Maximum number of speakers.",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
        help="Directory for generated transcript files.",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="WhisperX batch size.",
    ),
) -> None:
    if not audio_path.exists():
        raise typer.BadParameter(f"Audio file not found: {audio_path}")

    if speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise typer.BadParameter(
            "Use either --speakers or --min-speakers/--max-speakers, not both."
        )

    settings = get_settings()
    selected_model = model or settings.whisperx_model

    console.print(
        Panel.fit(
            f"[bold]Audio:[/bold] {audio_path}\n"
            f"[bold]Language:[/bold] {language}\n"
            f"[bold]Model:[/bold] {selected_model}",
            title="Transcription",
        )
    )

    service = WhisperXService(
        hf_token=settings.hf_token,
        model_name=selected_model,
        device=settings.whisperx_device,
        compute_type=settings.whisperx_compute_type,
    )

    transcript = service.transcribe(
        audio_path=audio_path,
        language=language,
        num_speakers=speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        batch_size=batch_size,
    )

    exporter = ExportService()

    stem = audio_path.stem
    json_path = output_dir / f"{stem}.transcript.json"
    md_path = output_dir / f"{stem}.transcript.md"

    exporter.export_json(transcript, json_path)
    exporter.export_markdown(transcript, md_path)

    console.print()
    console.print("[bold green]Transcript preview[/bold green]")
    console.print("-" * 80)

    for segment in transcript.segments[:20]:
        start = format_time(segment.start)
        end = format_time(segment.end)

        console.print(
            f"[bold]{segment.speaker}[/bold] "
            f"[dim][{start} - {end}][/dim]: "
            f"{segment.text}"
        )

    if len(transcript.segments) > 20:
        console.print(
            f"[dim]... {len(transcript.segments) - 20} more segments omitted from preview[/dim]"
        )

    console.print()
    console.print(f"[green]Saved JSON:[/green] {json_path}")
    console.print(f"[green]Saved Markdown:[/green] {md_path}")


@app.command("analyze")
def analyze(
    transcript_path: Path = typer.Argument(
        ...,
        help="Path to .transcript.json file.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory for generated analysis files. Defaults to transcript directory.",
    ),
    max_chunk_chars: int = typer.Option(
        8_000,
        "--max-chunk-chars",
        help="Maximum rendered characters per analysis chunk.",
    ),
    speaker_map: Path | None = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker map JSON file.",
    ),
) -> None:
    if not transcript_path.exists():
        raise typer.BadParameter(f"Transcript file not found: {transcript_path}")

    if speaker_map is not None and not speaker_map.exists():
        raise typer.BadParameter(f"Speaker map file not found: {speaker_map}")

    if not transcript_path.name.endswith(".transcript.json"):
        console.print(
            "[yellow]Warning:[/yellow] input file does not end with .transcript.json"
        )

    settings = get_settings()

    speaker_map_line = str(speaker_map) if speaker_map else "not provided"

    console.print(
        Panel.fit(
            f"[bold]Transcript:[/bold] {transcript_path}\n"
            f"[bold]Speaker map:[/bold] {speaker_map_line}\n"
            f"[bold]LLM model:[/bold] {settings.llm_model}\n"
            f"[bold]LLM base URL:[/bold] {settings.llm_base_url}\n"
            f"[bold]Max chunk chars:[/bold] {max_chunk_chars}",
            title="Analysis",
        )
    )

    service = TranscriptAnalysisService(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_chunk_chars=max_chunk_chars,
    )

    analysis = service.analyze_file(
        transcript_path=transcript_path,
        speaker_map_path=speaker_map,
    )

    exporter = ExportService()

    selected_output_dir = output_dir or transcript_path.parent
    stem = transcript_path.name.replace(".transcript.json", "")

    json_path = selected_output_dir / f"{stem}.analysis.json"
    md_path = selected_output_dir / f"{stem}.analysis.md"

    exporter.export_analysis_json(analysis, json_path)
    exporter.export_analysis_markdown(analysis, md_path)

    console.print()
    console.print("[bold green]Analysis summary[/bold green]")
    console.print("-" * 80)
    console.print(analysis.summary)
    console.print()
    console.print(f"[green]Saved JSON:[/green] {json_path}")
    console.print(f"[green]Saved Markdown:[/green] {md_path}")


@app.command("pipeline")
def pipeline(
    audio_path: Path = typer.Argument(
        ...,
        help="Path to audio file.",
    ),
    language: str = typer.Option(
        "ru",
        "--language",
        "-l",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
    ),
    speakers: int | None = typer.Option(
        None,
        "--speakers",
    ),
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
    ),
    max_chunk_chars: int = typer.Option(
        8_000,
        "--max-chunk-chars",
    ),
) -> None:
    """
    Run full pipeline: transcribe audio, then analyze generated transcript.
    """

    if not audio_path.exists():
        raise typer.BadParameter(f"Audio file not found: {audio_path}")

    if speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise typer.BadParameter(
            "Use either --speakers or --min-speakers/--max-speakers, not both."
        )

    settings = get_settings()
    selected_model = model or settings.whisperx_model

    console.print(
        Panel.fit(
            f"[bold]Audio:[/bold] {audio_path}\n"
            f"[bold]Language:[/bold] {language}\n"
            f"[bold]WhisperX model:[/bold] {selected_model}\n"
            f"[bold]LLM model:[/bold] {settings.llm_model}",
            title="Full Pipeline",
        )
    )

    whisperx_service = WhisperXService(
        hf_token=settings.hf_token,
        model_name=selected_model,
        device=settings.whisperx_device,
        compute_type=settings.whisperx_compute_type,
    )

    transcript = whisperx_service.transcribe(
        audio_path=audio_path,
        language=language,
        num_speakers=speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        batch_size=batch_size,
    )

    exporter = ExportService()

    stem = audio_path.stem
    transcript_json_path = output_dir / f"{stem}.transcript.json"
    transcript_md_path = output_dir / f"{stem}.transcript.md"

    exporter.export_json(transcript, transcript_json_path)
    exporter.export_markdown(transcript, transcript_md_path)

    console.print(f"[green]Saved transcript JSON:[/green] {transcript_json_path}")
    console.print(f"[green]Saved transcript Markdown:[/green] {transcript_md_path}")

    analysis_service = TranscriptAnalysisService(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_chunk_chars=max_chunk_chars,
    )

    analysis = analysis_service.analyze_file(transcript_json_path)

    analysis_json_path = output_dir / f"{stem}.analysis.json"
    analysis_md_path = output_dir / f"{stem}.analysis.md"

    exporter.export_analysis_json(analysis, analysis_json_path)
    exporter.export_analysis_markdown(analysis, analysis_md_path)

    console.print()
    console.print("[bold green]Analysis summary[/bold green]")
    console.print("-" * 80)
    console.print(analysis.summary)
    console.print()
    console.print(f"[green]Saved analysis JSON:[/green] {analysis_json_path}")
    console.print(f"[green]Saved analysis Markdown:[/green] {analysis_md_path}")


if __name__ == "__main__":
    app()