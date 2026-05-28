from pathlib import Path

import typer
from rich.console import Console

from super_transcriber.services.analysis_service import TranscriptAnalysisService
from super_transcriber.services.export_service import ExportService, format_time
from super_transcriber.services.whisperx_service import WhisperXService
from super_transcriber.settings import get_settings

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("transcribe")
def transcribe(
    audio_path: Path = typer.Argument(..., help="Path to audio file"),
    language: str = typer.Option("ru", "--language", "-l"),
    model: str | None = typer.Option(None, "--model", "-m"),
    speakers: int | None = typer.Option(None, "--speakers"),
    min_speakers: int | None = typer.Option(None, "--min-speakers"),
    max_speakers: int | None = typer.Option(None, "--max-speakers"),
    output_dir: Path = typer.Option(Path("output"), "--out", "-o"),
    batch_size: int = typer.Option(8, "--batch-size"),
) -> None:
    settings = get_settings()

    selected_model = model or settings.whisperx_model

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
    console.print("[bold green]Transcript preview:[/bold green]")
    console.print("-" * 80)

    for segment in transcript.segments:
        start = format_time(segment.start)
        end = format_time(segment.end)

        console.print(
            f"[bold]{segment.speaker}[/bold] "
            f"[dim][{start} - {end}][/dim]: "
            f"{segment.text}"
        )

    console.print()
    console.print(f"[green]Saved JSON:[/green] {json_path}")
    console.print(f"[green]Saved Markdown:[/green] {md_path}")


@app.command("analyze")
def analyze(
    transcript_path: Path = typer.Argument(..., help="Path to transcript JSON"),
    output_dir: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    settings = get_settings()

    service = TranscriptAnalysisService(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    analysis = service.analyze_file(transcript_path)

    exporter = ExportService()

    selected_output_dir = output_dir or transcript_path.parent

    stem = transcript_path.name.replace(".transcript.json", "")
    json_path = selected_output_dir / f"{stem}.analysis.json"
    md_path = selected_output_dir / f"{stem}.analysis.md"

    exporter.export_analysis_json(analysis, json_path)
    exporter.export_analysis_markdown(analysis, md_path)

    console.print()
    console.print("[bold green]Analysis:[/bold green]")
    console.print("-" * 80)
    console.print(analysis.summary)
    console.print()
    console.print(f"[green]Saved JSON:[/green] {json_path}")
    console.print(f"[green]Saved Markdown:[/green] {md_path}")


if __name__ == "__main__":
    app()