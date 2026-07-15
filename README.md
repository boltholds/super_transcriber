# super_transcriber

Super Transcriber is a locally hosted audio transcription, diarization and
conversation analysis service.

## CLI

```bash
poetry run conv transcribe data/meeting.flac --language ru
poetry run conv pipeline data/meeting.flac --language ru
```

## HTTP API

Run the asynchronous transcription API:

```bash
poetry run uvicorn super_transcriber.api.app:app --host 0.0.0.0 --port 8000
```

Submit audio and poll the returned job:

```bash
curl -X POST http://localhost:8000/v1/transcriptions \
  -H "X-API-Key: $API_KEY" \
  -F "audio=@meeting.flac" \
  -F "language=ru" \
  -F "min_speakers=2" \
  -F "max_speakers=6" \
  -F "analyze=true"
```

`GET /v1/transcriptions/{job_id}` returns the state and download URLs for
generated JSON and Markdown artifacts. The initial implementation keeps job
state in memory and serializes GPU work; a durable queue can replace the store
without changing the public contract.

