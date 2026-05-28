import json
from pathlib import Path
from typing import Any

import httpx

from super_transcriber.models import (
    ActionItem,
    ConversationAnalysis,
    Participant,
    TimelineItem,
    Transcript,
)


class TranscriptAnalysisService:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def analyze_file(self, transcript_path: Path) -> ConversationAnalysis:
        transcript = self._load_transcript(transcript_path)
        prompt = self._build_prompt(transcript)

        raw = self._call_llm(prompt)
        data = self._parse_json(raw)

        return self._to_analysis(
            transcript_path=transcript_path,
            data=data,
        )

    def _load_transcript(self, transcript_path: Path) -> Transcript:
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        raw = transcript_path.read_text(encoding="utf-8")
        return Transcript.model_validate_json(raw)

    def _build_prompt(self, transcript: Transcript) -> str:
        dialogue_lines = []

        for segment in transcript.segments:
            start = self._format_time(segment.start)
            end = self._format_time(segment.end)
            text = segment.text.strip()

            if not text:
                continue

            dialogue_lines.append(
                f"{segment.speaker} [{start} - {end}]: {text}"
            )

        dialogue = "\n".join(dialogue_lines)

        return f"""
Ты анализируешь расшифровку разговора.

Нужно извлечь структурированный анализ:
- краткое резюме
- участников
- основные темы
- решения
- задачи
- открытые вопросы
- риски
- таймлайн разговора

Важно:
- Не выдумывай факты.
- Если имя участника неясно, оставляй assumed_name = null.
- Для спорных моментов формулируй нейтрально.
- Не исправляй юридически значимые смыслы.
- Верни только валидный JSON без markdown-блока.

JSON schema:

{{
  "summary": "string",
  "participants": [
    {{
      "speaker": "SPEAKER_00",
      "assumed_name": null,
      "role": "string or null",
      "notes": "string or null"
    }}
  ],
  "topics": ["string"],
  "decisions": ["string"],
  "action_items": [
    {{
      "owner": "SPEAKER_00 or name or null",
      "task": "string",
      "deadline": "string or null",
      "evidence": "short quote or timestamp or null"
    }}
  ],
  "open_questions": ["string"],
  "risks": ["string"],
  "timeline": [
    {{
      "start": "00:00:00",
      "end": "00:01:00",
      "title": "string",
      "description": "string"
    }}
  ]
}}

Расшифровка:

{dialogue}
""".strip()

    def _call_llm(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты аккуратный аналитик переговоров. "
                        "Ты возвращаешь только валидный JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()

        if text.startswith("```"):
            text = text.removeprefix("```").strip()

        if text.endswith("```"):
            text = text.removesuffix("```").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON:\n{text}"
            ) from exc

    def _to_analysis(
        self,
        transcript_path: Path,
        data: dict[str, Any],
    ) -> ConversationAnalysis:
        return ConversationAnalysis(
            transcript_path=str(transcript_path),
            summary=data.get("summary", ""),
            participants=[
                Participant.model_validate(item)
                for item in data.get("participants", [])
            ],
            topics=list(data.get("topics", [])),
            decisions=list(data.get("decisions", [])),
            action_items=[
                ActionItem.model_validate(item)
                for item in data.get("action_items", [])
            ],
            open_questions=list(data.get("open_questions", [])),
            risks=list(data.get("risks", [])),
            timeline=[
                TimelineItem.model_validate(item)
                for item in data.get("timeline", [])
            ],
        )

    def _format_time(self, seconds: float | None) -> str:
        if seconds is None:
            return "00:00:00"

        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"