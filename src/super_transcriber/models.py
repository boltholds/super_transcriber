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


class Participant(BaseModel):
    speaker: str
    assumed_name: str | None = None
    role: str | None = None
    notes: str | None = None


class ActionItem(BaseModel):
    owner: str | None = None
    task: str
    deadline: str | None = None
    evidence: str | None = None


class TimelineItem(BaseModel):
    start: str | None = None
    end: str | None = None
    title: str
    description: str


class ConversationAnalysis(BaseModel):
    transcript_path: str
    summary: str
    participants: list[Participant] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)