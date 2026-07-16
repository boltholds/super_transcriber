from pydantic import BaseModel, Field


class TranscriptLine(BaseModel):
    speaker: str
    start: str
    end: str
    text: str


class ChunkPromptInput(BaseModel):
    chunk_index: int
    start: str
    end: str
    lines: list[TranscriptLine]
    speaker_map: dict[str, str] = Field(default_factory=dict)


class MergePromptInput(BaseModel):
    chunk_analyses_json: str
    
    
class ParticipantSchema(BaseModel):
    speaker: str = Field(description="Speaker label, for example SPEAKER_00")
    assumed_name: str | None = None
    role: str | None = None
    notes: str | None = None


class ActionItemSchema(BaseModel):
    owner: str | None = None
    task: str
    deadline: str | None = None
    evidence: str | None = None


class TimelineItemSchema(BaseModel):
    start: str | None = None
    end: str | None = None
    title: str
    description: str


class ChunkAnalysisSchema(BaseModel):
    chunk_index: int
    start: str
    end: str
    summary: str
    participants: list[ParticipantSchema] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItemSchema] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    important_facts: list[str] = Field(default_factory=list)
    timeline: list[TimelineItemSchema] = Field(default_factory=list)


class FinalAnalysisSchema(BaseModel):
    summary: str
    participants: list[ParticipantSchema] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItemSchema] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    timeline: list[TimelineItemSchema] = Field(default_factory=list)