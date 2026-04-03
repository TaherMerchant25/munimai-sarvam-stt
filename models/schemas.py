from typing import Any, List, Optional

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    text: str


class AudioProcessResponse(BaseModel):
    transcript: str
    language: str
    intent: str
    intent_confidence: float
    meta: dict[str, Any] | None = None


class TranscriptResult(BaseModel):
    text: str
    language: str
    segments: Optional[List[TranscriptSegment]] = None


class NLUResult(BaseModel):
    intent: str
    confidence: float
    normalized_text: str

