from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings
from models.schemas import NLUResult
from services.intent_classifier import get_intent_classifier


_whitespace_re = re.compile(r"\s+")


@dataclass
class NLUPipeline:
    settings: Settings

    def _normalize(self, text: str) -> str:
        # Basic, safe normalization for Hindi/Hinglish
        text = text.strip()
        text = _whitespace_re.sub(" ", text)
        return text

    def process_transcript(self, text: str) -> NLUResult:
        normalized = self._normalize(text)
        classifier = get_intent_classifier(self.settings)
        result = classifier.predict(normalized)
        # Ensure normalized_text set to our normalized variant
        result.normalized_text = normalized
        return result


_nlu_pipeline: NLUPipeline | None = None


def get_nlu_pipeline(settings: Settings) -> NLUPipeline:
    global _nlu_pipeline
    if _nlu_pipeline is None:
        _nlu_pipeline = NLUPipeline(settings=settings)
    return _nlu_pipeline

