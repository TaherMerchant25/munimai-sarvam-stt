from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

from app.config import Settings
from models.schemas import TranscriptResult


@dataclass
class GeminiSttError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class GeminiSttClient:
    def __init__(self, settings: Settings) -> None:
        # Requires GEMINI_API_KEY in environment (or via genai.configure)
        try:
            genai.configure()  # uses GEMINI_API_KEY env var
        except Exception as exc:  # noqa: BLE001
            raise GeminiSttError(f"Failed to configure Gemini client: {exc}") from exc

        # Use a multimodal model that supports audio → text
        self._model = genai.GenerativeModel("models/gemini-1.5-flash")

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        """
        Use Gemini multimodal model to get a transcript for the given audio bytes.
        """
        import io

        # Wrap bytes as an in-memory file for the SDK
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        try:
            # Prompt Gemini to act as a pure transcriber
            response = self._model.generate_content(
                [
                    "Transcribe this audio. Respond with only the raw transcript text, no explanations.",
                    {"mime_type": "audio/wav", "data": audio_file.read()},
                ]
            )
        except Exception as exc:  # noqa: BLE001
            raise GeminiSttError(f"Gemini STT failed: {exc}") from exc

        text = (response.text or "").strip()
        return TranscriptResult(text=text, language=language or "unknown", segments=None)


_gemini_stt_client: GeminiSttClient | None = None


def get_gemini_stt_client(settings: Settings) -> GeminiSttClient:
    global _gemini_stt_client
    if _gemini_stt_client is None:
        _gemini_stt_client = GeminiSttClient(settings)
    return _gemini_stt_client

