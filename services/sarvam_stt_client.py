from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from app.config import Settings
from models.schemas import TranscriptResult, TranscriptSegment


@dataclass
class SarvamError(Exception):
    message: str
    status_code: Optional[int] = None
    payload: Optional[dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.status_code}: {self.message}" if self.status_code else self.message


class SarvamSTTClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise SarvamError("SARVAM_API_KEY is not configured")

        self._api_key = settings.sarvam_api_key
        self._base_url = str(settings.sarvam_base_url).rstrip("/")
        self._model = settings.sarvam_stt_model
        self._timeout = 60.0

        # Single shared async client
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str,
        diarize: bool = False,
        punctuate: bool = True,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> TranscriptResult:
        """
        Call Sarvam STT on in-memory audio bytes.

        NOTE: Payload structure may need to be adjusted to match the latest
        Sarvam STT API; this implementation assumes a typical multipart + JSON pattern.
        """
        files = {"file": (filename, audio_bytes)}
        data: dict[str, Any] = {
            "model": self._model,
            "language": language,
            "diarize": diarize,
            "punctuate": punctuate,
        }
        if extra_payload:
            data.update(extra_payload)

        try:
            resp = await self._client.post("/stt", files=files, data=data)
        except httpx.RequestError as exc:  # noqa: PERF203
            raise SarvamError(f"Network error calling Sarvam STT: {exc}") from exc

        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001
                payload = None
            raise SarvamError(
                message=f"Sarvam STT HTTP {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                payload=payload,
            )

        body = resp.json()
        text = body.get("text") or body.get("transcript") or ""
        language_out = body.get("language", language)
        segments_raw = body.get("segments") or []
        segments = [
            TranscriptSegment(
                start=seg.get("start"),
                end=seg.get("end"),
                text=seg.get("text", ""),
            )
            for seg in segments_raw
            if seg.get("text")
        ]

        return TranscriptResult(
            text=text,
            language=language_out,
            segments=segments or None,
        )


_sarvam_client: SarvamSTTClient | None = None
_sarvam_lock = asyncio.Lock()


def reset_sarvam_client() -> None:
    global _sarvam_client
    _sarvam_client = None


def get_sarvam_client(settings: Settings) -> SarvamSTTClient:
    """
    Lazy singleton for SarvamSTTClient.
    FastAPI dependencies can call this to reuse a single async client.
    """
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamSTTClient(settings)
    return _sarvam_client

