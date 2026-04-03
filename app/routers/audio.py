from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from app.config import Settings, get_settings
from models.schemas import AudioProcessResponse
from services.elevenlabs_stt_client import get_elevenlabs_stt_client
from services.nlu_pipeline import get_nlu_pipeline
from services.gemini_stt_client import get_gemini_stt_client
from services.sarvam_stt_client import get_sarvam_client


router = APIRouter()


def _stt_client_and_language(
    settings: Settings,
    language: Optional[str],
):
    provider = settings.stt_provider.lower()
    if provider == "sarvam":
        return get_sarvam_client(settings), language or settings.sarvam_language
    if provider == "elevenlabs":
        return get_elevenlabs_stt_client(settings), language
    return get_gemini_stt_client(settings), language


@router.post("/process", response_model=AudioProcessResponse)
async def process_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    source: Optional[str] = None,
    settings: Settings = Depends(get_settings),
):
    stt_client, use_language = _stt_client_and_language(settings, language)

    nlu = get_nlu_pipeline(settings)

    try:
        audio_bytes = await file.read()
        transcript_result = await stt_client.transcribe_bytes(
            audio_bytes,
            filename=file.filename or "audio",
            language=use_language,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"STT failed: {exc}") from exc

    nlu_result = nlu.process_transcript(transcript_result.text)

    return AudioProcessResponse(
        transcript=transcript_result.text,
        language=transcript_result.language,
        intent=nlu_result.intent,
        intent_confidence=nlu_result.confidence,
        meta={
            "source": source or "unknown",
            "segments": [s.dict() for s in transcript_result.segments] if transcript_result.segments else None,
            "normalized_text": nlu_result.normalized_text,
        },
    )


@router.post("/transcript", response_model=AudioProcessResponse)
async def transcript_only(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    source: Optional[str] = None,
    settings: Settings = Depends(get_settings),
):
    stt_client, use_language = _stt_client_and_language(settings, language)

    try:
        audio_bytes = await file.read()
        transcript_result = await stt_client.transcribe_bytes(
            audio_bytes,
            filename=file.filename or "audio",
            language=use_language,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"STT failed: {exc}") from exc

    # No NLU here
    return AudioProcessResponse(
        transcript=transcript_result.text,
        language=transcript_result.language,
        intent="",
        intent_confidence=0.0,
        meta={
            "source": source or "unknown",
            "segments": [s.dict() for s in transcript_result.segments] if transcript_result.segments else None,
        },
    )

