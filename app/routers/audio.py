from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from app.config import Settings, get_settings
from models.schemas import AudioProcessResponse
from services.nlu_pipeline import get_nlu_pipeline
from services.gemini_stt_client import get_gemini_stt_client
from services.sarvam_stt_client import get_sarvam_client


router = APIRouter()


@router.post("/process", response_model=AudioProcessResponse)
async def process_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    source: Optional[str] = None,
    settings: Settings = Depends(get_settings),
):
    # Choose STT backend
    provider = settings.stt_provider.lower()
    if provider == "sarvam":
        stt_client = get_sarvam_client(settings)
        use_language = language or settings.sarvam_language
    else:  # default to Gemini
        stt_client = get_gemini_stt_client(settings)
        use_language = language

    nlu = get_nlu_pipeline(settings)

    try:
        audio_bytes = await file.read()
        transcript_result = await stt_client.transcribe_bytes(
            audio_bytes,
            filename=file.filename or "audio",
            language=use_language,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Sarvam STT failed: {exc}") from exc

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
    provider = settings.stt_provider.lower()
    if provider == "sarvam":
        stt_client = get_sarvam_client(settings)
        use_language = language or settings.sarvam_language
    else:
        stt_client = get_gemini_stt_client(settings)
        use_language = language

    try:
        audio_bytes = await file.read()
        transcript_result = await stt_client.transcribe_bytes(
            audio_bytes,
            filename=file.filename or "audio",
            language=use_language,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Sarvam STT failed: {exc}") from exc

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

