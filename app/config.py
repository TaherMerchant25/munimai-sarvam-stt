from functools import lru_cache
from typing import List

from pydantic import BaseSettings, AnyHttpUrl


class Settings(BaseSettings):
    # Sarvam STT
    sarvam_api_key: str | None = None
    sarvam_base_url: AnyHttpUrl = "https://api.sarvam.ai"  # type: ignore[assignment]
    sarvam_stt_model: str = "whisper-large-v3"  # placeholder, align with actual Sarvam model id
    sarvam_language: str = "hi"

    # STT provider: "sarvam", "gemini", or "elevenlabs"
    stt_provider: str = "gemini"

    # ElevenLabs Speech-to-Text (https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
    elevenlabs_api_key: str | None = None
    elevenlabs_stt_model: str = "scribe_v2"
    elevenlabs_base_url: AnyHttpUrl = "https://api.elevenlabs.io"  # type: ignore[assignment]

    # Intent model
    intent_model_path: str = "ml_models/intent_classifier"
    use_onnx: bool = True

    # Server / CORS
    cors_allow_origins: List[str] = ["*"]

    class Config:
        env_prefix = ""
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()

