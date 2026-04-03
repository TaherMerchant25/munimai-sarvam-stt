from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.routers import audio


def create_app() -> FastAPI:
    settings: Settings = get_settings()

    app = FastAPI(
        title="MunimAI Sarvam STT Service",
        version="0.1.0",
        description="FastAPI backend for Sarvam STT + IndicBERTv2 intent classification.",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(audio.router, prefix="/api/audio", tags=["audio"])

    return app


app = create_app()

