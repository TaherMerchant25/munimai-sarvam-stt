import argparse
from pathlib import Path

import soundfile as sf  # type: ignore[import]

from app.config import get_settings
from services.nlu_pipeline import get_nlu_pipeline
from services.sarvam_stt_client import get_sarvam_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Sarvam STT + Intent classifier pipeline.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to audio file")
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Only run STT, skip intent classification",
    )
    args = parser.parse_args()

    settings = get_settings()
    sarvam = get_sarvam_client(settings)

    data, samplerate = sf.read(args.audio)
    # Re-encode to bytes via soundfile
    import io

    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV")
    audio_bytes = buf.getvalue()

    import asyncio

    async def run() -> None:
        result = await sarvam.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=args.audio.name,
            language=settings.sarvam_language,
        )
        print(f"Transcript: {result.text}")
        print(f"Language:   {result.language}")

        if args.transcript_only:
            return

        nlu = get_nlu_pipeline(settings)
        nlu_result = nlu.process_transcript(result.text)
        print(f"Intent:     {nlu_result.intent}")
        print(f"Confidence: {nlu_result.confidence:.3f}")
        print(f"Normalized: {nlu_result.normalized_text}")

    asyncio.run(run())


if __name__ == "__main__":
    main()

