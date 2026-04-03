## MunimAI Sarvam STT Service

Backend service that takes raw audio, calls Sarvam AI STT for Hindi/Hinglish transcription, and runs an IndicBERTv2 intent classifier to detect financial voice intents.

### Setup

1. Create a virtualenv and install dependencies:

```bash
cd munimai_sarvam_stt
pip install -r requirements.txt
```

2. Create a `.env` file:

```bash
SARVAM_API_KEY=your_sarvam_key_here
SARVAM_STT_MODEL=whisper-large-v3   # or the Sarvam STT model id you use
SARVAM_LANGUAGE=hi
INTENT_MODEL_PATH=ml_models/intent_classifier
USE_ONNX=true
STT_PROVIDER=gemini        # or "sarvam"
GEMINI_API_KEY=your_gemini_key_here
```

3. Export the intent classifier from `01_intent_classifier_training.ipynb` and copy the exported folder (containing `hf_model/`, `intent_classifier.onnx`, `label_config.json`) into:

```text
munimai_sarvam_stt/ml_models/intent_classifier/
```

### Run the API server

```bash
cd munimai_sarvam_stt
./scripts/run_local_server.sh
```

By default the service runs on `http://localhost:8001`.

### API Endpoints

- **POST** `/api/audio/process`

  - Request (multipart form-data):
    - `file`: audio file (`.wav`, `.mp3`, etc.)
    - Optional query/body: `language`, `source`
  - Response:
    - `transcript`: STT text
    - `language`: detected/assumed language
    - `intent`: predicted intent label
    - `intent_confidence`: confidence score
    - `meta`: extra metadata (source, segments, normalized text)

- **POST** `/api/audio/transcript`

  Same as above but returns only transcript + language (no intent).

Example `curl`:

```bash
curl -X POST "http://localhost:8001/api/audio/process" \
  -F "file=@sample.wav"
```

### CLI Testing

You can also test the full pipeline from the command line:

```bash
cd munimai_sarvam_stt
python scripts/test_intent_cli.py --audio sample.wav
```

To only run STT:

```bash
python scripts/test_intent_cli.py --audio sample.wav --transcript-only
```

