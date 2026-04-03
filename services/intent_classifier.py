from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import Settings
from models.schemas import NLUResult


@dataclass
class IntentClassifierError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class IntentClassifier:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_dir = Path(settings.intent_model_path)
        if not self._model_dir.exists():
            raise IntentClassifierError(f"Intent model path not found: {self._model_dir}")

        # Load label config
        config_path = self._model_dir / "label_config.json"
        if not config_path.exists():
            raise IntentClassifierError(f"Missing label_config.json in {self._model_dir}")

        import json

        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.label2id: Dict[str, int] = {k: int(v) for k, v in cfg["label2id"].items()}
        self.id2label: Dict[int, str] = {int(k): v for k, v in cfg["id2label"].items()}
        self.max_length: int = int(cfg.get("max_length", 64))
        self.model_name: str = cfg.get("model_name", "ai4bharat/IndicBERTv2-MLM-only")

        if settings.use_onnx:
            self._load_onnx()
        else:
            self._load_hf()

    def _load_onnx(self) -> None:
        onnx_path = self._model_dir / "intent_classifier.onnx"
        if not onnx_path.exists():
            raise IntentClassifierError(f"ONNX model not found at {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_dir / "hf_model")
        self.session = ort.InferenceSession(str(onnx_path))

        self.backend = "onnx"

    def _load_hf(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_dir / "hf_model")
        self.model = AutoModelForSequenceClassification.from_pretrained(self._model_dir / "hf_model")
        self.model.eval()
        self.backend = "hf"

    def predict(self, text: str) -> NLUResult:
        encoded = self.tokenizer(
            text,
            return_tensors="np" if self.backend == "onnx" else "pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        if self.backend == "onnx":
            outputs = self.session.run(
                None,
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                },
            )
            logits = outputs[0][0]
        else:
            with np.errstate(all="ignore"):
                import torch

                with torch.no_grad():
                    out = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )
                logits = out.logits[0].detach().cpu().numpy()

        # Softmax
        logits_arr = np.asarray(logits, dtype=np.float32)
        exp_logits = np.exp(logits_arr - np.max(logits_arr))
        probs = exp_logits / exp_logits.sum()
        pred_idx = int(np.argmax(probs))
        pred_label = self.id2label[pred_idx]
        confidence = float(probs[pred_idx])

        return NLUResult(
            intent=pred_label,
            confidence=confidence,
            normalized_text=text,
        )


_intent_classifier: IntentClassifier | None = None


def reset_intent_classifier() -> None:
    global _intent_classifier
    _intent_classifier = None


def get_intent_classifier(settings: Settings) -> IntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier(settings)
    return _intent_classifier

