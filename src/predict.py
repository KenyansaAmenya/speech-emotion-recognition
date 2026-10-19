# Inference helpers
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

from . import config
from .features import extract_mfcc
from .preprocessing import preprocess

_PIPELINE = None
_PIPELINE_PATH: Path | None = None


# Load the serialized pipeline and cache it by path.
def load_pipeline(path: str | Path = config.MODELS_DIR / "ser_pipeline.joblib") -> dict:

    global _PIPELINE, _PIPELINE_PATH
    path = Path(path)
    if _PIPELINE is None or _PIPELINE_PATH != path:
        _PIPELINE = joblib.load(path)
        _PIPELINE_PATH = path
        _verify_feature_settings(_PIPELINE)
    return _PIPELINE


def _verify_feature_settings(pipe: dict) -> None:
    saved = pipe.get("feature_settings", {})
    current = {
        "sample_rate": config.SAMPLE_RATE,
        "duration_sec": config.DURATION_SEC,
        "top_db": config.TOP_DB_TRIM,
        "n_mfcc": config.N_MFCC,
    }
    mismatches = {k: (saved.get(k), current[k])
                  for k in current if saved.get(k) != current[k]}
    if mismatches:
        raise ValueError(
            f"Feature settings mismatch between saved model and config.py: {mismatches}. "
            "The model was trained with different preprocessing — retrain or fix config."
        )


def _extract_features_from_payload(pipe: dict, file_path: str) -> np.ndarray:
    fs = pipe["feature_settings"]
    y = preprocess(file_path,
                   sr=fs["sample_rate"],
                   duration=fs["duration_sec"],
                   top_db=fs["top_db"])
    feats = extract_mfcc(y, sr=fs["sample_rate"], n_mfcc=fs["n_mfcc"])
    return feats.reshape(1, -1)

# Predict emotion for a single audio file. Returns label + probability ranking.
def predict_file(file_path: str, top_k: int = 3) -> dict:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    pipe = load_pipeline()
    feats = _extract_features_from_payload(pipe, file_path)
    label_idx = int(pipe["model"].predict(feats)[0])

    if hasattr(pipe["model"], "predict_proba"):
        proba = pipe["model"].predict_proba(feats)[0]
        ranking = [
            {"emotion": pipe["idx2label"][i], "probability": float(p)}
            for i, p in sorted(enumerate(proba), key=lambda t: t[1], reverse=True)[:top_k]
        ]
    else:
        ranking = [{"emotion": pipe["idx2label"][label_idx], "probability": None}]

    return {
        "emotion": pipe["idx2label"][label_idx],
        "label_idx": label_idx,
        "top_k": ranking,
        "disclaimer": (
            "Predicted label reflects patterns in training data "
            "(RAVDESS acted speech), not the speaker's true internal state."
        ),
    }

# Predict from raw bytes.
def predict_bytes(audio_bytes: bytes, suffix: str = ".wav") -> dict:

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return predict_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass