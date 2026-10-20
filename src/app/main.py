import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .. import config
from ..predict import load_pipeline, predict_file

logger = logging.getLogger("ser")
logging.basicConfig(level=logging.INFO)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_AUDIO_SEC = 0.5


class PredictionResponse(BaseModel):
    emotion: str
    top_k: list[dict]
    duration_sec: float
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    classes: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = load_pipeline()
    saved = pipe.get("feature_settings", {})
    current = {
        "sample_rate": config.SAMPLE_RATE,
        "duration_sec": config.DURATION_SEC,
        "top_db": config.TOP_DB_TRIM,
        "n_mfcc": config.N_MFCC,
    }
    mismatches = {k: (saved.get(k), current[k]) for k in current if saved.get(k) != current[k]}
    if mismatches:
        raise RuntimeError(f"Feature settings mismatch — model/config drift: {mismatches}")
    logger.info("SER pipeline loaded; config verified.")
    yield


app = FastAPI(
    title="Speech Emotion Recognition API",
    description=("Classifies acted speech into 6 emotion labels. "
                 "Predictions are data-driven labels, not measurements of true emotion."),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_and_predict(tmp_path: str) -> tuple[dict, float]:
    # 1. Fast header-only checks
    try:
        info = sf.info(tmp_path)
    except Exception:
        raise HTTPException(422, "File could not be decoded as audio.")
    if info.duration < MIN_AUDIO_SEC:
        raise HTTPException(422, f"Audio too short ({info.duration:.2f}s < {MIN_AUDIO_SEC}s).")

    # 2. Load for silence check
    try:
        y, _ = librosa.load(tmp_path, sr=None, mono=True)
    except Exception:
        raise HTTPException(422, "File could not be decoded as audio.")
    if np.max(np.abs(y)) < 1e-4:
        raise HTTPException(422, "Audio is silent.")

    # 3. Predict
    try:
        result = predict_file(tmp_path)
    except FileNotFoundError:
        raise HTTPException(404, "Audio file was deleted before prediction.")
    except ValueError as e:
        raise HTTPException(422, f"Prediction failed: {e}")

    return result, float(info.duration)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        load_pipeline()
        return HealthResponse(status="ok", classes=config.NUM_CLASSES)
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            422,
            f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # 2. Stream-read with size cap
    chunks, total = [], 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 20 MB).")
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)

    # 3. Write temp file, predict, cleanup
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Offload blocking work — keeps event loop responsive
        result, duration = await run_in_threadpool(_validate_and_predict, tmp_path)

        logger.info(f"predict ok: {file.filename} -> {result['emotion']} ({duration:.2f}s)")
        return PredictionResponse(
            emotion=result["emotion"],
            top_k=result["top_k"],
            duration_sec=round(duration, 2),
            disclaimer=result["disclaimer"],
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)