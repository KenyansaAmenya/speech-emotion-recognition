import asyncio
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

from ..predict import load_pipeline, predict_file
from ..supabase_store import ensure_model, log_prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model()
    load_pipeline()
    print("SER pipeline loaded.")
    yield


app = FastAPI(
    title="Speech Emotion Recognition API",
    description=("Classifies acted speech into 6 emotion labels. "
                 "Predictions are data-driven labels, not measurements of true emotion."),
    version="1.0.0",
    lifespan=lifespan,
)

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGINS if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_AUDIO_SEC = 0.5


class PredictionResponse(BaseModel):
    emotion: str
    top_k: list[dict]
    duration_sec: float
    disclaimer: str


def _validate_and_predict(tmp_path: str) -> tuple[dict, float]:

    # 1. Header check (fast)
    try:
        info = sf.info(tmp_path)
    except Exception:
        raise HTTPException(422, "File could not be decoded as audio.")

    if info.duration < MIN_AUDIO_SEC:
        raise HTTPException(422, f"Audio too short ({info.duration:.2f}s).")

    # 2. Decode + silence check
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


@app.get("/")
def root():
    return {
        "service": "Speech Emotion Recognition API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart file upload)",
    }


@app.get("/health")
def health():
    try:
        load_pipeline()                # cached — ~1 µs
        return {"status": "ok", "classes": 6}
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            422,
            f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    chunks, total = [], 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 20 MB).")
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result, duration = await run_in_threadpool(_validate_and_predict, tmp_path)

        asyncio.create_task(
            asyncio.to_thread(
                log_prediction,
                emotion=result["emotion"],
                top_k=result["top_k"],
                duration_sec=duration,
            )
        )

        return PredictionResponse(
            emotion=result["emotion"],
            top_k=result["top_k"],
            duration_sec=round(duration, 2),
            disclaimer=result["disclaimer"],
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)