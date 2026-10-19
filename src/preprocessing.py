import numpy as np
import librosa

from . import config


# Load audio, mono, standardized sample rate.
def load_audio(path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def trim_silence(y: np.ndarray, sr: int, top_db: int = config.TOP_DB_TRIM) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > int(sr * 0.25) else y  # guard: if trimming removes too much, return original

def fix_length(y: np.ndarray, n_samples: int = config.N_SAMPLES) -> np.ndarray:
    if len(y) >= n_samples:
        start = (len(y) - n_samples) // 2   # center-crop keeps expressive content
        return y[start:start + n_samples]
    return np.pad(y, (0, n_samples - len(y)), mode="edge")


def preprocess(path: str,
               sr: int = config.SAMPLE_RATE,
               duration: float = config.DURATION_SEC,
               top_db: int = config.TOP_DB_TRIM) -> np.ndarray:
    n_samples = int(sr * duration)
    y = load_audio(path, sr=sr)
    y = trim_silence(y, sr=sr, top_db=top_db)
    y = fix_length(y, n_samples)
    peak = np.max(np.abs(y))
    if peak > 1e-6:                      # mild peak normalization
        y = y / peak
    return y.astype(np.float32)