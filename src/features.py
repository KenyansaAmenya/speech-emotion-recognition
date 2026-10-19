# Feature extraction: MFCC (tabular baseline) and Mel spectrograms (CNN).
import numpy as np
import librosa

from . import config


# MFCC mean + std across time -> 2*n_mfcc dimensional vector.
def extract_mfcc(waveform: np.ndarray,
                 sr: int = config.SAMPLE_RATE,
                 n_mfcc: int = config.N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feats.astype(np.float32)


def mfcc_pipeline(path: str, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    from .preprocessing import preprocess
    y = preprocess(path, sr=sr)
    return extract_mfcc(y, sr=sr)


def extract_melspectrogram(waveform: np.ndarray,
                           sr: int = config.SAMPLE_RATE,
                           n_mels: int = config.N_MELS_CNN) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def spectrogram_to_image(log_mel: np.ndarray,
                         img_size: int = config.CNN_IMAGE_SIZE) -> np.ndarray:
    assert log_mel.ndim == 2, f"Expected 2-D log-mel, got shape {log_mel.shape}"
    h, w = log_mel.shape

    yi = np.linspace(0, h - 1, img_size)        
    xi = np.linspace(0, w - 1, img_size)          

    yi0 = np.floor(yi).astype(np.int64)
    xi0 = np.floor(xi).astype(np.int64)
    yi1 = np.minimum(yi0 + 1, h - 1)
    xi1 = np.minimum(xi0 + 1, w - 1)

    wy = (yi - yi0)[:, None]                      
    wx = (xi - xi0)[None, :]                      

    a = log_mel[np.ix_(yi0, xi0)]                
    b = log_mel[np.ix_(yi0, xi1)]                 
    c = log_mel[np.ix_(yi1, xi0)]                 
    d = log_mel[np.ix_(yi1, xi1)]                

    img = (a * (1 - wy) * (1 - wx)
           + b * (1 - wy) * wx
           + c * wy * (1 - wx)
           + d * wy * wx)

    # Min-max normalize to [0, 1] 
    img = (img - img.min()) / (np.ptp(img) + 1e-6)

    return img.astype(np.float32)[np.newaxis, :, :]   # (1, H, W)