# MFCC (tabular baseline) and Mel spectrograms (CNN).
import numpy as np
import librosa
import librosa.display  # noqa: F401  (registers display helpers)

from . import config


# MFCC mean + std across time -> 2*n_mfcc dimensional vector.
def extract_mfcc(waveform: np.ndarray,
                 sr: int = config.SAMPLE_RATE,
                 n_mfcc: int = config.N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc,
                                n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feats.astype(np.float32)

def mfcc_pipeline(path: str) -> np.ndarray:
    from .preprocessing import preprocess
    return extract_mfcc(preprocess(path))

# Log-Mel spectrogram -> (n_mels, time) array for the CNN.
def extract_melspectrogram(waveform: np.ndarray,
                           sr: int = config.SAMPLE_RATE,
                           n_mels: int = config.N_MELS_CNN) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels,
                                         n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)

def spectrogram_to_image(log_mel: np.ndarray, img_size: int = config.CNN_IMAGE_SIZE) -> np.ndarray:
    import cv2  
    img = cv2.resize(log_mel, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = (img - img.min()) / (img.ptp() + 1e-6)
    return img.astype(np.float32)[np.newaxis, :, :]