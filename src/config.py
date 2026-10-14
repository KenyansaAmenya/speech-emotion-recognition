from pathlib import Path

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

#Audio
SAMPLE_RATE = 22050
DURATION_SEC = 3.0          # fixed length: pad/trim to this
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
TOP_DB_TRIM = 25            # silence trimming threshold

# Features
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MELS_CNN = 128
CNN_IMAGE_SIZE = 128        # mel spectrogram resized to CNN_IMAGE_SIZE x CNN_IMAGE_SIZE

#RAVDESS emotion mapping (six-class task)
EMOTION_MAP = {
    1: "neutral",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
}

LABEL2IDX = {label: i for i, label in enumerate(sorted(EMOTION_MAP.values()))}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(LABEL2IDX)

# --- Split ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42