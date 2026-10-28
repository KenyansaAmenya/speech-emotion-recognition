# Speech Emotion Recognition

A Python speech emotion recognition project with two training paths and a FastAPI inference service. It trains classifiers on acted speech from the RAVDESS dataset and predicts one of six emotion labels from an uploaded audio file.

> **Interpretation:** The output is a predicted dataset label. It does not measure a speaker’s true internal emotional state.

## What the project does

The repository supports this workflow:

1. Download RAVDESS speech audio and build metadata.
2. Assign actors to train, validation, and test splits so an actor’s recordings stay in one split.
3. Preprocess the audio and extract either MFCC features or log-mel spectrograms.
4. Train and evaluate baseline classifiers or a CNN.
5. Serve predictions through a FastAPI endpoint.

The API uses the MFCC pipeline saved as `models/ser_pipeline.joblib`. The CNN is a separate training artifact and is not currently loaded by the API.

## Emotion classes

The project maps RAVDESS emotion codes to six labels:

| RAVDESS code | Label |
| --- | --- |
| 1 | neutral |
| 3 | happy |
| 4 | sad |
| 5 | angry |
| 6 | fearful |
| 7 | disgust |

Other emotion codes in the dataset are skipped by the loader.

## Requirements

- Python 3.11 is used by the repository’s Docker image.
- Internet access is needed for the first RAVDESS download.
- Training requires the packages in `requirements.txt`.
- Serving requires the packages in `requirements-api.txt`.
- Inference requires a trained `ser_pipeline.joblib` file, either in `models/` locally or available from the configured Supabase Storage bucket.

The repository does not include the RAVDESS audio files or trained model artifacts.

## Setup

From the repository root, create a virtual environment and install the training dependencies:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an environment used only to run the API:

```bash
pip install -r requirements-api.txt
```

## Prepare the dataset

Run the data loader:

```bash
python -m src.data_loader
```

It downloads the RAVDESS speech archive into `data/raw/`, scans the WAV files, extracts metadata from their filenames, checks that the audio can be read, and writes `data/processed/metadata.csv`.

The split is speaker-independent: recordings from an actor are assigned together to train, validation, or test. The configured validation and test fractions are each 15%; the split uses random state 42 and stratifies actors by gender.

The loader also creates a class-distribution plot and reports dataset integrity information under `reports/`.

## Train the MFCC models

Train all configured baseline models:

```bash
python -m src.train
```

The available choices are `svm`, `random_forest`, `xgboost`, and `all`:

```bash
python -m src.train --model svm
python -m src.train --model random_forest
python -m src.train --model xgboost
```

The default `all` option trains the SVM, random forest, and XGBoost models when XGBoost is installed. The script evaluates each model on validation and test splits, selects the deployable model by validation macro-F1, and saves:

- Individual model files in `models/`, such as `svm_mfcc.joblib`.
- The selected inference payload at `models/ser_pipeline.joblib`.
- Metrics in `reports/ml_results.json`.

The saved pipeline includes the classifier, label mappings, feature settings, and training metadata. Inference checks that the feature settings saved with the model match `src/config.py`.

## Train the CNN
## Audio processing and features

The shared configuration is in `src/config.py`. Current defaults include:

- Sample rate: 22,050 Hz
- Audio duration: 3 seconds
- Silence-trimming threshold: 25 dB
- MFCC coefficients: 40
- FFT size: 2,048
- Hop length: 512
- CNN mel bins: 128
- CNN input image size: 128 × 128

Audio is loaded as mono, resampled, trimmed for silence, center-cropped or padded to the configured duration, and peak-normalized. The MFCC feature vector concatenates the time mean and standard deviation for each coefficient, producing 80 values with the default 40 coefficients.

The CNN path converts audio to a log-mel spectrogram, resizes it to a 128 × 128 image, and normalizes the values.

## Run the API

The API expects to be started from the repository root. Make sure `models/ser_pipeline.joblib` is present locally, or configure Supabase Storage so the API can download it during startup.

```bash
uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at <http://127.0.0.1:8000/docs>.

The service exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service information and route hints |
| `GET` | `/health` | Checks that the model pipeline can load |
| `POST` | `/predict` | Predicts an emotion from an uploaded audio file |
| `GET` | `/predictions` | Returns recent prediction history from Supabase |

### Make a prediction

The `POST /predict` endpoint expects a multipart form field named `file`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/audio.wav"
```

Supported filename extensions are `.wav`, `.mp3`, `.flac`, `.ogg`, and `.m4a`. Uploads are limited to 20 MB. The service rejects files it cannot decode, files shorter than 0.5 seconds, and silent audio.

A successful response contains the predicted emotion, the top three ranked class probabilities, the audio duration, and a disclaimer. Response shape:

```json
{
  "emotion": "happy",
  "top_k": [
    {"emotion": "happy", "probability": 0.72},
    {"emotion": "neutral", "probability": 0.18},
    {"emotion": "sad", "probability": 0.06}
  ],
  "duration_sec": 2.8,
  "disclaimer": "Predicted label reflects patterns in training data (RAVDESS acted speech), not the speaker's true internal state."
}
```

The values above are illustrative; actual output depends on the supplied audio and trained model.

Possible error status codes include:

- `413` — upload exceeds 20 MB.
- `422` — unsupported extension, undecodable audio, audio shorter than 0.5 seconds, silence, or prediction input error.
- `404` — temporary audio file is missing before prediction.
- `503` — the model is unavailable at the health endpoint.

## Configuration

### Audio and training settings

Change audio processing, feature extraction, emotion mappings, and split settings in `src/config.py`. The scripts also expose command-line settings; see their `--help` output.

### CORS
The API reads `FRONTEND_ORIGINS` as a comma-separated list of permitted browser origins. If unset, the code allows:

- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:8000`
- `https://tonelabb.vercel.app`

### Supabase model storage and prediction history

At startup, the API calls `ensure_model()`. If `models/ser_pipeline.joblib` is already present and non-empty, it uses that local file. Otherwise, it downloads the model from Supabase Storage.

The storage code uses these defaults:

| Setting | Default | Purpose |
| --- | --- | --- |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_SERVICE_KEY` or `SUPABASE_KEY` | — | Key used by the API’s Supabase client |
| `MODEL_BUCKET` | `models` | Storage bucket containing the model |
| `MODEL_PATH` | `ser_pipeline.joblib` | Model object path in the bucket |
| `LOCAL_MODEL_DIR` | `models` | Local directory for the downloaded model |
| `PREDICTION_TABLE` | `predictions` | Table used for prediction history |

The API attempts to log predictions to Supabase in a background task. A failure to log is printed but does not fail the prediction response. `GET /predictions` reads recent rows from the configured table. The repository does not define the table schema or database/storage policies; those must be configured in Supabase.

### Supabase key-name mismatch in the repository

There is a naming inconsistency to resolve when configuring Supabase:

- `.env.example` and `render.yaml` use `SUPABASE_SERVICE_ROLE_KEY`.
- `src/supabase_store.py`, used by the API, reads `SUPABASE_SERVICE_KEY` or falls back to `SUPABASE_KEY`.
- `scripts/upload_model.py` uses `SUPABASE_SERVICE_ROLE_KEY`.

For API model downloads, set one of the names the API actually reads, or update the code/config so the names match. Keep service-role credentials private and out of source control.

The checked-in `.env.example` currently contains placeholder values. Do not use those values as credentials.

## Docker and Render

The Dockerfile uses Python 3.11, installs system audio libraries (`libsndfile1` and `ffmpeg`), installs `requirements-api.txt`, and starts:

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT}
```

`PORT` defaults to 8000 in the container. The Render Blueprint in `render.yaml` defines a Docker web service named `ser-api` with `/health` as its health check. It declares `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as externally supplied environment values; see the key-name mismatch above before deployment.

## Repository layout

- `src/app/main.py` — FastAPI app and endpoints
- `src/config.py` — audio, feature, label, and split settings
- `src/data_loader.py` — RAVDESS download and metadata preparation
- `src/dataset.py` — feature/spectrogram datasets and caches
- `src/evaluate.py` — model evaluation
- `src/features.py` — MFCC and log-mel extraction
- `src/predict.py` — MFCC model loading and inference
- `src/preprocessing.py` — audio loading and normalization
- `src/supabase_store.py` — model download and prediction logging
- `src/train.py` and `src/train_cnn.py` — training workflows
- `scripts/upload_model.py` — uploads the trained MFCC pipeline to Supabase Storage
- `scripts/check_upload.py` — checks and loads the stored model
- `data/`, `models/`, `reports/` — downloaded or generated data, model artifacts, and reports

## Limitations and project-specific notes

- RAVDESS contains acted speech. Results on it do not establish performance on spontaneous speech or general populations.
- No accuracy or other performance figure is claimed here; training generates evaluation results in `reports/`.
- The repository does not include the dataset audio, generated metadata, or model files.
- The API loads the MFCC pipeline. The separately trained CNN is not available through the API code currently in the repository.

## How it works Video

https://github.com/user-attachments/assets/538c0977-4774-4147-9407-d67ac2819488


## License
Felix Amenya Kenyansa | Data science & Analytics
