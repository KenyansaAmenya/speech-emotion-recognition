FROM python:3.11-slim

# System packages needed by librosa/soundfile/torchaudio 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first 
COPY requirements-api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY .env.example ./

# Create output dirs so config.py doesn't complain at import time 
RUN mkdir -p data/raw data/processed models reports

# Render injects $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

# exec: pass SIGTERM to uvicorn 
CMD ["sh", "-c", "exec uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT}"]