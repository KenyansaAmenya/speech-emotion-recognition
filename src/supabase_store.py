from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY")          # preferred
    or os.environ.get("SUPABASE_KEY", "")           # fallback
)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "models")
MODEL_PATH = os.environ.get("MODEL_PATH", "ser_pipeline.joblib")

LOCAL_MODEL_DIR = Path(os.environ.get("LOCAL_MODEL_DIR", "models"))
LOCAL_MODEL_FILE = LOCAL_MODEL_DIR / "ser_pipeline.joblib"

PREDICTION_TABLE = os.environ.get("PREDICTION_TABLE", "predictions")


_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set. "
            "Add them in the Render dashboard under Environment."
        )

    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info(
        "Supabase client initialised (url=%s, key=service_role=%s)",
        SUPABASE_URL,
        SUPABASE_KEY.startswith("eyJ") and "jwt" or "unknown",
    )
    return _client

def _download_with_retry(
    client: Client,
    bucket: str,
    path: str,
    retries: int = 3,
    timeout: float = 30.0,
) -> bytes:

    storage = client.storage.from_(bucket)
    try:
        storage._client.base_url = f"{SUPABASE_URL}/storage/v1"
    except Exception:
        pass 

    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "Downloading supabase://%s/%s (attempt %d/%d)",
                bucket, path, attempt, retries,
            )
            return storage.download(path)

        except KeyError as e:
            last_exc = e
            raise FileNotFoundError(
                f"Model not found in Supabase Storage: "
                f"bucket='{bucket}', path='{path}'. "
                f"Verify: (1) the file exists in the bucket, "
                f"(2) RLS on storage.objects allows service_role SELECT."
            ) from e

        except Exception as e:
            msg = str(e).lower()
            last_exc = e

            # Real 404 from HTTP layer
            if "404" in msg or "not found" in msg:
                raise FileNotFoundError(
                    f"Model not found in Supabase Storage: "
                    f"bucket='{bucket}', path='{path}'."
                ) from e

            # Real 401/403
            if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
                raise PermissionError(
                    f"Supabase denied access to bucket='{bucket}'. "
                    f"Check that SUPABASE_SERVICE_KEY is the service_role key "
                    f"(not the anon key) and that bucket RLS permits SELECT."
                ) from e

            # Transient — retry with exponential backoff
            if attempt < retries:
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "Storage download failed (attempt %d/%d): %s — retrying in %ss",
                    attempt, retries, e, delay,
                )
                time.sleep(delay)
                continue

    raise RuntimeError(
        f"Storage download failed after {retries} attempts: {last_exc}"
    ) from last_exc


def _log_bucket_contents(client: Client, bucket: str) -> None:
    try:
        files = client.storage.from_(bucket).list()
        if not files:
            logger.error(
                "Bucket '%s' is EMPTY. Upload '%s' via the Supabase Dashboard "
                "→ Storage → %s → Upload file.",
                bucket, MODEL_PATH, bucket,
            )
            return

        names = [f.get("name") for f in files if isinstance(f, dict)]
        logger.error(
            "Bucket '%s' contains %d item(s): %s",
            bucket, len(files), names[:20],
        )
        if MODEL_PATH not in names:
            logger.error(
                "Expected file '%s' is NOT in bucket '%s'. "
                "Upload it at the exact root path.",
                MODEL_PATH, bucket,
            )
    except Exception as e:
        logger.error("Could not list bucket '%s': %s", bucket, e)


def ensure_model() -> Path:

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Already cached locally
    if LOCAL_MODEL_FILE.exists() and LOCAL_MODEL_FILE.stat().st_size > 0:
        logger.info(
            "Model already present locally at %s (%d bytes)",
            LOCAL_MODEL_FILE, LOCAL_MODEL_FILE.stat().st_size,
        )
        return LOCAL_MODEL_FILE

    client = _get_client()

    try:
        data = _download_with_retry(client, MODEL_BUCKET, MODEL_PATH)
    except FileNotFoundError:
        _log_bucket_contents(client, MODEL_BUCKET)
        raise

    if not data:
        raise RuntimeError(
            f"Downloaded '{MODEL_PATH}' from bucket '{MODEL_BUCKET}' but it was empty."
        )

    LOCAL_MODEL_FILE.write_bytes(data)
    logger.info(
        "Model downloaded to %s (%d bytes)",
        LOCAL_MODEL_FILE, len(data),
    )
    return LOCAL_MODEL_FILE

# Prediction logging 
def log_prediction(
    emotion: str,
    top_k: list[dict],
    duration_sec: float,
    customer_id: Optional[str] = None,
) -> None:

    try:
        client = _get_client()
        payload = {
            "emotion": emotion,
            "top_k": top_k,
            "duration_sec": duration_sec,
        }
        if customer_id:
            payload["customer_id"] = customer_id

        client.table(PREDICTION_TABLE).insert(payload).execute()
        logger.debug("Prediction logged: %s", emotion)
    except Exception as e:
        logger.warning("Failed to log prediction to Supabase: %s", e)