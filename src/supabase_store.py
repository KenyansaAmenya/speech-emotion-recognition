import json
import os
import tempfile
import time
from pathlib import Path

import httpx
import joblib
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

from . import config

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
MODEL_BUCKET = os.getenv("SUPABASE_MODEL_BUCKET", "models")
MODEL_PATH = os.getenv("SUPABASE_MODEL_PATH", "ser_pipeline.joblib")
LOCAL_MODEL_PATH = config.MODELS_DIR / "ser_pipeline.joblib"

_client: Client | None = None


def _make_httpx() -> httpx.Client:
    return httpx.Client(
        http2=False,
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(
            max_keepalive_connections=10,
            max_connections=40,
            keepalive_expiry=15.0,
        ),
    )


def get_client() -> Client | None:
    global _client
    if _client is not None:
        return _client
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        return None

    _httpx_client = _make_httpx()
    _client = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
        options=ClientOptions(httpx_client=_httpx_client),
    )

    _storage_httpx_client = _make_httpx()
    try:
        _client._storage = _client._init_storage_client(
            storage_url=_client.storage_url,
            headers=_client.options.headers,
            storage_client_timeout=_client.options.storage_client_timeout,
            http_client=_storage_httpx_client,
        )
    except Exception as e:
        print(f"[supabase] WARNING: could not isolate storage client ({e})")

    return _client


def _download_with_retry(client: Client, bucket: str, path: str,
                         attempts: int = 3) -> bytes:
    last_exc = None
    for i in range(attempts):
        try:
            return client.storage.from_(bucket).download(path)
        except (httpx.RemoteProtocolError, httpx.ConnectError,
                httpx.ReadError, httpx.WriteError) as e:
            last_exc = e
            if i < attempts - 1:
                wait = 2 ** i
                print(f"[supabase] download failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
        except Exception as e:
            # Non-transient error — don't retry
            raise RuntimeError(f"Download failed: {e}") from e
    raise RuntimeError(
        f"Failed to download {bucket}/{path} after {attempts} attempts: {last_exc}")


def ensure_model() -> Path:
    if LOCAL_MODEL_PATH.exists():
        try:
            joblib.load(LOCAL_MODEL_PATH)
            return LOCAL_MODEL_PATH
        except Exception as e:
            print(f"[supabase] cached model is corrupt ({e}); re-downloading...")
            LOCAL_MODEL_PATH.unlink(missing_ok=True)

    client = get_client()
    if client is None:
        raise RuntimeError(
            "Model not found locally and Supabase is not configured. "
            "Set SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY, or place "
            f"ser_pipeline.joblib at {LOCAL_MODEL_PATH}")

    print(f"Downloading model from supabase://{MODEL_BUCKET}/{MODEL_PATH} ...")
    data = _download_with_retry(client, MODEL_BUCKET, MODEL_PATH)

    if not data:
        raise RuntimeError(f"Downloaded model is empty (0 bytes): {MODEL_PATH}")

    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="wb", dir=LOCAL_MODEL_PATH.parent, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        joblib.load(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model failed validation: {e}")

    tmp_path.replace(LOCAL_MODEL_PATH)
    print(f"Model cached at {LOCAL_MODEL_PATH} ({len(data):,} bytes)")
    return LOCAL_MODEL_PATH


def log_prediction(emotion: str, top_k: list, duration_sec: float,
                   model_name: str = "ser_pipeline") -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table("predictions").insert({
            "emotion": emotion,
            "top_k": json.dumps(top_k),
            "duration_sec": duration_sec,
            "model_name": model_name,
        }).execute()
    except Exception as e:
        print(f"[supabase] failed to log prediction: {e}")