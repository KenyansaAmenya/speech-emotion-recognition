"""Verify the model artifact is present in Supabase Storage."""
import io
import os

import httpx
import joblib
from supabase import create_client
from supabase.lib.client_options import ClientOptions

# 1. Create a dedicated httpx client for STORAGE with HTTP/1.1
storage_httpx = httpx.Client(http2=False, timeout=httpx.Timeout(30.0, connect=10.0))

# 2. Create the Supabase client
client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    options=ClientOptions(httpx_client=storage_httpx),
)

# 3. CRITICAL: Re-bind the storage sub-client to use the same HTTP/1.1 client
#    (This prevents it from sharing a client with PostgREST if you use one)
try:
    client._storage = client._init_storage_client(
        storage_url=client.storage_url,
        headers=client.options.headers,
        storage_client_timeout=client.options.storage_client_timeout,
        http_client=storage_httpx,
    )
except Exception as e:
    print(f"Warning: could not isolate storage client: {e}")

# --- Your check logic (now using HTTP/1.1) ---
files = client.storage.from_("models").list()
print("Files in 'models' bucket:")
for f in files:
    size = f.get("metadata", {}).get("size", "?")
    print(f"  {f['name']}  —  {size} bytes")

data = client.storage.from_("models").download("ser_pipeline.joblib")
print(f"\nDownloaded {len(data):,} bytes")

payload = joblib.load(io.BytesIO(data))
print(f"Model type: {type(payload['model']).__name__}")
print("✅ Upload verified — Render will be able to load this model.")