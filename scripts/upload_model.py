import os
import httpx
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions

# Create the httpx client with a longer read timeout
http_client = httpx.Client(timeout=httpx.Timeout(10.0, read=300.0))

# Wrap it in SyncClientOptions — NOT a dict
options = SyncClientOptions(httpx_client=http_client)

client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    options=options
)

with open("models/ser_pipeline.joblib", "rb") as f:
    client.storage.from_("models").upload(
        "ser_pipeline.joblib", f.read(),
        {"content-type": "application/octet-stream", "upsert": "true"})
print("Uploaded.")