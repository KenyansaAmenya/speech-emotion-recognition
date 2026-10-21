import os
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.supabase_store import ensure_model, get_client, log_prediction
from src import config

print("=" * 60)
print("TEST 1: get_client()")
print("=" * 60)
client = get_client()
assert client is not None, "Client is None — env vars missing"
print(f"✅ Client created")

print("\n" + "=" * 60)
print("TEST 2: ensure_model() — download from Supabase")
print("=" * 60)

# Force a fresh download by deleting the local copy
if config.MODELS_DIR.joinpath("ser_pipeline.joblib").exists():
    config.MODELS_DIR.joinpath("ser_pipeline.joblib").unlink()
    print("Deleted local copy to force download")

model_path = ensure_model()
print(f"✅ Model downloaded to: {model_path}")
print(f"   Size: {model_path.stat().st_size:,} bytes")

print("\n" + "=" * 60)
print("TEST 3: ensure_model() again — should use cache")
print("=" * 60)
model_path2 = ensure_model()
assert model_path == model_path2
print(f"✅ Cache hit (no re-download)")

print("\n" + "=" * 60)
print("TEST 4: log_prediction() — insert a row")
print("=" * 60)
log_prediction(
    emotion="happy",
    top_k=[
        {"emotion": "happy", "probability": 0.85},
        {"emotion": "sad", "probability": 0.10},
        {"emotion": "angry", "probability": 0.05},
    ],
    duration_sec=2.5,
    model_name="__test_script__",
)
print("✅ log_prediction() completed without error")

print("\n" + "=" * 60)
print("TEST 5: Verify the row in Supabase")
print("=" * 60)
rows = (
    client.table("predictions")
    .select("*")
    .eq("model_name", "__test_script__")
    .execute()
)
assert len(rows.data) >= 1, f"Expected >=1 row, got {len(rows.data)}"
row = rows.data[0]
print(f"✅ Row found: id={row['id']}, emotion={row['emotion']}, duration={row['duration_sec']}")
print(f"   top_k: {row['top_k']}")
print(f"   created_at: {row['created_at']}")

# Clean up
client.table("predictions").delete().eq("model_name", "__test_script__").execute()
print("✅ Test row deleted")

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED — supabase_store.py is working correctly")
print("=" * 60)