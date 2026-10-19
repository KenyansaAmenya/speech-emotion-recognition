import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

RAVDESS_URLS = [
    f"https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
]

FILENAME_RE = re.compile(
    r"(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav"
)

def download_ravdess(dest_dir: Path = config.DATA_RAW) -> None:
    import requests
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ravdess_speech.zip"
    if zip_path.exists() or list(dest_dir.rglob("*.wav")):
        print("RAVDESS already present, skipping download.")
        return
    url = RAVDESS_URLS[0]
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        zf.extractall(dest_dir)
    print(f"Done. WAV files extracted to {dest_dir}")


def parse_filename(path: Path) -> dict | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    emotion_code = int(m.group("emotion"))
    if emotion_code not in config.EMOTION_MAP:   
        return None
    actor = int(m.group("actor"))
    return {
        "file_path": str(path),
        "emotion": config.EMOTION_MAP[emotion_code],
        "emotion_code": emotion_code,
        "actor_id": actor,
        "actor_gender": "female" if actor % 2 == 0 else "male",
        "intensity": "strong" if m.group("intensity") == "02" else "normal",
        "statement": int(m.group("statement")),
        "repetition": int(m.group("repetition")),
    }


def load_metadata(raw_dir: Path = config.DATA_RAW) -> pd.DataFrame:
    import soundfile as sf

    records = []
    for p in sorted(raw_dir.rglob("*.wav")):
        rec = parse_filename(p)
        if rec is None:
            continue
        try:
            info = sf.info(str(p))
            rec["duration_sec"] = info.duration
            rec["sample_rate"] = info.samplerate
            rec["load_ok"] = True
        except Exception as e:
            rec["duration_sec"] = np.nan
            rec["sample_rate"] = np.nan
            rec["load_ok"] = False
            rec["error"] = str(e)
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No usable WAV files found under {raw_dir}")
    df["label_idx"] = df["emotion"].map(config.LABEL2IDX)
    return df

# Class distribution plot + integrity report.
def basic_eda(df: pd.DataFrame, reports_dir: Path = config.REPORTS_DIR) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n=== Class distribution ===")
    print(df["emotion"].value_counts())

    print("\n=== Per-actor counts (should be equal for a clean dataset) ===")
    print(df.groupby(["actor_id", "emotion"]).size().unstack(fill_value=0))

    bad = df[~df["load_ok"]]
    print(f"\nCorrupted/unloadable files: {len(bad)}")
    if len(bad):
        print(bad[["file_path", "error"]])

    dupes = df[df.duplicated(subset=["actor_id", "emotion", "statement",
                                     "repetition", "intensity"], keep=False)]
    print(f"Duplicate (actor, emotion, statement, rep) combos: {len(dupes)}")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="emotion", order=sorted(df["emotion"].unique()))
    plt.title("RAVDESS class distribution (six-class task)")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to {reports_dir / 'class_distribution.png'}")


# Assign each actor to exactly ONE split. ~70/15/15, both genders per split.
def speaker_independent_split(df: pd.DataFrame):
   
    actors = df[["actor_id", "actor_gender"]].drop_duplicates().sort_values("actor_id")
    train_a, temp_a = train_test_split(
        actors, test_size=(config.VAL_SIZE + config.TEST_SIZE),
        stratify=actors["actor_gender"], random_state=config.RANDOM_STATE)
    val_frac = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    val_a, test_a = train_test_split(
        temp_a, test_size=1 - val_frac,
        stratify=temp_a["actor_gender"], random_state=config.RANDOM_STATE)

    def tag(sub, name):
        sub = sub.copy()
        sub["split"] = name
        return sub

    splits = pd.concat([tag(train_a, "train"), tag(val_a, "val"), tag(test_a, "test")])
    df = df.merge(splits[["actor_id", "split"]], on="actor_id", how="left")

    print("\n=== Speaker-independent split ===")
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        print(f"{name:5s}: {sub['actor_id'].nunique():2d} actors, "
              f"{len(sub):4d} samples, emotions: {sorted(sub['emotion'].unique())}")
    return df


if __name__ == "__main__":
    download_ravdess()
    df = load_metadata()
    basic_eda(df)
    df = speaker_independent_split(df)
    df.to_csv(config.DATA_PROCESSED / "metadata.csv", index=False)
    print(f"\nSaved metadata -> {config.DATA_PROCESSED / 'metadata.csv'}")