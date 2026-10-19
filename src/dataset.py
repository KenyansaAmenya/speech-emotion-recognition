from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from . import config
from .features import mfcc_pipeline, extract_melspectrogram, spectrogram_to_image
from .preprocessing import preprocess


# MFCC tabular features (for sklearn / XGBoost)
def build_feature_matrix(df: pd.DataFrame, cache_path: Path | None = None):
    df = df[df["load_ok"]].reset_index(drop=True)

    if cache_path and Path(cache_path).exists():
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"], data["splits"]

    X, y, splits = [], [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting MFCC"):
        try:
            X.append(mfcc_pipeline(row.file_path))
            y.append(row.label_idx)
            splits.append(row.split)
        except Exception as e:
            print(f"Skipping {row.file_path}: {e}")

    if not X:
        raise RuntimeError(
            f"All {len(df)} files failed feature extraction. "
            f"First path: {df.iloc[0]['file_path']}"
        )

    X, y = np.vstack(X), np.array(y)
    splits = np.array(splits)
    if cache_path:
        np.savez_compressed(cache_path, X=X, y=y, splits=splits)
    return X, y, splits


# Mel-spectrogram cache (for CNN)
def build_spectrogram_cache(df: pd.DataFrame,
                            cache_dir: Path | None = None) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = config.DATA_PROCESSED / "spec_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["load_ok"]].reset_index(drop=True)
    spec_paths = []
    for row in tqdm(df.itertuples(index=False), total=len(df),
                    desc="Building spectrogram cache"):
        stem = Path(row.file_path).stem
        out = cache_dir / f"{stem}.npy"
        if not out.exists():
            wave = preprocess(row.file_path)
            img = spectrogram_to_image(extract_melspectrogram(wave))
            np.save(out, img)
        spec_paths.append(str(out))

    df = df.copy()
    df["spec_path"] = spec_paths
    return df


class MelSpectrogramDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        df = df[df["load_ok"]].reset_index(drop=True)
        assert "spec_path" in df.columns, (
            "Run build_spectrogram_cache(df) before constructing the dataset."
        )
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = np.load(row["spec_path"])                # (1, H, W) float32
        return torch.from_numpy(img), int(row["label_idx"])