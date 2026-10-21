import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from . import config
from .dataset import MelSpectrogramDataset, build_spectrogram_cache
from .evaluate import evaluate_model


class SmallCNN(nn.Module):

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.15))

        self.features = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), block(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


class _TorchAdapter:

    def __init__(self, model, device, batch_size=64):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            if X.ndim == 3:
                X = X[:, None, :, :]
            out = []
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).float().to(self.device)
                out.append(torch.softmax(self.model(xb), dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (0 is safe on Windows/CPU).")
    args = ap.parse_args()

    # Seeding 
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Metadata + spectrogram cache 
    df = pd.read_csv(config.DATA_PROCESSED / "metadata.csv")
    df = df[df["load_ok"]].reset_index(drop=True)

    print("Building spectrogram cache (first run only; cached afterwards)...")
    df = build_spectrogram_cache(df)
    print(f"Cached {len(df)} spectrograms.")

    # DataLoaders 
    loaders = {
        s: DataLoader(
            MelSpectrogramDataset(df[df["split"] == s]),
            batch_size=args.batch_size,
            shuffle=(s == "train"),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
            pin_memory=(device == "cuda"),
        )
        for s in ("train", "val", "test")
    }

    # Model, weighted loss (handles neutral 2:1 imbalance), optimizer 
    model = SmallCNN().to(device)

    y_train = df[df["split"] == "train"]["label_idx"].values
    cw = compute_class_weight(
        "balanced", classes=np.arange(config.NUM_CLASSES), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"Class weights: {np.round(cw, 3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    # Training loop
    best_val, best_state, patience_cnt = 0.0, copy.deepcopy(model.state_dict()), 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step(va_acc)

        elapsed = time.time() - t0
        eta = (elapsed / ep) * (args.epochs - ep)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f} "
              f"| {elapsed:.0f}s elapsed, {eta:.0f}s ETA")

        if va_acc > best_val:
            best_val, patience_cnt = va_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {ep}.")
                break

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state,
        "label2idx": config.LABEL2IDX,
        "idx2label": config.IDX2LABEL,
        "feature_settings": {
            "sample_rate": config.SAMPLE_RATE,
            "duration_sec": config.DURATION_SEC,
            "n_mels": config.N_MELS_CNN,
            "image_size": config.CNN_IMAGE_SIZE,
        },
        "best_val_acc": best_val,
    }, config.MODELS_DIR / "cnn_mel.pt")
    print(f"\nSaved model -> {config.MODELS_DIR / 'cnn_mel.pt'} (best val acc {best_val:.4f})")

    # Test evaluation through the model
    Xte, yte = [], []
    for xb, yb in loaders["test"]:
        Xte.append(xb.numpy())
        yte.append(yb.numpy())
    Xte = np.concatenate(Xte, axis=0)
    yte = np.concatenate(yte, axis=0)

    adapter = _TorchAdapter(model, device)
    res = evaluate_model(adapter, Xte, yte, title="cnn_mel (test)")
    print(f"\nCNN test macro-F1: {res['macro_f1']:.4f}")
    print(f"CNN test accuracy: {res['accuracy']:.4f}")

    with open(config.REPORTS_DIR / "cnn_results.json", "w") as f:
        json.dump({"test": res, "best_val_acc": best_val}, f, indent=2)
    print(f"Saved results -> {config.REPORTS_DIR / 'cnn_results.json'}")


if __name__ == "__main__":
    main()