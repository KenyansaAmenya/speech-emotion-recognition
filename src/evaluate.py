# Evaluation utilities: accuracy, macro F1, per-class report, confusion matrix.
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from . import config

def _sanitize(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")

def evaluate_model(model, X, y, title: str = "model",
                   reports_dir: Path | None = None,
                   save_plot: bool = True) -> dict:

    reports_dir = reports_dir or config.REPORTS_DIR
    label_names = [config.IDX2LABEL[i] for i in range(config.NUM_CLASSES)]

    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y, y_pred, target_names=label_names,
        output_dict=True, zero_division=0)

    print(f"\n----- {title} -----")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(classification_report(y, y_pred, target_names=label_names, zero_division=0))

    cm = confusion_matrix(y, y_pred)

    if save_plot:
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Confusion matrix — {title}")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(reports_dir / f"cm_{_sanitize(title)}.png", dpi=150)
        plt.close()

    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "mean_max_proba": float(proba.max(axis=1).mean()) if proba is not None else None,
    }