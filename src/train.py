import argparse
import datetime
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .dataset import build_feature_matrix
from .evaluate import evaluate_model


def get_models() -> dict:
    return {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=config.RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=config.RANDOM_STATE),
        "xgboost": _xgb(),
    }


def _xgb():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed — skipping XGBoost.")
        return None
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, tree_method="hist",
        objective="multi:softprob", num_class=config.NUM_CLASSES,
        eval_metric="mlogloss", random_state=config.RANDOM_STATE, n_jobs=-1)

# Convert NumPy scalars/arrays to JSON-serializable Python types.
def _to_py(obj):
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["svm", "random_forest", "xgboost", "all"])
    args = ap.parse_args()

    df = pd.read_csv(config.DATA_PROCESSED / "metadata.csv")
    X, y, splits = build_feature_matrix(df, config.DATA_PROCESSED / "mfcc_cache.npz")

    Xtr, ytr = X[splits == "train"], y[splits == "train"]
    Xva, yva = X[splits == "val"], y[splits == "val"]
    Xte, yte = X[splits == "test"], y[splits == "test"]

    classes = np.arange(config.NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)

    models = get_models()
    results = {}
    for name, model in models.items():
        if model is None:
            continue
        if args.model != "all" and name != args.model:
            continue

        print(f"\n=== Training {name} ===")
        if name == "xgboost":
            model.fit(Xtr, ytr, sample_weight=weights[ytr])
        else:
            model.fit(Xtr, ytr)

        res_va = evaluate_model(model, Xva, yva, title=f"{name} (val)")
        res_te = evaluate_model(model, Xte, yte, title=f"{name} (test)")

        results[name] = {"val": res_va, "test": res_te}
        joblib.dump(model, config.MODELS_DIR / f"{name}_mfcc.joblib")

    if not results:
        raise RuntimeError("No models trained successfully.")

    # Persist the best model by val macro-F1 as the deployable pipeline
    best = max(results, key=lambda k: results[k]["val"]["macro_f1"])
    print(f"\nBest model: {best} (val macro-F1 = {results[best]['val']['macro_f1']:.3f})")

    payload = {
        "model": models[best],
        "label2idx": config.LABEL2IDX,
        "idx2label": config.IDX2LABEL,
        "feature_settings": {
            "sample_rate": config.SAMPLE_RATE,
            "duration_sec": config.DURATION_SEC,
            "top_db": config.TOP_DB_TRIM,
            "n_mfcc": config.N_MFCC,
        },
        "metadata": {
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "trained_at": datetime.datetime.now().isoformat(),
            "random_state": config.RANDOM_STATE,
            "selected_by": "val_macro_f1",
        },
    }
    joblib.dump(payload, config.MODELS_DIR / "ser_pipeline.joblib")

    with open(config.REPORTS_DIR / "ml_results.json", "w") as f:
        json.dump(_to_py(results), f, indent=2)

    print(f"Saved pipeline -> {config.MODELS_DIR / 'ser_pipeline.joblib'}")
    print(f"Saved results  -> {config.REPORTS_DIR / 'ml_results.json'}")


if __name__ == "__main__":
    main()