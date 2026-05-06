from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.features.preprocessing import build_preprocessor, ALL_FEATURES
from src.utils.config import MODEL_DIR, load_all_configs

CONFIG = load_all_configs()["model"]


def train_models(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["is_problematic"], random_state=CONFIG["random_state"])
    X_train = train_df[ALL_FEATURES]
    y_train = train_df["is_problematic"]
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)

    iso_cfg = CONFIG["isolation_forest"]
    iso = IsolationForest(
        n_estimators=iso_cfg["n_estimators"],
        contamination=iso_cfg["contamination"],
        random_state=CONFIG["random_state"],
    )
    iso.fit(X_train_proc)

    rf_cfg = CONFIG["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        class_weight=rf_cfg["class_weight"],
        random_state=CONFIG["random_state"],
    )
    clf.fit(X_train_proc, y_train)

    return {"preprocessor": preprocessor, "isolation_forest": iso, "classifier": clf, "train_rows": len(train_df), "test_rows": len(test_df)}


def save_models(bundle: dict, model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ["preprocessor", "isolation_forest", "classifier"]:
        joblib.dump(bundle[name], model_dir / f"{name}.joblib")
    metadata = {"train_rows": bundle["train_rows"], "test_rows": bundle["test_rows"]}
    joblib.dump(metadata, model_dir / "metadata.joblib")
