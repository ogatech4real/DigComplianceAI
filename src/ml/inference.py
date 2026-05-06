from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.features.preprocessing import ALL_FEATURES
from src.utils.config import MODEL_DIR, load_all_configs

CONFIG = load_all_configs()["model"]


def load_models(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    return {
        "preprocessor": joblib.load(model_dir / "preprocessor.joblib"),
        "isolation_forest": joblib.load(model_dir / "isolation_forest.joblib"),
        "classifier": joblib.load(model_dir / "classifier.joblib"),
    }


def normalize(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or np.isclose(smax - smin, 0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - smin) / (smax - smin)


def assign_risk_label(score: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(score >= 0.66, "high", np.where(score >= 0.33, "medium", "low")),
        index=score.index,
    )

def apply_models(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = df.copy()

    weights = CONFIG.get(
        "hybrid_weights",
        {"rf": 0.3, "anomaly": 0.3, "rules": 0.2, "interaction": 0.2},
    )
    threshold = CONFIG.get("hybrid_threshold", 0.4)

    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = 0

    X = out[ALL_FEATURES]
    X_proc = models["preprocessor"].transform(X)

    # Rule layer
    out["rule_score"] = 1 - np.exp(-out["rule_flag_count"].astype(float))
    out["rule_pred"] = (out["rule_flag_count"] > 0).astype(int)

    # Isolation Forest
    out["if_raw"] = -models["isolation_forest"].decision_function(X_proc)
    out["if_score"] = normalize(np.clip(out["if_raw"], 1e-6, None))
    out["if_pred"] = (out["if_score"] >= 0.5).astype(int)

    # Random Forest
    out["rf_probability"] = models["classifier"].predict_proba(X_proc)[:, 1]
    out["rf_pred"] = (out["rf_probability"] >= 0.5).astype(int)

    # ML-only composite (aligned naming)
    out["ml_only_score"] = 0.3 * out["if_score"] + 0.7 * out["rf_probability"]
    out["ml_only_pred"] = (out["ml_only_score"] >= 0.5).astype(int)

    # Ablation components
    out["rules_if_score"] = 0.5 * out["rule_score"] + 0.5 * out["if_score"]
    out["rules_if_pred"] = (out["rules_if_score"] >= 0.5).astype(int)

    out["rules_rf_score"] = 0.5 * out["rule_score"] + 0.5 * out["rf_probability"]
    out["rules_rf_pred"] = (out["rules_rf_score"] >= 0.5).astype(int)

    # Hybrid (CONFIG-DRIVEN)
    out["hybrid_score"] = (
        weights["rf"] * out["rf_probability"] +
        weights["anomaly"] * out["if_score"] +
        weights["rules"] * out["rule_score"] +
        weights["interaction"] * (out["rf_probability"] * out["rule_score"])
    )

    out["hybrid_pred"] = (out["hybrid_score"] >= threshold).astype(int)
    out["hybrid_risk"] = assign_risk_label(out["hybrid_score"])

    # Cleanup
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(0)

    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    out[obj_cols] = out[obj_cols].fillna("unknown")

    return out