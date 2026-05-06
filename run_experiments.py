from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.generator import generate_dataset
from src.data.io import load_tabular_file, save_dataframe
from src.rules.rule_engine import run_rule_engine
from src.features.preprocessing import (
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
)
from src.utils.config import MODEL_DIR, load_all_configs
from src.data.mapping import canonicalise_dataframe
from src.features.preprocessing import enrich_from_icc


CONFIG = load_all_configs()
MODEL_CONFIG = CONFIG["model"]
RANDOM_STATE = MODEL_CONFIG["random_state"]

DEFAULT_HYBRID_WEIGHTS = MODEL_CONFIG.get(
    "hybrid_weights",
    {"rf": 0.3, "anomaly": 0.3, "rules": 0.2, "interaction": 0.2},
)
DEFAULT_HYBRID_THRESHOLD = MODEL_CONFIG.get("hybrid_threshold", 0.4)

DEFAULT_DATA_PATH = ROOT / "data" / "synthetic" / "synthetic_trade_records.csv"
DEFAULT_RESULTS_DIR = ROOT / "results"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_robust_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERICAL_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
        ]
    )


def minmax_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or np.isclose(smax - smin, 0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - smin) / (smax - smin)


def compute_top_k_detection(y_true: pd.Series, scores: pd.Series, k: float = 0.2) -> float:
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.0
    n_top = max(1, int(len(scores) * k))
    top_idx = np.argsort(scores.to_numpy())[::-1][:n_top]
    return float(y_true.iloc[top_idx].sum() / y_true.sum())


def metrics_dict(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> Dict[str, float]:
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "top_10_detection_rate": float(compute_top_k_detection(y_true, y_score, k=0.1)),
        "top_20_detection_rate": float(compute_top_k_detection(y_true, y_score, k=0.2)),
        "top_30_detection_rate": float(compute_top_k_detection(y_true, y_score, k=0.3)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


def assign_risk_label(score: pd.Series, low_th: float = 0.33, high_th: float = 0.66) -> pd.Series:
    return pd.Series(
        np.where(score >= high_th, "high", np.where(score >= low_th, "medium", "low")),
        index=score.index,
    )


def get_feature_importance_df(preprocessor, classifier) -> pd.DataFrame:
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(classifier.feature_importances_))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.reset_index(drop=True)


def save_plot_roc(y_true: pd.Series, score_map: Dict[str, pd.Series], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, scores in score_map.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = roc_auc_score(y_true, scores)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_plot_pr(y_true: pd.Series, score_map: Dict[str, pd.Series], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_plot_topk(y_true: pd.Series, score_map: Dict[str, pd.Series], output_path: Path) -> None:
    ks = np.arange(0.05, 0.55, 0.05)
    plt.figure(figsize=(8, 6))
    for name, scores in score_map.items():
        vals = [compute_top_k_detection(y_true, scores, k=float(k)) for k in ks]
        plt.plot(ks * 100, vals, marker="o", label=name)

    plt.axvline(x=10, linestyle="--", alpha=0.3)
    plt.axvline(x=20, linestyle="--", alpha=0.3)
    plt.axvline(x=30, linestyle="--", alpha=0.3)

    plt.xlabel("Top-k Reviewed (%)")
    plt.ylabel("Detection Rate")
    plt.title("Top-k Detection Efficiency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_or_generate_dataset(input_path: Path, n_records: int, force_generate: bool) -> pd.DataFrame:
    if force_generate or not input_path.exists():
        df = generate_dataset(n_records)
        save_dataframe(df, input_path)
        print(f"[INFO] Generated dataset at: {input_path}")
        return df

    print(f"[INFO] Loading dataset from: {input_path}")
    return load_tabular_file(input_path)

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # 🔴 Preserve original labels BEFORE canonicalisation
    original_labels = {}
    preserve_cols = [
        "record_id",
        "is_problematic",
        "anomaly_class",
        "anomaly_count",
    ]

    for col in preserve_cols:
        if col in df.columns:
            original_labels[col] = df[col].copy()

    # 🔹 Canonical pipeline
    df, _ = canonicalise_dataframe(df)
    df = enrich_from_icc(df)
    df = run_rule_engine(df.copy())

    # 🔴 Restore original labels after canonicalisation
    for col, values in original_labels.items():
        df[col] = values.values

    # 🔴 Proper label cleanup (NO LEAKAGE)
    df["is_problematic"] = (
        pd.to_numeric(df["is_problematic"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["anomaly_class"] = (
        df["anomaly_class"]
        .fillna("none")
        .astype(str)
    )

    df["anomaly_count"] = (
        pd.to_numeric(df["anomaly_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # 🔴 Verify anomaly diversity survives
    print("\n=== Label Distribution ===")
    print(df["is_problematic"].value_counts(dropna=False))

    print("\n=== Anomaly Class Distribution ===")
    print(df["anomaly_class"].value_counts(dropna=False))

    missing_cols = [
        col for col in ALL_FEATURES + ["is_problematic", "anomaly_class"]
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df

def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["is_problematic"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["is_problematic"],
        random_state=RANDOM_STATE,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_models(train_df: pd.DataFrame):
    X_train = train_df[ALL_FEATURES]
    y_train = train_df["is_problematic"]

    # 🔴 Safety check
    if y_train.nunique() < 2:
        raise ValueError(
            "Training labels contain only one class."
        )

    preprocessor = build_robust_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)

    iso_cfg = MODEL_CONFIG["isolation_forest"]
    rf_cfg = MODEL_CONFIG["random_forest"]

    isolation_forest = IsolationForest(
        n_estimators=iso_cfg["n_estimators"],
        contamination=iso_cfg["contamination"],
        random_state=RANDOM_STATE,
    )
    isolation_forest.fit(X_train_proc)

    classifier = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        class_weight=rf_cfg["class_weight"],
        random_state=RANDOM_STATE,
    )
    classifier.fit(X_train_proc, y_train)

    return preprocessor, isolation_forest, classifier


def score_dataframe(df: pd.DataFrame, preprocessor, isolation_forest, classifier) -> pd.DataFrame:
    out = df.copy()
    # 🔴 Ensure all expected features exist
    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = 0

    X = out[ALL_FEATURES]
    X_proc = preprocessor.transform(X)

    out["rule_score"] = 1 - np.exp(-out["rule_flag_count"].astype(float))
    out["rule_pred"] = (out["rule_flag_count"] > 0).astype(int)

    out["if_raw"] = -isolation_forest.decision_function(X_proc)
    out["if_score"] = minmax_scale(np.clip(out["if_raw"], 1e-6, None))
    out["if_pred"] = (out["if_score"] >= 0.5).astype(int)

    out["rf_probability"] = classifier.predict_proba(X_proc)[:, 1]
    out["rf_pred"] = (out["rf_probability"] >= 0.5).astype(int)

    out["ml_only_score"] = 0.3 * out["if_score"] + 0.7 * out["rf_probability"]
    out["ml_only_pred"] = (out["ml_only_score"] >= 0.5).astype(int)

    out["rules_if_score"] = 0.5 * out["rule_score"] + 0.5 * out["if_score"]
    out["rules_if_pred"] = (out["rules_if_score"] >= 0.5).astype(int)

    out["rules_rf_score"] = 0.5 * out["rule_score"] + 0.5 * out["rf_probability"]
    out["rules_rf_pred"] = (out["rules_rf_score"] >= 0.5).astype(int)

    weights = MODEL_CONFIG.get(
        "hybrid_weights",
        {"rf": 0.3, "anomaly": 0.3, "rules": 0.2, "interaction": 0.2},
    )
    threshold = MODEL_CONFIG.get("hybrid_threshold", 0.4)

    out["hybrid_score"] = (
        weights["rf"] * out["rf_probability"] +
        weights["anomaly"] * out["if_score"] +
        weights["rules"] * out["rule_score"] +
        weights["interaction"] * (out["rf_probability"] * out["rule_score"])
    )
    out["hybrid_pred"] = (out["hybrid_score"] >= threshold).astype(int)
    out["hybrid_risk_label"] = assign_risk_label(out["hybrid_score"])

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(0)

    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    out[obj_cols] = out[obj_cols].fillna("unknown")

    return out


def evaluate_all(test_scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = test_scored["is_problematic"].astype(int)

    summary = {
        "Rules only": metrics_dict(y_true, test_scored["rule_pred"], test_scored["rule_score"]),
        "Isolation Forest only": metrics_dict(y_true, test_scored["if_pred"], test_scored["if_score"]),
        "Random Forest only": metrics_dict(y_true, test_scored["rf_pred"], test_scored["rf_probability"]),
        "Rules + IF": metrics_dict(y_true, test_scored["rules_if_pred"], test_scored["rules_if_score"]),
        "Rules + RF": metrics_dict(y_true, test_scored["rules_rf_pred"], test_scored["rules_rf_score"]),
        "ML-only": metrics_dict(y_true, test_scored["ml_only_pred"], test_scored["ml_only_score"]),
        "Hybrid": metrics_dict(y_true, test_scored["hybrid_pred"], test_scored["hybrid_score"]),
    }
    metrics_df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "model"})

    ablation_df = metrics_df[
        metrics_df["model"].isin(
            ["Rules only", "Isolation Forest only", "Random Forest only", "Rules + IF", "Rules + RF", "Hybrid"]
        )
    ].copy()

    class_rows = []
    for anomaly_class in sorted(test_scored["anomaly_class"].dropna().unique()):
        if anomaly_class == "none":
            continue
        subset = test_scored[test_scored["anomaly_class"] == anomaly_class]
        y_subset = subset["is_problematic"].astype(int)

        class_rows.append({
            "anomaly_class": anomaly_class,
            "Rules only": recall_score(y_subset, subset["rule_pred"], zero_division=0),
            "ML-only": recall_score(y_subset, subset["ml_only_pred"], zero_division=0),
            "Hybrid": recall_score(y_subset, subset["hybrid_pred"], zero_division=0),
        })

    class_perf_df = pd.DataFrame(class_rows)
    return metrics_df, class_perf_df, ablation_df


def sensitivity_analysis(test_df: pd.DataFrame) -> pd.DataFrame:
    configs = [
        {"rf": 0.4, "anomaly": 0.3, "rules": 0.2, "interaction": 0.1},
        {"rf": 0.3, "anomaly": 0.3, "rules": 0.3, "interaction": 0.1},
        {"rf": 0.25, "anomaly": 0.25, "rules": 0.25, "interaction": 0.25},
    ]

    threshold = MODEL_CONFIG.get("hybrid_threshold", 0.4)
    results = []

    for w in configs:
        score = (
            w["rf"] * test_df["rf_probability"] +
            w["anomaly"] * test_df["if_score"] +
            w["rules"] * test_df["rule_score"] +
            w["interaction"] * (test_df["rf_probability"] * test_df["rule_score"])
        )

        pred = (score >= threshold).astype(int)

        results.append({
            "config": str(w),
            "precision": float(precision_score(test_df["is_problematic"], pred, zero_division=0)),
            "recall": float(recall_score(test_df["is_problematic"], pred, zero_division=0)),
            "f1_score": float(f1_score(test_df["is_problematic"], pred, zero_division=0)),
        })

    return pd.DataFrame(results)


def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_scored: pd.DataFrame,
    preprocessor,
    isolation_forest,
    classifier,
    results_dir: Path,
) -> None:
    ensure_dirs(results_dir, MODEL_DIR)

    sens_df = sensitivity_analysis(test_scored)
    sens_df.to_csv(results_dir / "sensitivity_analysis.csv", index=False)

    save_dataframe(train_df, results_dir / "train_split.csv")
    save_dataframe(val_df, results_dir / "validation_split.csv")
    save_dataframe(test_scored, results_dir / "test_scored_records.csv")

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(isolation_forest, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(classifier, MODEL_DIR / "classifier.joblib")

    metadata = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_scored)),
        "random_state": RANDOM_STATE,
        "hybrid_weights": DEFAULT_HYBRID_WEIGHTS,
        "hybrid_threshold": DEFAULT_HYBRID_THRESHOLD,
        "features": ALL_FEATURES,
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    feature_importance_df = get_feature_importance_df(preprocessor, classifier)
    feature_importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    high_risk = test_scored.sort_values("hybrid_score", ascending=False).head(20)
    high_risk.to_csv(results_dir / "top_20_high_risk_cases.csv", index=False)

    metrics_df, class_perf_df, ablation_df = evaluate_all(test_scored)
    metrics_df.to_csv(results_dir / "metrics_summary.csv", index=False)
    class_perf_df.to_csv(results_dir / "metrics_by_anomaly_class.csv", index=False)
    ablation_df.to_csv(results_dir / "ablation_study.csv", index=False)

    with open(results_dir / "classification_report_hybrid.json", "w", encoding="utf-8") as f:
        report = classification_report(
            test_scored["is_problematic"],
            test_scored["hybrid_pred"],
            output_dict=True,
            zero_division=0,
        )
        json.dump(report, f, indent=2)

    score_map = {
        "Rules only": test_scored["rule_score"],
        "ML-only": test_scored["ml_only_score"],
        "Hybrid": test_scored["hybrid_score"],
    }
    y_true = test_scored["is_problematic"]

    save_plot_roc(y_true, score_map, results_dir / "roc_curve.png")
    save_plot_pr(y_true, score_map, results_dir / "precision_recall_curve.png")
    save_plot_topk(y_true, score_map, results_dir / "topk_detection_curve.png")


def print_console_summary(test_scored: pd.DataFrame) -> None:
    metrics_df, class_perf_df, ablation_df = evaluate_all(test_scored)

    print("\n=== Detection Performance Summary ===")
    print(metrics_df.round(4).to_string(index=False))

    print("\n=== Ablation Study ===")
    print(ablation_df.round(4).to_string(index=False))

    print("\n=== Detection Recall by Anomaly Class ===")
    if class_perf_df.empty:
        print("No anomaly classes found in test set.")
    else:
        print(class_perf_df.round(4).to_string(index=False))

    fi = (
        test_scored.sort_values("hybrid_score", ascending=False)[
            [
                "record_id",
                "anomaly_class",
                "rule_flag_count",
                "rf_probability",
                "if_score",
                "hybrid_score",
                "hybrid_risk_label",
            ]
        ]
        .head(5)
    )
    print("\n=== Top 5 Highest-Risk Records ===")
    print(fi.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end experiments for Digital Trade Compliance AI.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--records", type=int, default=10000)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    results_dir = Path(args.results_dir)

    ensure_dirs(input_path.parent, results_dir, MODEL_DIR)

    raw_df = load_or_generate_dataset(
        input_path=input_path,
        n_records=args.records,
        force_generate=args.force_generate,
    )

    df = prepare_dataset(raw_df)
    train_df, val_df, test_df = split_dataset(df)

    preprocessor, isolation_forest, classifier = fit_models(train_df)
    test_scored = score_dataframe(test_df, preprocessor, isolation_forest, classifier)

    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_scored=test_scored,
        preprocessor=preprocessor,
        isolation_forest=isolation_forest,
        classifier=classifier,
        results_dir=results_dir,
    )

    print_console_summary(test_scored)

    print("\n[OK] Experiment run complete.")
    print(f"[OK] Models saved to: {MODEL_DIR}")
    print(f"[OK] Results saved to: {results_dir}")


if __name__ == "__main__":
    main()