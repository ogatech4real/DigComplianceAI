from __future__ import annotations
import pandas as pd
from typing import List, Dict, Optional

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds evaluation table including PR-AUC and supports extended configurations.
    Assumes df contains:
        - is_problematic (ground truth)
        - prediction columns
        - probability/score columns (where applicable)
    """

    rows = []

    configs = [
        ("Rules only", "rule_pred", None),
        ("Isolation Forest only", "if_pred", "if_score"),
        ("Random Forest only", "rf_pred", "rf_probability"),
        ("Rules + IF", "rules_if_pred", "rules_if_score"),
        ("Rules + RF", "rules_rf_pred", "rules_rf_score"),
        ("Full Hybrid", "hybrid_pred", "hybrid_score"),
    ]

    y_true = df["is_problematic"]

    for name, pred_col, prob_col in configs:

        if pred_col not in df.columns:
            continue  # skip missing configs gracefully

        y_pred = df[pred_col]

        row = {
            "Model": name,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": None,
            "PR-AUC": None,
        }

        if prob_col and prob_col in df.columns:
            y_score = df[prob_col]

            try:
                row["ROC-AUC"] = roc_auc_score(y_true, y_score)
            except ValueError:
                row["ROC-AUC"] = None

            try:
                row["PR-AUC"] = average_precision_score(y_true, y_score)
            except ValueError:
                row["PR-AUC"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def top_k_detection(df: pd.DataFrame, score_col: str, k: float = 0.2) -> float:
    """
    Computes top-k detection rate.
    """
    ranked = df.sort_values(score_col, ascending=False)
    top_n = max(1, int(len(ranked) * k))

    subset = ranked.head(top_n)
    total = df["is_problematic"].sum()

    if total == 0:
        return 0.0

    return float(subset["is_problematic"].sum() / total)


def multi_top_k_detection(
    df: pd.DataFrame,
    score_col: str,
    ks: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Computes multiple top-k detection rates (e.g., Top-10%, 20%, 30%).
    """
    if ks is None:
        ks = [0.1, 0.2, 0.3]

    results = {}

    for k in ks:
        key = f"Top-{int(k * 100)}%"
        results[key] = top_k_detection(df, score_col, k)

    return results