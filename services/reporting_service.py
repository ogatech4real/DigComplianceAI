from __future__ import annotations

import pandas as pd


def build_summary(df: pd.DataFrame) -> dict:
    rows = len(df)
    high_risk = int((df["hybrid_risk"] == "high").sum())
    medium_risk = int((df["hybrid_risk"] == "medium").sum())
    low_risk = int((df["hybrid_risk"] == "low").sum())

    compliance_readiness = 0.0
    if rows > 0:
        compliance_readiness = max(0.0, 1.0 - (high_risk / rows))

    return {
        "rows": rows,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "avg_hybrid_score": float(df["hybrid_score"].mean()) if "hybrid_score" in df.columns else 0.0,
        "avg_rule_flags": float(df["rule_flag_count"].mean()) if "rule_flag_count" in df.columns else 0.0,
        "compliance_readiness_score": float(compliance_readiness),
    }


def exportable_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "record_id",
        "product_family",
        "declared_origin_country",
        "country_of_last_substantial_transformation",
        "embedded_emissions_tco2e_per_tonne",
        "traceability_completeness_score",
        "rule_flags",
        "rule_flag_count",
        "if_score",
        "rf_probability",
        "ml_only_score",
        "hybrid_score",
        "hybrid_risk",
        "record_explanation",
        "recommended_action",
        "mapping_note",
        "mapping_confidence",
        "is_problematic",
        "anomaly_class",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()