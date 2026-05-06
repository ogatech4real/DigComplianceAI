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


# 🔴 NEW: ICC flatten helper
def _extract_icc_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "icc" not in df.columns:
        return df

    df = df.copy()

    def safe_get(d, path, default=None):
        for key in path:
            if not isinstance(d, dict):
                return default
            d = d.get(key)
        return d if d is not None else default

    df["icc_hs_code"] = df["icc"].apply(
        lambda x: safe_get(x, ["product", "hs_code"])
    )

    df["icc_trade_value"] = df["icc"].apply(
        lambda x: safe_get(x, ["trade", "value"])
    )

    df["icc_currency"] = df["icc"].apply(
        lambda x: safe_get(x, ["trade", "currency"])
    )

    df["icc_origin_country"] = df["icc"].apply(
        lambda x: safe_get(x, ["origin", "country"])
    )

    df["icc_context"] = df["icc"].apply(
        lambda x: safe_get(x, ["context"])
    )

    return df


def exportable_report(df: pd.DataFrame) -> pd.DataFrame:
    # 🔴 NEW: Flatten ICC fields before export
    df = _extract_icc_fields(df)

    cols = [
        "record_id",

        # Existing business fields
        "product_family",
        "declared_origin_country",
        "country_of_last_substantial_transformation",
        "embedded_emissions_tco2e_per_tonne",
        "traceability_completeness_score",

        # 🔴 NEW ICC export fields
        "icc_hs_code",
        "icc_trade_value",
        "icc_currency",
        "icc_origin_country",
        "icc_context",

        # Risk / compliance
        "rule_flags",
        "rule_flag_count",
        "if_score",
        "rf_probability",
        "ml_only_score",
        "hybrid_score",
        "hybrid_risk",

        # Explainability
        "record_explanation",
        "recommended_action",

        # Mapping metadata
        "mapping_note",
        "mapping_confidence",

        # Labels
        "is_problematic",
        "anomaly_class",
    ]

    cols = [c for c in cols if c in df.columns]

    return df[cols].copy()