from __future__ import annotations
import pandas as pd
from src.utils.config import load_all_configs

PRODUCTS = load_all_configs()["product"]["products"]


def get_emissions_bounds(product_family: str):
    if product_family not in PRODUCTS:
        return None, None
    return PRODUCTS[product_family]["emissions_range"]


# 🔴 NEW: Safe ICC accessor
def _get_icc(row: pd.Series) -> dict:
    icc = row.get("icc")
    return icc if isinstance(icc, dict) else {}


def evaluate_rules(row: pd.Series) -> list:
    flags = []

    icc = _get_icc(row)

    # 🔹 Prefer ICC values, fallback to legacy (non-breaking)
    quantity = icc.get("trade", {}).get("quantity", row.get("shipment_quantity"))
    value = icc.get("trade", {}).get("value", row.get("shipment_value_usd"))
    origin = icc.get("origin", {}).get("country", row.get("declared_origin_country"))
    emissions = icc.get("emissions", {}).get("value", row.get("embedded_emissions_tco2e_per_tonne"))
    traceability = icc.get("traceability", {}).get("traceability_score", row.get("traceability_completeness_score"))
    recycled = icc.get("emissions", {}).get("recycled_content_percent", row.get("recycled_content_percent"))

    # 🔹 Core checks (UNCHANGED LOGIC, ICC-aware)
    if pd.isna(quantity) or quantity <= 0:
        flags.append("invalid_shipment_quantity")

    if pd.isna(value) or value <= 0:
        flags.append("invalid_shipment_value")

    if pd.isna(origin):
        flags.append("missing_declared_origin")

    if pd.isna(emissions):
        flags.append("missing_emissions_value")

    if pd.isna(traceability):
        flags.append("missing_traceability_score")
    elif traceability < 0 or traceability > 1:
        flags.append("invalid_traceability_score")

    if not pd.isna(recycled) and (recycled < 0 or recycled > 100):
        flags.append("invalid_recycled_content")

    if row.get("production_method_tag") == "recycled" and (pd.isna(recycled) or recycled < 50):
        flags.append("recycled_method_conflict")

    if (
        row.get("origin_certificate_available") == 0
        and not pd.isna(row.get("origin_certificate_match_score"))
        and row.get("origin_certificate_match_score") > 0.2
    ):
        flags.append("origin_certificate_conflict")

    if (
        not pd.isna(row.get("supplier_chain_depth"))
        and row.get("supplier_chain_depth") >= 3
        and row.get("supplier_traceability_metadata_available") == 0
    ):
        flags.append("missing_supplier_traceability_metadata")

    if (
        row.get("supporting_document_count") == 0
        and not pd.isna(row.get("document_consistency_score"))
        and row.get("document_consistency_score") > 0.2
    ):
        flags.append("document_count_consistency_conflict")

    # 🔴 Emissions plausibility (ICC-driven)
    product_family = icc.get("product", {}).get("category", row.get("product_family"))
    low, high = get_emissions_bounds(product_family)

    if not pd.isna(emissions) and low is not None and (emissions < low or emissions > high):
        flags.append("emissions_outside_plausibility_band")

    # 🔴 Origin consistency (ICC-driven)
    transformation = icc.get("origin", {}).get(
        "last_transformation_country",
        row.get("country_of_last_substantial_transformation")
    )

    if not pd.isna(origin) and not pd.isna(transformation) and origin != transformation:
        flags.append("origin_transformation_mismatch")

    return flags


def run_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rule_flags"] = out.apply(evaluate_rules, axis=1)
    out["rule_flag_count"] = out["rule_flags"].apply(len)

    # 🔹 Risk classification (UNCHANGED)
    out["rule_based_risk"] = out["rule_flag_count"].apply(
        lambda x: "low" if x == 0 else ("medium" if x <= 2 else "high")
    )

    return out