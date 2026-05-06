from __future__ import annotations


# 🔹 EXISTING (UNCHANGED)
def explain_flags(flags: list[str]) -> list[str]:
    mapping = {
        "invalid_shipment_quantity": "Shipment quantity is missing or not positive.",
        "invalid_shipment_value": "Shipment value is missing or not positive.",
        "missing_declared_origin": "Declared origin country is missing.",
        "missing_emissions_value": "Embedded emissions value is missing.",
        "missing_traceability_score": "Traceability completeness score is missing.",
        "invalid_traceability_score": "Traceability score is outside the valid 0 to 1 range.",
        "invalid_recycled_content": "Recycled content percentage is outside the valid 0 to 100 range.",
        "recycled_method_conflict": "Record is tagged as recycled but recycled content is too low.",
        "origin_certificate_conflict": "Origin certificate availability conflicts with the reported match score.",
        "missing_supplier_traceability_metadata": "Supplier chain depth suggests metadata should exist, but it is missing.",
        "document_count_consistency_conflict": "No supporting documents are present despite a high consistency score.",
        "emissions_outside_plausibility_band": "Embedded emissions fall outside the expected product-family plausibility band.",
        "origin_transformation_mismatch": "Declared origin is inconsistent with the last substantial transformation country.",
    }
    return [mapping.get(flag, flag.replace("_", " ")) for flag in flags]


def recommended_action(risk: str) -> str:
    if risk == "high":
        return "Manual review required before formal submission or acceptance."
    if risk == "medium":
        return "Targeted document validation and data correction recommended."
    return "Proceed with standard workflow and periodic quality checks."


# 🔴 NEW: ICC CONTEXT-AWARE EXPLANATION ENRICHMENT
def enrich_explanations_with_context(flags: list[str], icc: dict | None) -> list[str]:
    """
    Adds lightweight ICC-aware context to explanations (non-breaking).
    """
    base_explanations = explain_flags(flags)

    if not isinstance(icc, dict):
        return base_explanations

    enriched = []

    for exp in base_explanations:
        # Add minimal contextual anchors (no heavy logic)
        if "emissions" in exp.lower():
            hs = icc.get("product", {}).get("hs_code")
            if hs:
                exp = f"{exp} (HS: {hs})"

        if "origin" in exp.lower():
            origin = icc.get("origin", {}).get("country")
            if origin:
                exp = f"{exp} (Origin: {origin})"

        enriched.append(exp)

    return enriched


# 🔴 NEW: HIGH-LEVEL EXPLANATION WRAPPER (PIPELINE READY)
def generate_explanations(row: dict) -> dict:
    """
    Unified explanation output using ICC context where available.
    """
    flags = row.get("rule_flags", [])
    risk = row.get("rule_based_risk", "low")
    icc = row.get("icc")

    return {
        "flags": flags,
        "explanations": enrich_explanations_with_context(flags, icc),
        "recommended_action": recommended_action(risk),
    }