from __future__ import annotations

import pandas as pd

from src.data.mapping import canonicalise_dataframe
from src.data.validation import validate_canonical_dataframe
from src.domain.schema import REQUIRED_SCREENING_FIELDS
from src.rules.rule_engine import run_rule_engine
from src.ml.inference import load_models, apply_models
from src.explainability.explanations import (
    generate_explanations,   # 🔴 NEW
    recommended_action
)
from src.features.preprocessing import enrich_from_icc  # 🔴 NEW


def _validate_mapping_contract(manual_mapping: dict | None = None) -> dict:
    manual_mapping = manual_mapping or {}
    unresolved = [field for field in REQUIRED_SCREENING_FIELDS if not manual_mapping.get(field)]
    return {
        "mapping_complete": len(unresolved) == 0,
        "missing_required_mappings": unresolved,
    }


def _derive_record_explanation(row: pd.Series) -> str:
    parts = []

    if int(row.get("rule_flag_count", 0)) > 0:
        flags = row.get("rule_flags", [])
        if isinstance(flags, list) and flags:
            parts.append(f"Rules: {', '.join(map(str, flags[:3]))}")
        else:
            parts.append(f"Rules triggered: {int(row.get('rule_flag_count', 0))}")

    rf_probability = float(row.get("rf_probability", 0.0))
    if_score = float(row.get("if_score", 0.0))
    hybrid_score = float(row.get("hybrid_score", 0.0))

    parts.append(f"RF probability={rf_probability:.2f}")
    parts.append(f"IF score={if_score:.2f}")
    parts.append(f"Hybrid score={hybrid_score:.2f}")

    return " | ".join(parts)


def screen_dataframe(
    input_df: pd.DataFrame,
    manual_mapping: dict | None = None,
    mode: str = "Production (no labels)",
):
    # 🔹 Step 1: Mapping contract validation
    mapping_check = _validate_mapping_contract(manual_mapping)

    # 🔹 Step 2: Canonicalise (includes ICC)
    canonical_df, mapping_result = canonicalise_dataframe(
        input_df, manual_mapping=manual_mapping
    )

    # 🔴 Step 3: Enrich features from ICC (NEW)
    enriched_df = enrich_from_icc(canonical_df)

    # 🔹 Step 4: Validate
    validation = validate_canonical_dataframe(enriched_df)

    validation_payload = {
        **validation,
        "mapping_complete": mapping_check["mapping_complete"],
        "missing_required_mappings": mapping_check["missing_required_mappings"],
    }

    if not mapping_check["mapping_complete"]:
        return {
            "status": "mapping_incomplete",
            "validation": validation_payload,
            "mapping": mapping_result,
            "data": enriched_df,
            "mode": mode,
        }

    if not validation["valid"]:
        return {
            "status": "validation_failed",
            "validation": validation_payload,
            "mapping": mapping_result,
            "data": enriched_df,
            "mode": mode,
        }

    # 🔹 Step 5: Rule engine (ICC-aware)
    ruled = run_rule_engine(enriched_df)
    ruled["rule_pred"] = (ruled["rule_flag_count"] > 0).astype(int)

    # 🔹 Step 6: ML scoring
    models = load_models()
    scored = apply_models(ruled, models)

    # 🔴 Step 7: ICC-aware explanations (NEW)
    explanations_bundle = scored.apply(
        lambda row: generate_explanations(row.to_dict()),
        axis=1
    )

    scored["explanations"] = explanations_bundle.apply(lambda x: x["explanations"])
    scored["recommended_action"] = explanations_bundle.apply(lambda x: x["recommended_action"])

    # 🔹 Step 8: Record-level explanation
    scored["record_explanation"] = scored.apply(_derive_record_explanation, axis=1)

    return {
        "status": "ok",
        "validation": validation_payload,
        "mapping": mapping_result,
        "data": scored,
        "mode": mode,
    }