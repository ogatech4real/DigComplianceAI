from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Tuple
import pandas as pd

from src.domain.schema import CANONICAL_FIELDS, REQUIRED_SCREENING_FIELDS
from src.data.icc_transformer import ICCTranslator  # 🔴 NEW

# 🔹 EXISTING (UNCHANGED)
ALIASES = {
    "record_id": ["record_id", "id", "reference_id"],
    "product_family": ["product_family", "product", "product_type", "commodity_type"],
    "shipment_quantity": ["shipment_quantity", "quantity", "qty", "weight_tonnes"],
    "shipment_value_usd": ["shipment_value_usd", "value_usd", "shipment_value", "value"],
    "declared_origin_country": ["declared_origin_country", "origin_country", "country_of_origin", "origin"],
    "country_of_last_substantial_transformation": ["country_of_last_substantial_transformation", "last_transformation_country", "transformation_country"],
    "embedded_emissions_tco2e_per_tonne": ["embedded_emissions_tco2e_per_tonne", "emissions", "emissions_intensity", "co2e_per_tonne"],
    "traceability_completeness_score": ["traceability_completeness_score", "traceability_score", "traceability", "completeness_score"],
    "country_of_export": ["country_of_export", "export_country", "exporter_country"],
    "supplier_chain_depth": ["supplier_chain_depth", "tier_depth", "supply_chain_depth"],
    "origin_certificate_available": ["origin_certificate_available", "origin_certificate", "certificate_available"],
    "origin_certificate_match_score": ["origin_certificate_match_score", "certificate_match_score", "origin_match_score"],
    "recycled_content_percent": ["recycled_content_percent", "recycled_content", "recycled_pct"],
    "supporting_document_count": ["supporting_document_count", "document_count", "docs_count"],
    "document_consistency_score": ["document_consistency_score", "doc_consistency", "consistency_score"],
}

DEFAULTS = {
    "quantity_unit": "tonnes",
    "destination_market": "EU",
    "incoterm": "FOB",
    "transport_mode": "sea",
    "production_method_tag": "mixed",
    "emissions_estimation_method": "estimated",
    "electricity_source_tag": "mixed",
    "plant_emissions_disclosure_available": 0,
    "supplier_traceability_metadata_available": 0,
    "batch_lot_reference_available": 0,
    "production_site_identifier_available": 0,
    "sector_emissions_band": "medium",
    "anomaly_class": "none",
    "anomaly_count": 0,
    "risk_label": "low",
    "rule_flag_count": 0,
    "synthetic_record_type": "uploaded",
    "injected_pattern": "user_input",
}


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class MappingResult:
    mapping: Dict[str, str]
    missing_required: List[str]
    confidence: float
    suggested_mapping: Dict[str, str]


def infer_mapping(columns: List[str]) -> MappingResult:
    mapping = {}
    norm_cols = {_normalise(col): col for col in columns}
    hits = 0
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            candidate = norm_cols.get(_normalise(alias))
            if candidate:
                mapping[canonical] = candidate
                hits += 1
                break

    missing_required = [field for field in REQUIRED_SCREENING_FIELDS if field not in mapping]
    confidence = hits / max(len(REQUIRED_SCREENING_FIELDS), 1)

    return MappingResult(mapping=mapping, missing_required=missing_required, confidence=confidence, suggested_mapping=dict(mapping))


def build_mapping_table(columns: List[str], manual_mapping: Dict[str, str] | None = None) -> pd.DataFrame:
    inferred = infer_mapping(columns)
    rows = []
    manual_mapping = manual_mapping or {}

    for field in REQUIRED_SCREENING_FIELDS:
        rows.append({
            "canonical_field": field,
            "auto_detected_source": inferred.mapping.get(field, ""),
            "selected_source": manual_mapping.get(field, inferred.mapping.get(field, "")),
            "is_required": True,
        })

    return pd.DataFrame(rows)


# 🔴 CORE FUNCTION (UPGRADED)
def canonicalise_dataframe(
    df: pd.DataFrame,
    manual_mapping: Dict[str, str] | None = None
) -> Tuple[pd.DataFrame, MappingResult]:

    inferred = infer_mapping(list(df.columns))
    final_mapping = dict(inferred.mapping)

    if manual_mapping:
        final_mapping.update({k: v for k, v in manual_mapping.items() if v})

    output = pd.DataFrame()

    # 🔹 Step 1: Canonical mapping (UNCHANGED LOGIC)
    for field in CANONICAL_FIELDS:
        source = final_mapping.get(field)
        if source and source in df.columns:
            output[field] = df[source]
        elif field in DEFAULTS:
            output[field] = DEFAULTS[field]
        else:
            output[field] = None

    # 🔹 Step 2: Ensure record_id exists
    if output["record_id"].isna().all():
        output["record_id"] = [f"UPL_{i+1:06d}" for i in range(len(output))]

    # 🔹 Step 3: Add mapping metadata
    output["mapping_confidence"] = inferred.confidence
    output["mapping_note"] = "auto_inferred" if not manual_mapping else "manual_override"

    missing_required = [field for field in REQUIRED_SCREENING_FIELDS if output[field].isna().all()]

    result = MappingResult(
        final_mapping,
        missing_required,
        inferred.confidence,
        suggested_mapping=inferred.mapping
    )

    # 🔴 Step 4: NEW — ICC TRANSLATION (ROW-WISE)
    translator = ICCTranslator()

    output["icc"] = output.apply(
        lambda row: translator.to_icc(row.to_dict()),
        axis=1
    )

    return output, result