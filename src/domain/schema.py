# Existing canonical fields (UNCHANGED)
CANONICAL_FIELDS = [
    "record_id",
    "declaration_timestamp",
    "shipment_id",
    "invoice_id",
    "exporter_id",
    "importer_id",
    "destination_market",
    "port_of_export",
    "port_of_entry",
    "product_family",
    "hs_code",
    "product_description",
    "shipment_quantity",
    "quantity_unit",
    "shipment_value_usd",
    "incoterm",
    "transport_mode",
    "declared_origin_country",
    "country_of_export",
    "country_of_last_substantial_transformation",
    "production_method_tag",
    "production_stage_country_1",
    "production_stage_country_2",
    "supplier_chain_depth",
    "origin_certificate_available",
    "origin_certificate_match_score",
    "embedded_emissions_tco2e_per_tonne",
    "emissions_estimation_method",
    "electricity_source_tag",
    "recycled_content_percent",
    "plant_emissions_disclosure_available",
    "sector_emissions_band",
    "traceability_completeness_score",
    "supplier_traceability_metadata_available",
    "batch_lot_reference_available",
    "production_site_identifier_available",
    "supporting_document_count",
    "document_consistency_score",
    "is_problematic",
    "anomaly_class",
    "anomaly_count",
    "risk_label",
    "rule_flag_count",
    "synthetic_record_type",
    "injected_pattern"
]

# Existing required fields (UNCHANGED)
REQUIRED_SCREENING_FIELDS = [
    "product_family",
    "shipment_quantity",
    "shipment_value_usd",
    "declared_origin_country",
    "country_of_last_substantial_transformation",
    "embedded_emissions_tco2e_per_tonne",
    "traceability_completeness_score",
]

# 🔴 NEW: ICC-ALIGNED CANONICAL STRUCTURE (NON-BREAKING ADDITION)

ICC_CANONICAL_TEMPLATE = {
    "product": {
        "hs_code": None,
        "description": None,
        "category": None
    },
    "origin": {
        "country": None,
        "export_country": None,
        "last_transformation_country": None
    },
    "trade": {
        "quantity": None,
        "unit": None,
        "value": None,
        "currency": "USD",
        "incoterm": None
    },
    "transport": {
        "mode": None,
        "port_of_export": None,
        "port_of_entry": None
    },
    "emissions": {
        "value": None,
        "unit": "tCO2e",
        "estimation_method": None,
        "electricity_source": None,
        "recycled_content_percent": None
    },
    "identifiers": {
        "shipment_id": None,
        "invoice_id": None,
        "exporter_id": None,
        "importer_id": None,
        "entity_id": None,     # LEI (future-ready)
        "location_id": None    # GLN (future-ready)
    },
    "traceability": {
        "supplier_chain_depth": None,
        "traceability_score": None,
        "batch_reference_available": None,
        "production_site_available": None
    },
    "compliance": {
        "origin_certificate_available": None,
        "origin_certificate_match_score": None,
        "document_consistency_score": None,
        "supporting_document_count": None
    },
    "risk": {
        "risk_label": None,
        "rule_flag_count": None,
        "anomaly_count": None
    },
    "context": None  # buy / ship / pay
}