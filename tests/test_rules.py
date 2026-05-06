import pandas as pd
from src.rules.rule_engine import run_rule_engine


def test_rule_engine_flags_invalid_quantity():
    df = pd.DataFrame([{
        'product_family': 'aluminium_sheet',
        'shipment_quantity': -1,
        'shipment_value_usd': 100,
        'declared_origin_country': 'DE',
        'country_of_last_substantial_transformation': 'DE',
        'embedded_emissions_tco2e_per_tonne': 5.0,
        'traceability_completeness_score': 0.8,
        'production_method_tag': 'mixed',
        'recycled_content_percent': 40,
        'origin_certificate_available': 1,
        'origin_certificate_match_score': 0.9,
        'supplier_chain_depth': 1,
        'supplier_traceability_metadata_available': 1,
        'supporting_document_count': 1,
        'document_consistency_score': 0.8,
    }])
    out = run_rule_engine(df)
    assert out.loc[0, 'rule_flag_count'] >= 1
