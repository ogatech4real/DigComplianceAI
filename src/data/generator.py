from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.config import load_all_configs

CONFIG = load_all_configs()
PRODUCTS = CONFIG["product"]["products"]
ANOMALY_DISTRIBUTION = CONFIG["anomaly"]["distribution"]

COUNTRIES = ["DE", "FR", "NL", "SE", "NO", "TR", "AE", "IN", "CN", "ZA", "CA", "BR"]
PORTS = ["Rotterdam", "Hamburg", "Antwerp", "Felixstowe", "Shanghai", "Jebel_Ali", "Mumbai", "Durban", "Vancouver", "Santos"]
DESTINATION_MARKETS = ["EU", "UK", "US", "Other"]
INCOTERMS = ["FOB", "CIF", "EXW", "DAP"]
TRANSPORT_MODES = ["sea", "road", "rail"]
PRODUCTION_METHODS = ["primary", "recycled", "mixed"]
EMISSIONS_METHODS = ["measured", "default", "estimated"]
ELECTRICITY_TAGS = ["grid", "renewable", "mixed"]

np.random.seed(42)
random.seed(42)


def random_date(days=365):
    return datetime.now() - timedelta(days=random.randint(0, days))


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def choose_weighted(mapping: dict) -> str:
    keys = list(mapping.keys())
    probs = list(mapping.values())
    return np.random.choice(keys, p=probs)


# -----------------------------
# BASE RECORD
# -----------------------------
def generate_compliant_record() -> dict:
    product_family = random.choice(list(PRODUCTS.keys()))
    cfg = PRODUCTS[product_family]

    qty = float(np.round(np.random.uniform(10, 500), 2))
    price = float(np.random.uniform(*cfg["price_usd_per_tonne"]))
    emissions_low, emissions_high = cfg["emissions_range"]

    production_method = random.choices(PRODUCTION_METHODS, weights=[0.5, 0.25, 0.25], k=1)[0]

    if production_method == "recycled":
        recycled = float(np.round(np.random.uniform(60, 100), 2))
    elif production_method == "mixed":
        recycled = float(np.round(np.random.uniform(20, 70), 2))
    else:
        recycled = float(np.round(np.random.uniform(0, 20), 2))

    origin = random.choice(COUNTRIES)
    export = random.choice(COUNTRIES)
    depth = random.randint(1, 4)

    # emissions with noise (harder dataset)
    base = np.random.uniform(emissions_low, emissions_high)
    noise = np.random.normal(0, 0.5)
    emissions = float(np.round(base + noise, 2))

    return {
        "record_id": generate_id("REC"),
        "declaration_timestamp": random_date(),
        "shipment_id": generate_id("SHP"),
        "invoice_id": generate_id("INV"),
        "exporter_id": generate_id("EXP"),
        "importer_id": generate_id("IMP"),

        "destination_market": random.choice(DESTINATION_MARKETS),
        "port_of_export": random.choice(PORTS),
        "port_of_entry": random.choice(PORTS),

        "product_family": product_family,
        "hs_code": cfg["hs_code"],
        "product_description": cfg["description"],

        "shipment_quantity": qty,
        "quantity_unit": "tonnes",
        "shipment_value_usd": float(np.round(qty * price, 2)),

        "incoterm": random.choice(INCOTERMS),
        "transport_mode": random.choice(TRANSPORT_MODES),

        "declared_origin_country": origin,
        "country_of_export": export,
        "country_of_last_substantial_transformation": origin,

        "production_method_tag": production_method,
        "production_stage_country_1": random.choice(COUNTRIES),
        "production_stage_country_2": random.choice(COUNTRIES),

        "supplier_chain_depth": depth,

        "origin_certificate_available": random.choice([0, 1]),
        "origin_certificate_match_score": float(np.round(np.random.uniform(0.8, 1.0), 2)),

        "embedded_emissions_tco2e_per_tonne": emissions,
        "emissions_estimation_method": random.choice(EMISSIONS_METHODS),
        "electricity_source_tag": random.choice(ELECTRICITY_TAGS),

        "recycled_content_percent": recycled,
        "plant_emissions_disclosure_available": random.choice([0, 1]),

        "sector_emissions_band": random.choice(["low", "medium", "high"]),

        "traceability_completeness_score": float(np.round(np.random.beta(7, 2), 2)),
        "supplier_traceability_metadata_available": 1 if depth >= 2 else random.choice([0, 1]),

        "batch_lot_reference_available": 1,
        "production_site_identifier_available": 1,

        "supporting_document_count": random.randint(2, 6),
        "document_consistency_score": float(np.round(np.random.uniform(0.8, 1.0), 2)),

        "is_problematic": 0,
        "anomaly_class": "none",
        "anomaly_count": 0,
        "risk_label": "low",
        "rule_flag_count": 0,

        "synthetic_record_type": "compliant",
        "injected_pattern": "none",
    }


# -----------------------------
# ANOMALY INJECTIONS
# -----------------------------
def inject_missingness(record):
    fields = ["embedded_emissions_tco2e_per_tonne", "declared_origin_country", "traceability_completeness_score"]
    chosen = random.sample(fields, random.choice([1, 2]))
    for f in chosen:
        record[f] = None
    record.update({"is_problematic": 1, "anomaly_class": "missingness", "anomaly_count": len(chosen)})
    return record


def inject_logical_inconsistency(record):
    record["country_of_last_substantial_transformation"] = random.choice(
        [c for c in COUNTRIES if c != record["declared_origin_country"]]
    )
    record.update({"is_problematic": 1, "anomaly_class": "logical_inconsistency", "anomaly_count": 1})
    return record


def inject_plausibility(record):
    emissions_low, emissions_high = PRODUCTS[record["product_family"]]["emissions_range"]
    record["embedded_emissions_tco2e_per_tonne"] = float(
        np.round(emissions_high * np.random.uniform(1.05, 1.2), 2)
    )
    record.update({"is_problematic": 1, "anomaly_class": "plausibility", "anomaly_count": 1})
    return record


def inject_documentation_quality(record):
    record["document_consistency_score"] = 0.2
    record.update({"is_problematic": 1, "anomaly_class": "documentation_quality", "anomaly_count": 1})
    return record


def inject_composite(record):
    for fn in random.sample(
        [inject_missingness, inject_logical_inconsistency, inject_plausibility], 2
    ):
        record = fn(record)
    record.update({"anomaly_class": "composite", "anomaly_count": 2})
    return record


def apply_anomaly(record, anomaly_class):
    return {
        "none": lambda r: r,
        "missingness": inject_missingness,
        "logical_inconsistency": inject_logical_inconsistency,
        "plausibility": inject_plausibility,
        "documentation_quality": inject_documentation_quality,
        "composite": inject_composite,
    }[anomaly_class](record)


# -----------------------------
# DATASET GENERATION
# -----------------------------
def generate_dataset(n_records: int = 10000) -> pd.DataFrame:
    records = []
    for _ in range(n_records):
        rec = generate_compliant_record()
        anomaly = choose_weighted(ANOMALY_DISTRIBUTION)
        rec = apply_anomaly(rec, anomaly)
        records.append(rec)
    return pd.DataFrame(records)