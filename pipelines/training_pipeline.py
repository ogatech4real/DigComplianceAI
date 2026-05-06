from __future__ import annotations
from pathlib import Path

from src.data.io import load_tabular_file
from src.data.mapping import canonicalise_dataframe      # 🔴 NEW
from src.data.icc_transformer import ICCTranslator        # 🔴 NEW
from src.features.preprocessing import enrich_from_icc    # 🔴 NEW
from src.rules.rule_engine import run_rule_engine
from src.ml.training import train_models, save_models


def run_training(input_path: str):
    # 🔹 Step 1: Load raw data
    df = load_tabular_file(input_path)

    # 🔹 Step 2: Canonicalise (align with inference)
    df_canonical, _ = canonicalise_dataframe(df)

    # 🔴 Step 3: Ensure ICC present (safety)
    translator = ICCTranslator()
    if "icc" not in df_canonical.columns:
        df_canonical["icc"] = df_canonical.apply(
            lambda row: translator.to_icc(row.to_dict()),
            axis=1
        )

    # 🔴 Step 4: Enrich ML features from ICC
    df_enriched = enrich_from_icc(df_canonical)

    # 🔹 Step 5: Rule engine
    df_rules = run_rule_engine(df_enriched)

    # 🔹 Step 6: Train models
    bundle = train_models(df_rules)

    # 🔹 Step 7: Save models
    save_models(bundle)

    return bundle