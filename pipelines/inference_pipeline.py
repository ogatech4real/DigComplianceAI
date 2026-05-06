from __future__ import annotations

from src.data.io import load_tabular_file
from src.data.mapping import canonicalise_dataframe   # 🔴 NEW
from src.data.icc_transformer import ICCTranslator    # 🔴 NEW
from services.screening_service import screen_dataframe


def run_inference(input_path: str):
    # 🔹 Step 1: Load raw data (UNCHANGED)
    df = load_tabular_file(input_path)

    # 🔹 Step 2: Canonicalise (existing mapping layer)
    df_canonical, _ = canonicalise_dataframe(df)

    # 🔴 Step 3: Ensure ICC present (safety, even though mapping already adds it)
    translator = ICCTranslator()
    if "icc" not in df_canonical.columns:
        df_canonical["icc"] = df_canonical.apply(
            lambda row: translator.to_icc(row.to_dict()),
            axis=1
        )

    # 🔹 Step 4: Pass to screening service
    return screen_dataframe(df_canonical)