from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
from app.components.ui import render_header

render_header("Analytics", "Analyse systemic patterns for operational and policy-facing review.")

report_path = ROOT / "data" / "processed" / "latest_screened_report.csv"
if not report_path.exists():
    st.info("Run screening first to generate analytics inputs.")
else:
    df = pd.read_csv(report_path)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk by origin country")
        if "declared_origin_country" in df.columns:
            table = pd.crosstab(df["declared_origin_country"], df["hybrid_risk"])
            st.dataframe(table, use_container_width=True)
        else:
            st.write("Origin country not available.")

    with c2:
        st.subheader("Average score by product family")
        if "product_family" in df.columns:
            series = df.groupby("product_family")["hybrid_score"].mean().sort_values(ascending=False)
            st.bar_chart(series)
        else:
            st.write("Product family not available.")

    if "mapping_note" in df.columns:
        st.subheader("Mapping posture")
        st.dataframe(
            df["mapping_note"].value_counts().rename_axis("mapping_note").reset_index(name="count"),
            use_container_width=True
        )

    explain_cols = [c for c in ["hybrid_score", "rf_probability", "if_score", "rule_flag_count"] if c in df.columns]
    if explain_cols:
        st.subheader("Model signal summary")
        st.dataframe(df[explain_cols].describe().T, use_container_width=True)