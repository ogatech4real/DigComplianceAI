from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
from src.evaluation.metrics import build_results_table
from app.components.ui import render_header
from app.state.session_manager import get_app_mode

render_header("Model Insights", "View evaluation outputs and model governance notes.")

report_path = ROOT / "data" / "processed" / "latest_screened_report.csv"
mode = get_app_mode()

if not report_path.exists():
    st.info("No screened report available yet.")
else:
    df = pd.read_csv(report_path)

    if "is_problematic" in df.columns and df["is_problematic"].notna().any() and mode == "Demo (with labels)":
        st.subheader("Evaluation snapshot")
        st.dataframe(build_results_table(df), use_container_width=True)
    else:
        st.warning("Ground-truth labels are not available or demo mode is not active, so evaluation metrics are hidden.")

    st.subheader("Governance note")
    st.markdown(
        "The ML layer augments but does not replace deterministic checks. Hybrid scoring should be treated as a triage aid. "
        "Formal review remains a human-controlled decision point."
    )

    explain_cols = [c for c in ["rf_probability", "if_score", "rule_score", "hybrid_score"] if c in df.columns]
    if explain_cols:
        st.subheader("Signal overview")
        st.dataframe(df[explain_cols].describe().T, use_container_width=True)