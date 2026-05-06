from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
from app.components.ui import render_header

render_header("Record Explorer", "Inspect individual records and their screening rationale.")

report_path = ROOT / "data" / "processed" / "latest_screened_report.csv"
if not report_path.exists():
    st.info("No screened report available. Run the screening flow first.")
else:
    df = pd.read_csv(report_path)

    risk_filter = st.multiselect(
        "Filter by risk",
        options=sorted(df["hybrid_risk"].dropna().unique()),
        default=sorted(df["hybrid_risk"].dropna().unique()),
    )
    filtered = df[df["hybrid_risk"].isin(risk_filter)] if risk_filter else df

    record_ids = filtered["record_id"].astype(str).tolist()
    selected = st.selectbox("Select record", options=record_ids)
    row = filtered[filtered["record_id"].astype(str) == selected].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", str(row.get("hybrid_risk", "n/a")).upper())
    c2.metric("Hybrid score", f"{float(row.get('hybrid_score', 0)):.2f}")
    c3.metric("Rule flags", int(row.get("rule_flag_count", 0)))

    st.subheader("Screening explanation")
    st.write({
        "record_explanation": row.get("record_explanation", "n/a"),
        "recommended_action": row.get("recommended_action", "n/a"),
        "rule_flags": row.get("rule_flags", "n/a"),
    })

    st.subheader("Record detail")
    st.json(row.to_dict())