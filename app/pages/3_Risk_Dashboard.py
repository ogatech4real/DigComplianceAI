from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
from app.components.ui import render_header, render_disclaimer

render_header("Risk Dashboard", "Review portfolio-level screening outcomes and prioritisation signals.")
render_disclaimer()

report_path = ROOT / "data" / "processed" / "latest_screened_report.csv"
if not report_path.exists():
    st.info("No screened report found yet. Run Upload and Screen first.")
else:
    df = pd.read_csv(report_path)

    top = st.columns(5)
    top[0].metric("Records", f"{len(df):,}")
    top[1].metric("High risk", int((df["hybrid_risk"] == "high").sum()))
    top[2].metric("Medium risk", int((df["hybrid_risk"] == "medium").sum()))
    top[3].metric("Low risk", int((df["hybrid_risk"] == "low").sum()))
    readiness = max(0.0, 1.0 - ((df["hybrid_risk"] == "high").sum() / max(len(df), 1)))
    top[4].metric("Readiness score", f"{readiness:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk distribution")
        st.bar_chart(df["hybrid_risk"].value_counts().sort_index())

    with c2:
        st.subheader("Top rule flags")
        if "rule_flags" in df.columns:
            flags = df["rule_flags"].fillna("").astype(str).str.replace(r"[\[\]']", "", regex=True).str.split(",")
            exploded = flags.explode().str.strip()
            exploded = exploded[exploded != ""]
            if len(exploded):
                st.bar_chart(exploded.value_counts().head(10))
            else:
                st.write("No rule flags found in the latest report.")
        else:
            st.write("No rule flags available.")

    st.subheader("Highest-priority records")
    display_cols = [c for c in ["record_id", "hybrid_risk", "hybrid_score", "rf_probability", "if_score", "rule_flag_count", "recommended_action"] if c in df.columns]
    st.dataframe(df.sort_values("hybrid_score", ascending=False)[display_cols].head(50), use_container_width=True)