from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from app.components.ui import (
    render_header,
    render_disclaimer,
    render_metric_row,
    render_stakeholder_note,
    render_mode_guide,
)
from app.state.session_manager import set_app_mode

st.set_page_config(page_title="Digital Trade Compliance AI", layout="wide", page_icon="📦")

render_header(
    "Digital Trade Compliance AI",
    "Research-aligned prototype for pre-screening carbon, origin, and traceability records.",
    tag="Hybrid rules + ML pre-screening"
)
render_disclaimer()
render_stakeholder_note()

mode = render_mode_guide()
set_app_mode(mode)

left, right = st.columns([1.35, 1])

with left:
    st.subheader("What the system does")
    st.markdown(
        "This prototype canonicalises uploaded trade records, validates the mapped schema, applies deterministic compliance checks, "
        "scores anomalies with machine learning, and returns an interpretable triage outcome for human review."
    )
    st.markdown(
        "Use the left navigation to move through the workflow: **Data Mapping → Upload and Screen → Batch Processing → Dashboard → Record Explorer → Analytics → Model Insights**."
    )

with right:
    render_metric_row([
        ("Input formats", "CSV / XLSX / JSON / Parquet"),
        ("Screening mode", "Single or batch"),
        ("Current mode", mode),
    ])

st.subheader("Recommended operating workflow")
st.markdown(
    "1. Upload the source file on **Data Mapping**.\n"
    "2. Confirm or override the required field mapping.\n"
    "3. Run screening on **Upload and Screen**.\n"
    "4. Review dashboard outputs, explanations, and flagged records.\n"
    "5. Export the report for audit or downstream review."
)

with st.expander("Repository posture", expanded=False):
    st.markdown(
        "The research contribution remains the hybrid pre-screening framework and its evaluation. "
        "The application layer demonstrates operational feasibility and supports live showcase delivery."
    )