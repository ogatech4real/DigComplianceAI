from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from src.data.io import load_tabular_file, save_dataframe
from src.domain.schema import REQUIRED_SCREENING_FIELDS
from services.screening_service import screen_dataframe
from services.reporting_service import exportable_report, build_summary
from src.evaluation.metrics import build_results_table
from src.features.preprocessing import ALL_FEATURES
from app.state.session_manager import (
    set_uploaded_df,
    get_uploaded_df,
    get_manual_mapping,
    set_screening_result,
    get_screening_result,
    get_app_mode,
)

st.title("Upload and Screen")
st.caption("Run the hybrid compliance screening pipeline on canonicalised data.")

file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "json", "parquet"], key="screen_upload")
if file:
    path = ROOT / "data" / "raw" / file.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file.getvalue())
    df = load_tabular_file(path)
    set_uploaded_df(df, file.name)

input_df = get_uploaded_df()
manual_mapping = get_manual_mapping()
mode = get_app_mode()

if input_df is None:
    st.info("Upload a file here or map one first on the Data Mapping page.")
else:
    mapped_count = sum(1 for _, v in manual_mapping.items() if v)
    required_missing = [field for field in REQUIRED_SCREENING_FIELDS if not manual_mapping.get(field)]

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Rows ready for screening", len(input_df))
    c2.metric("Mapped required fields", mapped_count)
    c3.metric("Mode", mode)

    if required_missing:
        st.error(f"Missing required mappings: {required_missing}")
        st.stop()

    if st.button("Screen records", type="primary"):
        result = screen_dataframe(input_df, manual_mapping=manual_mapping, mode=mode)
        set_screening_result(result)

        if result["status"] == "ok":
            processed_path = ROOT / "data" / "processed" / "latest_screened_report.csv"
            save_dataframe(exportable_report(result["data"]), processed_path)

    result = get_screening_result()
    if result:
        if result["status"] == "ok":
            screened = result["data"]
            summary = build_summary(screened)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rows", summary["rows"])
            c2.metric("High risk", summary["high_risk"])
            c3.metric("Medium risk", summary["medium_risk"])
            c4.metric("Low risk", summary["low_risk"])
            c5.metric("Readiness score", f"{summary['compliance_readiness_score']:.2f}")

            st.subheader("Mapping status")
            st.write({
                "confidence": round(result["mapping"].confidence, 2),
                "missing_required": result["mapping"].missing_required,
                "mode": result["mode"],
            })

            st.subheader("Screened report")
            report_df = exportable_report(screened)
            st.dataframe(report_df, use_container_width=True)
            st.download_button(
                "Download screened report",
                report_df.to_csv(index=False).encode("utf-8"),
                file_name="screened_report.csv",
                mime="text/csv",
            )

            with st.expander("Risk explanation preview", expanded=False):
                explain_cols = [c for c in ["record_id", "rule_flags", "rf_probability", "if_score", "hybrid_score", "record_explanation"] if c in screened.columns]
                st.dataframe(screened[explain_cols].head(20), use_container_width=True)

            with st.expander("Derived feature inputs", expanded=False):
                visible_features = [c for c in ALL_FEATURES if c in screened.columns]
                st.dataframe(screened[visible_features].head(20), use_container_width=True)

            if "is_problematic" in screened.columns and screened["is_problematic"].notna().any() and mode == "Demo (with labels)":
                st.subheader("Evaluation snapshot (labelled data)")
                st.dataframe(build_results_table(screened), use_container_width=True)
            else:
                st.info("No usable ground-truth labels detected. Showing operational risk scoring only.")
        elif result["status"] == "mapping_incomplete":
            st.error("Screening blocked because required mapping fields are incomplete.")
            st.json(result["validation"])
        else:
            st.error("Validation failed. Resolve field mapping or required data gaps first.")
            st.json(result["validation"])