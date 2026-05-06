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
from app.components.ui import render_header, render_disclaimer
from app.state.session_manager import get_app_mode

render_header("Batch Processing", "Screen larger datasets and export an operational report.")
render_disclaimer()

file = st.file_uploader("Upload batch dataset", type=["csv", "xlsx", "json", "parquet"], key="batch_upload")
manual_mapping = st.session_state.get("manual_mapping", {})
mode = get_app_mode()

if not file:
    st.info("Upload a mapped dataset to run batch screening.")
else:
    path = ROOT / "data" / "raw" / file.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file.getvalue())
    df = load_tabular_file(path)

    missing_mappings = [field for field in REQUIRED_SCREENING_FIELDS if not manual_mapping.get(field)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Mode", mode)

    if missing_mappings:
        st.error(f"Missing required mappings: {missing_mappings}")
    elif st.button("Run batch screening", type="primary"):
        result = screen_dataframe(df, manual_mapping=manual_mapping, mode=mode)
        if result["status"] != "ok":
            st.error("Batch validation failed.")
            st.json(result["validation"])
        else:
            screened = result["data"]
            summary = build_summary(screened)
            st.success("Batch screening completed.")
            st.write(summary)

            report = exportable_report(screened)
            save_path = ROOT / "data" / "processed" / f"batch_{file.name.rsplit('.', 1)[0]}_screened.csv"
            save_dataframe(report, save_path)

            st.dataframe(report.head(200), use_container_width=True)
            st.download_button(
                "Download batch report",
                report.to_csv(index=False).encode("utf-8"),
                file_name="batch_screened_report.csv",
                mime="text/csv",
            )
            st.caption(f"Saved local copy to {save_path.relative_to(ROOT)}")