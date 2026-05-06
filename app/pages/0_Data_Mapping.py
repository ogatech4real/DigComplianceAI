from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from src.data.io import load_tabular_file
from src.data.mapping import build_mapping_table, infer_mapping
from src.domain.schema import REQUIRED_SCREENING_FIELDS
from app.state.session_manager import (
    set_uploaded_df,
    get_uploaded_df,
    set_manual_mapping,
    get_manual_mapping,
)

st.title("Data Mapping and Validation")
st.caption("Canonicalise heterogeneous uploads before screening. This page is the correct entry point for non-standard datasets.")

file = st.file_uploader("Upload source dataset", type=["csv", "xlsx", "json", "parquet"], key="mapping_upload")
if file:
    path = ROOT / "data" / "raw" / file.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file.getvalue())
    df = load_tabular_file(path)
    set_uploaded_df(df, file.name)

uploaded_df = get_uploaded_df()
manual_mapping = get_manual_mapping()

if uploaded_df is None:
    st.info("Upload a file to start mapping. The prototype supports CSV, Excel, JSON, and Parquet.")
else:
    inferred = infer_mapping(list(uploaded_df.columns))

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(uploaded_df))
    c2.metric("Columns", len(uploaded_df.columns))
    c3.metric("Auto-mapping confidence", f"{inferred.confidence:.2f}")

    with st.expander("Source columns", expanded=False):
        st.write(list(uploaded_df.columns))
        st.dataframe(uploaded_df.head(20), use_container_width=True)

    st.subheader("Required field mapping")
    mapping_table = build_mapping_table(list(uploaded_df.columns), manual_mapping=manual_mapping)

    new_mapping = {}
    cols = [""] + list(uploaded_df.columns)
    for row in mapping_table.to_dict(orient="records"):
        key = f"map_{row['canonical_field']}"
        default_value = row["selected_source"] if row["selected_source"] in cols else ""
        index = cols.index(default_value) if default_value in cols else 0
        new_mapping[row["canonical_field"]] = st.selectbox(
            row["canonical_field"],
            options=cols,
            index=index,
            help=f"Auto-detected: {row['auto_detected_source'] or 'none'}",
            key=key,
        )

    set_manual_mapping(new_mapping)

    unresolved = [field for field in REQUIRED_SCREENING_FIELDS if not new_mapping.get(field)]
    if unresolved:
        st.error(f"Missing required mappings: {unresolved}")
    else:
        st.success("All required screening fields are mapped and ready for screening.")

    st.info(
        "The app supports heterogeneous source schemas, but meaningful results depend on mapping the uploaded dataset "
        "to the canonical screening contract."
    )