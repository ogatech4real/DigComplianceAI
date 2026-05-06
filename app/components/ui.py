from __future__ import annotations
import streamlit as st


def render_header(title: str, caption: str, tag: str | None = None) -> None:
    st.title(title)
    if tag:
        st.markdown(f"**{tag}**")
    st.caption(caption)


def render_disclaimer() -> None:
    st.info(
        "This is a research-aligned decision-support prototype. It supports screening and triage. "
        "It is not a substitute for formal customs, legal, or regulatory review."
    )


def render_metric_row(metrics: list[tuple[str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_stakeholder_note() -> None:
    with st.expander("Who this prototype is for", expanded=False):
        st.markdown(
            "- **Exporters and importers:** pre-screen records before formal submission.\n"
            "- **Compliance teams and front-desk officers:** prioritise manual review workload.\n"
            "- **Analysts and policy stakeholders:** observe risk trends and anomaly patterns.\n"
            "- **Researchers:** evaluate the performance and interpretability of the hybrid framework."
        )


def render_mode_guide() -> str:
    return st.selectbox(
        "Operating mode",
        options=["Production (no labels)", "Demo (with labels)"],
        help="Use Production mode for real uploaded datasets without ground-truth labels. "
             "Use Demo mode for labelled benchmark or synthetic datasets."
    )