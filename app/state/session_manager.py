from __future__ import annotations
import pandas as pd
import streamlit as st


def set_uploaded_df(df: pd.DataFrame, file_name: str) -> None:
    st.session_state["uploaded_df"] = df
    st.session_state["uploaded_file_name"] = file_name


def get_uploaded_df() -> pd.DataFrame | None:
    return st.session_state.get("uploaded_df")


def set_screening_result(result: dict) -> None:
    st.session_state["screening_result"] = result


def get_screening_result() -> dict | None:
    return st.session_state.get("screening_result")


def set_manual_mapping(mapping: dict) -> None:
    st.session_state["manual_mapping"] = mapping


def get_manual_mapping() -> dict:
    return st.session_state.get("manual_mapping", {})


def set_app_mode(mode: str) -> None:
    st.session_state["app_mode"] = mode


def get_app_mode() -> str:
    return st.session_state.get("app_mode", "Production (no labels)")