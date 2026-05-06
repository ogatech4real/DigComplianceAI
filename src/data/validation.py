from __future__ import annotations
import pandas as pd
from src.domain.schema import REQUIRED_SCREENING_FIELDS


def validate_canonical_dataframe(df: pd.DataFrame) -> dict:
    issues = []

    # 🔹 Existing validation (UNCHANGED)
    for field in REQUIRED_SCREENING_FIELDS:
        if field not in df.columns:
            issues.append(f"missing column: {field}")
        elif df[field].isna().all():
            issues.append(f"column fully null: {field}")

    # 🔴 NEW: ICC STRUCTURE VALIDATION
    if "icc" not in df.columns:
        issues.append("missing ICC column")
    else:
        # Check first non-null ICC record for structure sanity
        sample = df["icc"].dropna()
        if not sample.empty:
            icc = sample.iloc[0]

            # Minimal structural checks
            if not isinstance(icc, dict):
                issues.append("ICC record not a dict")

            else:
                # Core fields required for compliance logic
                if not icc.get("product", {}).get("hs_code"):
                    issues.append("ICC missing hs_code")

                if icc.get("emissions", {}).get("value") is None:
                    issues.append("ICC missing emissions value")

                if not icc.get("origin", {}).get("country"):
                    issues.append("ICC missing origin country")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "row_count": len(df),
        "column_count": len(df.columns),
    }