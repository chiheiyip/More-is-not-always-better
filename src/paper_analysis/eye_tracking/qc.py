from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EyeQCPolicy:
    screen_w: int | None = None
    screen_h: int | None = None
    validity_accepted: tuple[str, ...] | None = None


def valid_eye_mask(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    policy: EyeQCPolicy | None = None,
) -> tuple[pd.Series, dict]:
    policy = policy or EyeQCPolicy()
    x = pd.to_numeric(df[x_col], errors="coerce") if x_col in df.columns else pd.Series(np.nan, index=df.index)
    y = pd.to_numeric(df[y_col], errors="coerce") if y_col in df.columns else pd.Series(np.nan, index=df.index)
    finite_xy = x.notna() & y.notna()
    in_screen = finite_xy.copy()
    screen_applied = policy.screen_w is not None and policy.screen_h is not None
    if screen_applied:
        in_screen = finite_xy & x.between(0, policy.screen_w) & y.between(0, policy.screen_h)

    validity_mask = pd.Series(True, index=df.index)
    validity_fields_present = {"Validity Left", "Validity Right"}.issubset(df.columns)
    validity_applied = policy.validity_accepted is not None and validity_fields_present
    if validity_applied:
        accepted = {str(v).strip().lower() for v in policy.validity_accepted}
        left = df["Validity Left"].astype(str).str.strip().str.lower()
        right = df["Validity Right"].astype(str).str.strip().str.lower()
        validity_mask = left.isin(accepted) & right.isin(accepted)

    mask = in_screen & validity_mask
    total = int(len(df))
    stats = {
        "eye_sample_count": total,
        "finite_xy_count": int(finite_xy.sum()),
        "screen_valid_count": int(in_screen.sum()) if screen_applied else np.nan,
        "validity_valid_count": int(validity_mask.sum()) if validity_applied else np.nan,
        "analysis_valid_count": int(mask.sum()),
        "finite_xy_ratio": float(finite_xy.mean()) if total else np.nan,
        "screen_valid_ratio": float(in_screen.mean()) if total and screen_applied else np.nan,
        "validity_valid_ratio": float(validity_mask.mean()) if total and validity_applied else np.nan,
        "analysis_valid_ratio": float(mask.mean()) if total else np.nan,
        "screen_check_applied": screen_applied,
        "validity_check_applied": validity_applied,
        "screen_check_status": "applied" if screen_applied else "not_configured",
        "validity_check_status": "applied" if validity_applied else "configured_but_fields_missing" if policy.validity_accepted is not None else "not_configured",
    }
    return mask, stats
