from __future__ import annotations

from typing import Optional

import pandas as pd


def filter_by_screen_and_validity(
    df: pd.DataFrame,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    require_validity: bool = False,
    x_col: str = "Gaze Point X[px]",
    y_col: str = "Gaze Point Y[px]",
    validity_accepted: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    out = df
    if screen_w is not None and screen_h is not None:
        if {x_col, y_col}.issubset(out.columns):
            x = pd.to_numeric(out[x_col], errors="coerce")
            y = pd.to_numeric(out[y_col], errors="coerce")
            out = out[x.between(0, screen_w) & y.between(0, screen_h)].copy()
    if require_validity and {"Validity Left", "Validity Right"}.issubset(out.columns):
        accepted = {str(v).strip().lower() for v in (validity_accepted or ("1",))}
        left = out["Validity Left"].astype(str).str.strip().str.lower()
        right = out["Validity Right"].astype(str).str.strip().str.lower()
        out = out[left.isin(accepted) & right.isin(accepted)].copy()
    return out
