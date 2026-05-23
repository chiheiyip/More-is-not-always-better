from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.utils.io import read_table, write_table


def run_diagnostics(
    master_csv: str | Path,
    participants_csv: str | Path,
    outdir: str | Path = "outputs/06_robustness",
) -> dict[str, Path]:
    master = read_table(master_csv)
    participants = read_table(participants_csv)
    order = order_fatigue_effects(master)
    gender = factor_sensitivity(master, "Gender")
    batch = factor_sensitivity(master, "RecruitmentBatch")
    nonlinear = nonlinear_wwr_sensitivity(master)
    power = power_sensitivity(participants)
    outdir = Path(outdir)
    return {
        "order_fatigue_effects": write_table(order, outdir / "order_fatigue_effects.csv"),
        "gender_sensitivity": write_table(gender, outdir / "gender_sensitivity.csv"),
        "batch_sensitivity": write_table(batch, outdir / "batch_sensitivity.csv"),
        "nonlinear_wwr_sensitivity": write_table(nonlinear, outdir / "nonlinear_wwr_sensitivity.csv"),
        "power_sensitivity": write_table(power, outdir / "power_sensitivity.csv"),
    }


def order_fatigue_effects(master: pd.DataFrame) -> pd.DataFrame:
    outcomes = _candidate_outcomes(master)
    rows = []
    for outcome in outcomes:
        values = pd.to_numeric(master[outcome], errors="coerce")
        for covar in ["position", "block", "round"]:
            if covar not in master.columns:
                continue
            x = pd.to_numeric(master[covar], errors="coerce")
            mask = values.notna() & x.notna()
            corr = (
                float(np.corrcoef(x[mask], values[mask])[0, 1])
                if mask.sum() > 2 and x[mask].nunique() > 1 and values[mask].nunique() > 1
                else np.nan
            )
            rows.append({"outcome": outcome, "order_variable": covar, "n": int(mask.sum()), "correlation": corr, "interpretation": "include_as_covariate"})
    return pd.DataFrame(rows)


def nonlinear_wwr_sensitivity(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in _candidate_outcomes(master):
        if "WWR" not in master.columns:
            continue
        values = pd.to_numeric(master[outcome], errors="coerce")
        means = master.assign(_y=values).groupby("WWR")["_y"].mean()
        row = {"outcome": outcome, "available_wwr_levels": ",".join(map(str, means.dropna().index.tolist()))}
        numeric = {float(k): v for k, v in means.items() if pd.notna(v)}
        if {15.0, 45.0, 75.0}.issubset(numeric):
            row["wwr45_peak_index"] = float(numeric[45.0] - (numeric[15.0] + numeric[75.0]) / 2)
            row["claim_strength"] = "exploratory_only_three_levels"
        else:
            row["claim_strength"] = "insufficient_wwr_granularity"
        rows.append(row)
    return pd.DataFrame(rows)


def factor_sensitivity(master: pd.DataFrame, factor: str) -> pd.DataFrame:
    if factor not in master.columns:
        return pd.DataFrame([{"factor": factor, "status": "missing"}])
    rows = []
    for outcome in _candidate_outcomes(master):
        values = pd.to_numeric(master[outcome], errors="coerce")
        for level, sub in master.assign(_y=values).groupby(factor, dropna=False):
            rows.append({"factor": factor, "level": level, "outcome": outcome, "n": int(sub["_y"].notna().sum()), "mean": float(sub["_y"].mean()) if sub["_y"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def power_sensitivity(participants: pd.DataFrame) -> pd.DataFrame:
    group_col = "ExperienceGroup" if "ExperienceGroup" in participants.columns else "Experience"
    counts = participants[group_col].fillna("Unknown").astype(str).value_counts()
    rows = [{"grouping": group_col, "level": level, "n": int(n)} for level, n in counts.items()]
    if len(counts) >= 2:
        rows.append({"grouping": group_col, "level": "min_group_n", "n": int(counts.min()), "interpretation": "interaction_power_limited_if_small"})
    return pd.DataFrame(rows)


def _candidate_outcomes(master: pd.DataFrame) -> list[str]:
    tokens = ["q_S", "theta", "alpha", "beta", "FCR", "TFD", "TTFF", "attention_share"]
    return [c for c in master.columns if any(token.lower() in c.lower() for token in tokens)]
