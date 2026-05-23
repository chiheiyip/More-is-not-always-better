from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from paper_analysis.utils.io import read_table, write_table


DEFAULT_PREDICTORS = ["C(WWR)", "C(Complexity)", "C(ExperienceGroup)", "C(Gender)", "Age", "block", "position", "C(RecruitmentBatch)"]


def run_statistical_models(
    master_csv: str | Path,
    model_config: str | Path = "configs/model_families.json",
    outdir: str | Path = "outputs/06_models",
) -> dict[str, Path]:
    master = read_table(master_csv)
    config = load_model_config(model_config)
    model_rows = []
    diagnostics_rows = []
    contrast_rows = []
    for family in config.get("families", []):
        predictors = family.get("predictors", DEFAULT_PREDICTORS)
        for outcome in family.get("outcomes", []):
            if outcome not in master.columns:
                diagnostics_rows.append({"outcome": outcome, "status": "missing_outcome"})
                continue
            result, diag = fit_model(master, outcome, predictors, family.get("kind", "continuous"))
            model_rows.extend(result)
            diagnostics_rows.append(diag)
            contrast_rows.extend(wwr_contrasts(master, outcome))
    outdir = Path(outdir)
    return {
        "model_results": write_table(pd.DataFrame(model_rows), outdir / "model_results.csv"),
        "emmeans_contrasts": write_table(pd.DataFrame(contrast_rows), outdir / "emmeans_contrasts.csv"),
        "model_diagnostics": write_table(pd.DataFrame(diagnostics_rows), outdir / "model_diagnostics.csv"),
    }


def load_model_config(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"families": [{"name": "default_questionnaire", "kind": "continuous", "outcomes": ["q_S1", "q_S2", "q_S3", "q_S4", "q_S5"], "predictors": DEFAULT_PREDICTORS}]}


def fit_model(df: pd.DataFrame, outcome: str, predictors: list[str], kind: str) -> tuple[list[dict], dict]:
    cols = [outcome, "participant_id"] + _raw_predictor_columns(predictors)
    data = df[[c for c in cols if c in df.columns]].copy()
    if outcome.startswith(("q_", "eeg_")) and {"participant_id", "scene_id"}.issubset(df.columns):
        extra_cols = [c for c in ["scene_id"] if c in df.columns and c not in data.columns]
        data = df[[c for c in cols + extra_cols if c in df.columns]].drop_duplicates(["participant_id", "scene_id"]).copy()
    if kind == "binary":
        data[outcome] = data[outcome].map(_binary_value)
    else:
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[outcome])
    if len(data) < 4:
        return [], {"outcome": outcome, "status": "insufficient_data", "n": len(data)}
    if kind == "binary" and data[outcome].nunique(dropna=True) < 2:
        return [], {"outcome": outcome, "status": "no_binary_variation", "n": len(data)}
    usable_predictors = [p for p in predictors if _predictor_available(data, p)]
    if not usable_predictors:
        usable_predictors = ["1"]
    formula = f"{outcome} ~ " + " + ".join(usable_predictors)
    try:
        if kind == "binary":
            fit = smf.logit(formula, data=data).fit(disp=False)
            model_type = "logit"
        else:
            fit = smf.mixedlm(formula, data=data, groups=data["participant_id"]).fit(reml=False, method="lbfgs", disp=False)
            model_type = "mixedlm_random_intercept"
    except Exception as exc:
        fit = smf.ols(formula, data=data).fit()
        model_type = f"ols_fallback_after:{type(exc).__name__}"
    rows = []
    conf = fit.conf_int()
    for term, estimate in fit.params.items():
        rows.append({
            "outcome": outcome,
            "term": term,
            "estimate": float(estimate),
            "std_error": float(fit.bse.get(term, np.nan)),
            "p_value": float(fit.pvalues.get(term, np.nan)),
            "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
            "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
            "model_type": model_type,
            "n": int(fit.nobs),
            "formula": formula,
        })
    return rows, {"outcome": outcome, "status": "fit", "model_type": model_type, "n": int(fit.nobs), "formula": formula}


def wwr_contrasts(df: pd.DataFrame, outcome: str) -> list[dict]:
    if "WWR" not in df.columns or outcome not in df.columns:
        return []
    tmp = df[["WWR", outcome]].copy()
    tmp[outcome] = pd.to_numeric(tmp[outcome], errors="coerce")
    means = tmp.groupby("WWR")[outcome].mean()
    rows = [{"outcome": outcome, "contrast": f"mean_WWR_{level}", "estimate": float(value)} for level, value in means.items() if pd.notna(value)]
    numeric_levels = {float(k): v for k, v in means.items() if pd.notna(v)}
    if {15.0, 45.0, 75.0}.issubset(numeric_levels):
        peak = numeric_levels[45.0] - (numeric_levels[15.0] + numeric_levels[75.0]) / 2
        rows.append({"outcome": outcome, "contrast": "WWR45_minus_mean_WWR15_WWR75", "estimate": float(peak), "claim_strength": "exploratory_trend_only"})
    return rows


def _raw_predictor_columns(predictors: list[str]) -> list[str]:
    cols = []
    for predictor in predictors:
        if predictor == "1":
            continue
        cols.append(predictor.replace("C(", "").replace(")", ""))
    return cols


def _predictor_available(data: pd.DataFrame, predictor: str) -> bool:
    if predictor == "1":
        return True
    col = predictor.replace("C(", "").replace(")", "")
    return col in data.columns and data[col].notna().any() and data[col].nunique(dropna=True) > 1


def _binary_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "no", "n"}:
        return 0.0
    try:
        return 1.0 if float(text) != 0 else 0.0
    except ValueError:
        return np.nan
