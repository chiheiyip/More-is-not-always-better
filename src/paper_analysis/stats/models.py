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
    eeg_outcomes = eeg_outcomes_from_config(config, master)
    eeg_core = eeg_core_lmm(master, eeg_outcomes)
    eeg_peak = eeg_peak_index_models(master, eeg_outcomes)
    eeg_trial = eeg_trial_index_models(master, eeg_outcomes)
    outdir = Path(outdir)
    return {
        "model_results": write_table(pd.DataFrame(model_rows), outdir / "model_results.csv"),
        "emmeans_contrasts": write_table(pd.DataFrame(contrast_rows), outdir / "emmeans_contrasts.csv"),
        "model_diagnostics": write_table(pd.DataFrame(diagnostics_rows), outdir / "model_diagnostics.csv"),
        "eeg_core_lmm": write_table(pd.DataFrame(eeg_core), outdir / "eeg_core_lmm.csv"),
        "eeg_peak_index": write_table(pd.DataFrame(eeg_peak), outdir / "eeg_peak_index.csv"),
        "eeg_trial_index_models": write_table(pd.DataFrame(eeg_trial), outdir / "eeg_trial_index_models.csv"),
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


def eeg_outcomes_from_config(config: dict, master: pd.DataFrame) -> list[str]:
    outcomes: list[str] = []
    for family in config.get("families", []):
        for outcome in family.get("outcomes", []):
            if outcome.startswith("eeg_") and outcome in master.columns:
                outcomes.append(outcome)
    if not outcomes:
        outcomes = [c for c in master.columns if c.startswith("eeg_") and any(token in c.lower() for token in ["theta", "alpha", "beta"])]
    return sorted(dict.fromkeys(outcomes))


def eeg_core_lmm(df: pd.DataFrame, outcomes: list[str]) -> list[dict]:
    rows: list[dict] = []
    for outcome in outcomes:
        formula = f"{outcome} ~ " + " + ".join(_eeg_core_terms(df))
        result = _fit_formula_rows(df, outcome, formula, "eeg_core_lmm")
        rows.extend(result)
    return rows or [{"model": "eeg_core_lmm", "status": "no_eeg_outcomes"}]


def eeg_peak_index_models(df: pd.DataFrame, outcomes: list[str]) -> list[dict]:
    rows: list[dict] = []
    if "WWR" not in df.columns or "participant_id" not in df.columns:
        return [{"model": "eeg_peak_index", "status": "missing_WWR_or_participant_id"}]
    wwr_num = _wwr_numeric_series(df)
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        data = df.copy()
        data["_wwr_numeric"] = wwr_num
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
        group_cols = ["participant_id"] + [c for c in ["Complexity", "ExperienceGroup", "Gender", "Age"] if c in data.columns]
        peak_rows = []
        for keys, sub in data.dropna(subset=[outcome, "_wwr_numeric"]).groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
            means = sub.groupby("_wwr_numeric")[outcome].mean()
            if {15.0, 45.0, 75.0}.issubset(set(means.index.astype(float))):
                peak = float(means.loc[45.0] - (means.loc[15.0] + means.loc[75.0]) / 2.0)
                peak_rows.append({**base, "metric": outcome, "peak_index": peak})
        if not peak_rows:
            rows.append({"model": "eeg_peak_index", "metric": outcome, "status": "insufficient_complete_WWR_levels"})
            continue
        peak_df = pd.DataFrame(peak_rows)
        for _, row in peak_df.iterrows():
            rows.append({
                "model": "eeg_peak_index",
                "metric": outcome,
                "row_type": "peak_index_value",
                "participant_id": row.get("participant_id"),
                "Complexity": row.get("Complexity"),
                "ExperienceGroup": row.get("ExperienceGroup"),
                "Gender": row.get("Gender"),
                "estimate": float(row["peak_index"]),
                "status": "computed",
            })
        formula = "peak_index ~ " + " + ".join(_peak_index_terms(peak_df))
        model_rows = _fit_formula_rows(peak_df, "peak_index", formula, "eeg_peak_index")
        for row in model_rows:
            row["metric"] = outcome
            row["row_type"] = "model_term"
        rows.extend(model_rows)
    return rows or [{"model": "eeg_peak_index", "status": "no_eeg_outcomes"}]


def eeg_trial_index_models(df: pd.DataFrame, outcomes: list[str]) -> list[dict]:
    rows: list[dict] = []
    if "participant_id" not in df.columns:
        return [{"model": "eeg_trial_index", "status": "missing_participant_id"}]
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        data = df.copy()
        if "position" in data.columns and pd.to_numeric(data["position"], errors="coerce").notna().any():
            data["TrialIndex"] = pd.to_numeric(data["position"], errors="coerce")
        elif "scene_id" in data.columns:
            data["TrialIndex"] = pd.to_numeric(data["scene_id"], errors="coerce")
        else:
            rows.append({"model": "eeg_trial_index", "outcome": outcome, "status": "missing_trial_index"})
            continue
        terms = _trial_index_terms(data)
        formula = f"{outcome} ~ " + " + ".join(terms)
        rows.extend(_fit_formula_rows(data, outcome, formula, "eeg_trial_index"))
    return rows or [{"model": "eeg_trial_index", "status": "no_eeg_outcomes"}]


def _fit_formula_rows(df: pd.DataFrame, outcome: str, formula: str, model_label: str) -> list[dict]:
    data = df.copy()
    if outcome not in data.columns:
        return [{"model": model_label, "outcome": outcome, "status": "missing_outcome", "formula": formula}]
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[outcome])
    if len(data) < 4:
        return [{"model": model_label, "outcome": outcome, "status": "insufficient_data", "n": int(len(data)), "formula": formula}]
    try:
        if "participant_id" in data.columns and data["participant_id"].nunique(dropna=True) > 1:
            fit = smf.mixedlm(formula, data=data, groups=data["participant_id"]).fit(reml=False, method="lbfgs", disp=False)
            model_type = "mixedlm_random_intercept"
        else:
            fit = smf.ols(formula, data=data).fit()
            model_type = "ols"
    except Exception as exc:
        try:
            fit = smf.ols(formula, data=data).fit()
            model_type = f"ols_fallback_after:{type(exc).__name__}"
        except Exception as exc2:
            return [{"model": model_label, "outcome": outcome, "status": f"fit_failed:{type(exc2).__name__}", "n": int(len(data)), "formula": formula}]
    conf = fit.conf_int()
    rows = []
    for term, estimate in fit.params.items():
        rows.append({
            "model": model_label,
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
            "status": "fit",
        })
    return rows


def _eeg_core_terms(df: pd.DataFrame) -> list[str]:
    terms: list[str] = []
    if all(_categorical_available(df, col) for col in ["WWR", "Complexity", "ExperienceGroup"]):
        terms.append("C(WWR) * C(Complexity) * C(ExperienceGroup)")
    else:
        for col in ["WWR", "Complexity", "ExperienceGroup"]:
            if _categorical_available(df, col):
                terms.append(f"C({col})")
    for col in ["Gender", "RecruitmentBatch"]:
        if _categorical_available(df, col):
            terms.append(f"C({col})")
    for col in ["Age", "block", "position"]:
        if _numeric_available(df, col):
            terms.append(col)
    return terms or ["1"]


def _peak_index_terms(df: pd.DataFrame) -> list[str]:
    terms: list[str] = []
    if all(_categorical_available(df, col) for col in ["Complexity", "ExperienceGroup"]):
        terms.append("C(Complexity) * C(ExperienceGroup)")
    else:
        for col in ["Complexity", "ExperienceGroup"]:
            if _categorical_available(df, col):
                terms.append(f"C({col})")
    if _categorical_available(df, "Gender"):
        terms.append("C(Gender)")
    if _numeric_available(df, "Age"):
        terms.append("Age")
    return terms or ["1"]


def _trial_index_terms(df: pd.DataFrame) -> list[str]:
    terms = _eeg_core_terms(df)
    terms = [term for term in terms if term not in {"position"}]
    if _numeric_available(df, "TrialIndex"):
        terms.append("TrialIndex")
        if _categorical_available(df, "ExperienceGroup"):
            terms.append("C(ExperienceGroup):TrialIndex")
        if _categorical_available(df, "order_scheme"):
            terms.append("C(order_scheme)")
            terms.append("C(order_scheme):TrialIndex")
    return terms or ["1"]


def _categorical_available(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any() and df[col].nunique(dropna=True) > 1


def _numeric_available(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    return values.notna().any() and values.nunique(dropna=True) > 1


def _wwr_numeric_series(df: pd.DataFrame) -> pd.Series:
    if "WWR_numeric" in df.columns:
        values = pd.to_numeric(df["WWR_numeric"], errors="coerce")
        if values.notna().any():
            return values
    text = df["WWR"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(text, errors="coerce")


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
