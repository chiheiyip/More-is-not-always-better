from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


S_ITEMS = ["S1", "S2", "S3", "S4", "S5"]
B_ITEMS = ["B1", "B2", "B3"]
IPQ_ITEMS = [f"IPQ{i}" for i in range(1, 7)]
PRIMARY_DVS = S_ITEMS + B_ITEMS + ["Bmean", "Afford4"]
LINEAR_COEF = np.array([-1.0, 0.0, 1.0])
QUADRATIC_COEF = np.array([1.0, -2.0, 1.0])


def item_level_lmm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict] = []
    diagnostics: list[dict] = []
    marginal: list[dict] = []
    for outcome in [c for c in PRIMARY_DVS if c in df.columns]:
        data, formula, status = _model_data_and_formula(df, outcome)
        if status != "ready":
            diagnostics.append({"outcome": outcome, "status": status, "n": int(len(data))})
            continue
        try:
            fit = smf.mixedlm(formula, data=data, groups=data["participant_id"]).fit(reml=False, method="lbfgs", disp=False)
            model_type = "mixedlm_random_intercept"
        except Exception as exc:
            try:
                fit = smf.ols(formula, data=data).fit()
                model_type = f"ols_fallback_after:{type(exc).__name__}"
            except Exception as exc2:
                diagnostics.append({"outcome": outcome, "status": f"fit_failed:{type(exc2).__name__}", "n": int(len(data)), "formula": formula})
                continue
        conf = fit.conf_int()
        for term, estimate in fit.params.items():
            if " Var" in str(term) or str(term).endswith("Var"):
                continue
            results.append({
                "outcome": outcome,
                "term": term,
                "estimate": float(estimate),
                "std_error": float(fit.bse.get(term, np.nan)),
                "p_value": float(fit.pvalues.get(term, np.nan)),
                "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                "model_type": model_type,
                "n": int(getattr(fit, "nobs", len(data))),
                "n_subjects": int(data["participant_id"].nunique()),
                "formula": formula,
                "note": "Python LMM coefficient table; not labeled as SPSS Type III/EMM.",
            })
        diagnostics.append({
            "outcome": outcome,
            "status": "fit",
            "model_type": model_type,
            "n": int(getattr(fit, "nobs", len(data))),
            "n_subjects": int(data["participant_id"].nunique()),
            "formula": formula,
            "aic": float(getattr(fit, "aic", np.nan)) if pd.notna(getattr(fit, "aic", np.nan)) else np.nan,
            "bic": float(getattr(fit, "bic", np.nan)) if pd.notna(getattr(fit, "bic", np.nan)) else np.nan,
        })
        marginal.extend(_marginal_means(data, outcome))
    return pd.DataFrame(results), pd.DataFrame(diagnostics), pd.DataFrame(marginal)


def wwr_polynomial_contrasts(df: pd.DataFrame, levels: tuple[float, float, float] = (15.0, 45.0, 75.0)) -> pd.DataFrame:
    if "participant_id" not in df.columns or "WWR" not in df.columns:
        return pd.DataFrame([{"status": "missing_participant_id_or_WWR"}])
    rows: list[dict] = []
    level_cols = [f"WWR_{_level_label(level)}" for level in levels]
    data = df.copy()
    data["_WWR_numeric"] = _wwr_numeric(data)
    for outcome in [c for c in PRIMARY_DVS if c in data.columns]:
        sub = data.dropna(subset=["participant_id", "_WWR_numeric", outcome]).copy()
        if sub.empty:
            rows.append({"outcome": outcome, "status": "no_data"})
            continue
        subject_means = sub.groupby(["participant_id", "_WWR_numeric"], as_index=False)[outcome].mean()
        wide = subject_means.pivot_table(index="participant_id", columns="_WWR_numeric", values=outcome, aggfunc="mean")
        for level in levels:
            if level not in wide.columns:
                wide[level] = np.nan
        wide = wide[list(levels)].rename(columns={level: col for level, col in zip(levels, level_cols)})
        mat = wide[level_cols].dropna(how="any").to_numpy(dtype=float)
        for label, coef in [("Linear", LINEAR_COEF), ("Quadratic", QUADRATIC_COEF)]:
            stat = _contrast_stats(mat, coef)
            rows.append({
                "outcome": outcome,
                "contrast": label,
                "coefficient_vector": " | ".join(f"{col}:{float(c):g}" for col, c in zip(level_cols, coef)),
                "direction": _direction(label, stat["mean_contrast"]),
                "n_subjects": stat["n_subjects"],
                "mean_contrast": stat["mean_contrast"],
                "sd_contrast": stat["sd_contrast"],
                "se_contrast": stat["se_contrast"],
                "t_value": stat["t"],
                "f_value": stat["f"],
                "df1": 1.0,
                "df2": stat["df2"],
                "p_value": stat["p"],
                "partial_eta2": stat["partial_eta2"],
                "status": "ok" if stat["n_subjects"] >= 2 else "insufficient_complete_levels",
                "claim_strength": "trend_only",
                "note": "Three WWR levels support linear/quadratic trend language, not a definitive optimum claim.",
            })
    return pd.DataFrame(rows)


def ipq_subject_level_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ipq_cols = [c for c in IPQ_ITEMS if c in df.columns]
    if len(ipq_cols) < 2 or "participant_id" not in df.columns:
        empty = pd.DataFrame([{"status": "no_ipq_items_or_participant_id"}])
        return {
            "ipq_subject_level": empty,
            "ipq_reliability": empty,
            "ipq_group_comparisons": empty,
        }
    agg = {c: "mean" for c in ipq_cols}
    for col in ["ExperienceGroup", "SportFreqGroup", "Gender", "Age"]:
        if col in df.columns:
            agg[col] = "first"
    subject = df.groupby("participant_id", as_index=False).agg(agg)
    subject["IPQ_n_valid"] = subject[ipq_cols].notna().sum(axis=1)
    subject["IPQ_mean"] = subject[ipq_cols].mean(axis=1, skipna=True)
    reliability = pd.DataFrame([{
        "scale": "IPQ1_IPQ6_subject_level",
        "items": ",".join(ipq_cols),
        "k_items": len(ipq_cols),
        "valid_rows": int(subject[ipq_cols].dropna(how="any").shape[0]),
        "cronbach_alpha": _cronbach_alpha(subject[ipq_cols].dropna(how="any")),
        "status": "subject_level_only",
        "note": "IPQ is participant-level; WWR/Complexity trial-level inference is not reported.",
    }])
    comparisons = []
    for group_col in [c for c in ["ExperienceGroup", "SportFreqGroup", "Gender"] if c in subject.columns]:
        comparisons.extend(_welch_pairwise(subject, group_col, "IPQ_mean"))
    return {
        "ipq_subject_level": subject,
        "ipq_reliability": reliability,
        "ipq_group_comparisons": pd.DataFrame(comparisons) if comparisons else pd.DataFrame([{"status": "no_group_comparisons"}]),
    }


def _model_data_and_formula(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, str, str]:
    predictors = []
    if _categorical_available(df, "WWR"):
        predictors.append("C(WWR)")
    if outcome in B_ITEMS + ["Bmean"]:
        # B items are C1-only by design; including Complexity would be mostly empty or non-identifiable.
        pass
    elif _categorical_available(df, "Complexity"):
        predictors.append("C(Complexity)")
    if _categorical_available(df, "ExperienceGroup"):
        predictors.append("C(ExperienceGroup)")
    for col in ["Gender", "RecruitmentBatch"]:
        if _categorical_available(df, col):
            predictors.append(f"C({col})")
    for col in ["Age", "block", "position"]:
        if _numeric_available(df, col):
            predictors.append(col)
    if not predictors:
        predictors = ["1"]
    raw_cols = [p.replace("C(", "").replace(")", "") for p in predictors if p != "1"]
    cols = ["participant_id", outcome] + [c for c in raw_cols if c in df.columns]
    data = df[cols].copy()
    if outcome in B_ITEMS + ["Bmean"] and "Complexity" in df.columns:
        data = data.loc[_complexity_is_c1(df).fillna(False).reindex(data.index, fill_value=False)]
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=["participant_id", outcome])
    if len(data) < 4:
        return data, "", "insufficient_data"
    if data["participant_id"].nunique() < 2:
        return data, "", "insufficient_subjects"
    usable = [p for p in predictors if p == "1" or _predictor_available(data, p)]
    if not usable:
        usable = ["1"]
    return data, f"{outcome} ~ " + " + ".join(usable), "ready"


def _marginal_means(data: pd.DataFrame, outcome: str) -> list[dict]:
    rows: list[dict] = []
    for factor in ["WWR", "Complexity", "ExperienceGroup"]:
        if factor not in data.columns:
            continue
        grouped = data.groupby(factor, dropna=False)[outcome]
        for level, values in grouped:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            rows.append({
                "outcome": outcome,
                "factor": factor,
                "level": level,
                "n": int(numeric.shape[0]),
                "mean": float(numeric.mean()) if not numeric.empty else np.nan,
                "sd": float(numeric.std(ddof=1)) if numeric.shape[0] > 1 else np.nan,
                "note": "Observed marginal mean, not model-estimated EMM.",
            })
    return rows


def _contrast_stats(mat: np.ndarray, coef: np.ndarray) -> dict[str, float]:
    if mat.ndim != 2 or mat.shape[1] != len(coef):
        return _empty_contrast(0)
    score = np.dot(mat, coef)
    score = score[np.isfinite(score)]
    n = int(len(score))
    if n < 2:
        out = _empty_contrast(n)
        out["mean_contrast"] = float(np.nanmean(score)) if n else np.nan
        return out
    mean_score = float(np.mean(score))
    sd_score = float(np.std(score, ddof=1))
    se_score = float(sd_score / np.sqrt(n)) if sd_score > 0 else np.nan
    df2 = float(n - 1)
    t_value = float(mean_score / se_score) if np.isfinite(se_score) and se_score > 0 else np.nan
    f_value = float(t_value**2) if np.isfinite(t_value) else np.nan
    p_value = float(stats.f.sf(f_value, 1, df2)) if np.isfinite(f_value) else np.nan
    partial_eta2 = float(f_value / (f_value + df2)) if np.isfinite(f_value) and f_value + df2 > 0 else np.nan
    return {
        "n_subjects": n,
        "mean_contrast": mean_score,
        "sd_contrast": sd_score,
        "se_contrast": se_score,
        "t": t_value,
        "f": f_value,
        "df2": df2,
        "p": p_value,
        "partial_eta2": partial_eta2,
    }


def _empty_contrast(n: int) -> dict[str, float]:
    return {"n_subjects": n, "mean_contrast": np.nan, "sd_contrast": np.nan, "se_contrast": np.nan, "t": np.nan, "f": np.nan, "df2": np.nan, "p": np.nan, "partial_eta2": np.nan}


def _direction(label: str, value: float) -> str:
    if not np.isfinite(value):
        return ""
    if label == "Linear":
        if value > 0:
            return "linear_increase_15_to_75"
        if value < 0:
            return "linear_decrease_15_to_75"
        return "linear_flat"
    if value > 0:
        return "u_shape_45_lowest"
    if value < 0:
        return "inverted_u_45_highest"
    return "quadratic_flat"


def _wwr_numeric(df: pd.DataFrame) -> pd.Series:
    if "WWR_numeric" in df.columns:
        values = pd.to_numeric(df["WWR_numeric"], errors="coerce")
        if values.notna().any():
            return values
    return pd.to_numeric(df["WWR"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")


def _level_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _welch_pairwise(df: pd.DataFrame, group_col: str, outcome: str) -> list[dict]:
    rows: list[dict] = []
    x = df.dropna(subset=[group_col, outcome]).copy()
    levels = sorted(x[group_col].astype(str).unique())
    for a, b in combinations(levels, 2):
        va = pd.to_numeric(x.loc[x[group_col].astype(str).eq(a), outcome], errors="coerce").dropna()
        vb = pd.to_numeric(x.loc[x[group_col].astype(str).eq(b), outcome], errors="coerce").dropna()
        if len(va) < 2 or len(vb) < 2:
            t_value = p_value = np.nan
        else:
            res = stats.ttest_ind(va, vb, equal_var=False, nan_policy="omit")
            t_value = float(res.statistic)
            p_value = float(res.pvalue)
        rows.append({"group_col": group_col, "outcome": outcome, "group_a": a, "group_b": b, "n_a": int(len(va)), "n_b": int(len(vb)), "mean_a": float(va.mean()) if len(va) else np.nan, "mean_b": float(vb.mean()) if len(vb) else np.nan, "t_welch": t_value, "p_value": p_value})
    return rows


def _cronbach_alpha(items: pd.DataFrame) -> float:
    if items.shape[0] < 3 or items.shape[1] < 2:
        return np.nan
    variances = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if not math.isfinite(float(total_var)) or total_var <= 0:
        return np.nan
    k = items.shape[1]
    return float((k / (k - 1)) * (1 - variances.sum() / total_var))


def _complexity_is_c1(df: pd.DataFrame) -> pd.Series:
    if "Complexity" in df.columns:
        comp = df["Complexity"].astype(str).str.strip().str.lower()
        return comp.isin({"1", "1.0", "c1", "high", "true"})
    if "Cond" in df.columns:
        return df["Cond"].astype(str).str.strip().str.upper().eq("C1")
    return pd.Series(False, index=df.index)


def _categorical_available(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any() and df[col].nunique(dropna=True) > 1


def _numeric_available(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    return values.notna().any() and values.nunique(dropna=True) > 1


def _predictor_available(data: pd.DataFrame, predictor: str) -> bool:
    if predictor == "1":
        return True
    col = predictor.replace("C(", "").replace(")", "")
    return col in data.columns and data[col].notna().any() and data[col].nunique(dropna=True) > 1
