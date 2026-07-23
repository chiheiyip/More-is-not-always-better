from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from paper_analysis.utils.io import read_table, write_table
from paper_analysis.stats.canonical import (
    EEG_ORDER_PRIMARY,
    EEG_OUTCOMES,
    _add_order_features,
    _apply_eeg_qc,
    _attach_scene_design,
    _normalize_eeg_columns,
    _normalize_questionnaire,
)


KEYS = ["participant_id", "scene_id"]
PRIMARY_QUESTIONNAIRE = ["q_S1", "q_S2", "q_S3", "q_S4", "q_S5"]


def run_diagnostics(
    master_csv: str | Path,
    participants_csv: str | Path,
    outdir: str | Path = "outputs/06_robustness",
    *,
    questionnaire_csv: str | Path | None = None,
    eye_dynamic_csv: str | Path | None = None,
    eeg_csv: str | Path | None = None,
    eeg_scene_qc_csv: str | Path | None = None,
    scene_manifest_csv: str | Path | None = None,
    model_results_csv: str | Path | None = None,
    model_diagnostics_csv: str | Path | None = None,
) -> dict[str, Path]:
    master = read_table(master_csv)
    participants = read_table(participants_csv)
    models = read_table(model_results_csv) if model_results_csv else pd.DataFrame()
    model_diagnostics = (
        read_table(model_diagnostics_csv) if model_diagnostics_csv else pd.DataFrame()
    )
    order = (
        formal_order_fatigue_effects(models, model_diagnostics)
        if not models.empty
        else order_fatigue_effects(master)
    )
    descriptives = order_fatigue_descriptives(
        questionnaire_csv=questionnaire_csv,
        eye_dynamic_csv=eye_dynamic_csv,
        eeg_csv=eeg_csv,
        eeg_scene_qc_csv=eeg_scene_qc_csv,
        scene_manifest_csv=scene_manifest_csv,
    )
    stability = order_condition_stability(models, model_diagnostics)
    carryover = carryover_sensitivity(models, model_diagnostics)
    gender = factor_sensitivity(master, "Gender")
    batch = factor_sensitivity(master, "RecruitmentBatch")
    date_batch = factor_sensitivity(master, "DateBatch")
    nonlinear = nonlinear_wwr_sensitivity(master)
    power = power_sensitivity(participants)
    datebatch_adjusted = datebatch_adjusted_core_models(master)
    experience_split = experience_split_questionnaire(master)
    effects = effect_size_summary(master)
    outdir = Path(outdir)
    return {
        "order_fatigue_effects": write_table(order, outdir / "order_fatigue_effects.csv"),
        "order_fatigue_descriptives": write_table(descriptives, outdir / "order_fatigue_descriptives.csv"),
        "order_condition_stability": write_table(stability, outdir / "order_condition_stability.csv"),
        "carryover_sensitivity": write_table(carryover, outdir / "carryover_sensitivity.csv"),
        "gender_sensitivity": write_table(gender, outdir / "gender_sensitivity.csv"),
        "batch_sensitivity": write_table(batch, outdir / "batch_sensitivity.csv"),
        "datebatch_sensitivity": write_table(date_batch, outdir / "datebatch_sensitivity.csv"),
        "nonlinear_wwr_sensitivity": write_table(nonlinear, outdir / "nonlinear_wwr_sensitivity.csv"),
        "power_sensitivity": write_table(power, outdir / "power_sensitivity.csv"),
        "datebatch_adjusted_core_models": write_table(datebatch_adjusted, outdir / "datebatch_adjusted_core_models.csv"),
        "experience_split_questionnaire": write_table(experience_split, outdir / "experience_split_questionnaire.csv"),
        "effect_size_summary": write_table(effects, outdir / "effect_size_summary.csv"),
    }


FORMAL_ORDER_COLUMNS = [
    "modality", "grain", "scope", "outcome", "order_term", "estimate",
    "std_error", "ci_low", "ci_high", "effect_scale", "p_value",
    "p_fdr_bh", "n_subjects", "n_trials", "model_type", "fit_status",
    "hypothesis_block", "interpretation_tier", "formula",
]


def formal_order_fatigue_effects(
    models: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return block/position estimates at their native scene-level grain."""
    if models.empty:
        return pd.DataFrame(columns=FORMAL_ORDER_COLUMNS)
    primary_scopes = {"primary_available", "primary_eye_available", "eeg_qc_passed"}
    terms = models.get("term", pd.Series(index=models.index, dtype=str)).astype(str)
    mask = (
        models.get("scope", pd.Series(index=models.index, dtype=str)).isin(primary_scopes)
        & terms.isin(["block", "position"])
        & models.get("hypothesis_block", pd.Series(index=models.index, dtype=str)).astype(str).str.startswith("H4_")
    )
    out = models.loc[mask].copy()
    out["modality"] = out.get("family", "").map(_model_modality)
    out["order_term"] = out["term"]
    out["fit_status"] = out.get("status", "fit")
    if diagnostics is not None and not diagnostics.empty:
        status = diagnostics[
            ["grain", "family", "scope", "outcome", "status"]
        ].drop_duplicates(["grain", "family", "scope", "outcome"])
        status = status.rename(columns={"status": "_diagnostic_status"})
        out = out.merge(
            status,
            on=["grain", "family", "scope", "outcome"],
            how="left",
        )
        out["fit_status"] = out["_diagnostic_status"].fillna(out["fit_status"])
    return out.reindex(columns=FORMAL_ORDER_COLUMNS).sort_values(
        ["modality", "outcome", "order_term"], kind="stable"
    )


def order_fatigue_descriptives(
    *,
    questionnaire_csv: str | Path | None,
    eye_dynamic_csv: str | Path | None,
    eeg_csv: str | Path | None,
    eeg_scene_qc_csv: str | Path | None,
    scene_manifest_csv: str | Path | None,
) -> pd.DataFrame:
    scene = read_table(scene_manifest_csv) if scene_manifest_csv else pd.DataFrame()
    frames: list[tuple[str, pd.DataFrame, list[str]]] = []
    if questionnaire_csv:
        q = _normalize_questionnaire(read_table(questionnaire_csv))
        frames.append(("questionnaire", q, [c for c in PRIMARY_QUESTIONNAIRE if c in q]))
    if eye_dynamic_csv:
        eye = read_table(eye_dynamic_csv)
        eye_outcomes = [
            c for c in (
                "blink_count", "blink_rate_per_min", "angular_scanpath_deg_per_s",
                "pupil_post_early_delta_mm",
            ) if c in eye
        ]
        frames.append(("eye", eye, eye_outcomes))
    if eeg_csv:
        eeg = _normalize_eeg_columns(read_table(eeg_csv))
        if eeg_scene_qc_csv:
            eeg = _apply_eeg_qc(eeg, read_table(eeg_scene_qc_csv))
        frames.append(("eeg", eeg, [c for c in EEG_OUTCOMES if c in eeg]))
    rows: list[dict] = []
    for modality, frame, outcomes in frames:
        work = _add_order_features(_attach_scene_design(frame, scene), scene)
        if set(KEYS).issubset(work.columns):
            work = work.drop_duplicates(KEYS)
        group_cols = [c for c in ("block", "position") if c in work.columns]
        if not group_cols:
            continue
        for outcome in outcomes:
            values = pd.to_numeric(work[outcome], errors="coerce")
            for levels, sub_idx in work.assign(_value=values).groupby(
                group_cols, dropna=False, sort=True
            ).groups.items():
                levels = levels if isinstance(levels, tuple) else (levels,)
                sub = work.loc[sub_idx].copy()
                sub[outcome] = values.loc[sub_idx]
                stats = _summary_stats(sub, outcome)
                rows.append({
                    "modality": modality,
                    "grain": "scene",
                    "outcome": outcome,
                    **dict(zip(group_cols, levels)),
                    **stats,
                    "interpretation_note": (
                        "Descriptive order/fatigue proxy only; block and position "
                        "are not direct fatigue measurements."
                    ),
                })
    return pd.DataFrame(rows)


def order_condition_stability(
    models: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    models = _merge_model_fit_status(models, diagnostics)
    rows: list[dict] = []
    condition = models.get("term", pd.Series(index=models.index, dtype=str)).astype(str).str.contains(
        "WWR|Complexity", regex=True
    )
    controlled = models.loc[
        models.get("scope", pd.Series(index=models.index, dtype=str)).isin(
            ["primary_available", "primary_eye_available", "eeg_qc_passed"]
        ) & condition
    ].copy()
    unadjusted = models.loc[
        models.get("scope", pd.Series(index=models.index, dtype=str)).eq(
            "order_unadjusted_sensitivity"
        ) & condition
    ].copy()
    keys = ["grain", "family", "outcome", "term"]
    compare = controlled.merge(
        unadjusted,
        on=keys,
        suffixes=("_controlled", "_unadjusted"),
        how="inner",
    )
    for row in compare.itertuples():
        controlled_est = float(row.estimate_controlled)
        unadjusted_est = float(row.estimate_unadjusted)
        rows.append({
            "row_type": "condition_coefficient_comparison",
            "modality": _model_modality(row.family),
            "grain": row.grain,
            "outcome": row.outcome,
            "term": row.term,
            "estimate_controlled": controlled_est,
            "ci_low_controlled": row.ci_low_controlled,
            "ci_high_controlled": row.ci_high_controlled,
            "estimate_unadjusted": unadjusted_est,
            "ci_low_unadjusted": row.ci_low_unadjusted,
            "ci_high_unadjusted": row.ci_high_unadjusted,
            "estimate_change": controlled_est - unadjusted_est,
            "direction_changed": (
                np.sign(controlled_est) != np.sign(unadjusted_est)
                and controlled_est != 0 and unadjusted_est != 0
            ),
            "n_subjects": row.n_subjects_controlled,
            "n_trials": row.n_trials_controlled,
            "model_type": row.model_type_controlled,
            "fit_status": row.fit_status_controlled,
        })
    time_rows = models.loc[
        models.get("scope", pd.Series(index=models.index, dtype=str)).eq(
            "order_time_stability_sensitivity"
        )
        & models.get("term", pd.Series(index=models.index, dtype=str)).astype(str).str.contains("trial_index")
    ]
    for row in time_rows.itertuples():
        rows.append({
            "row_type": "time_stability_term",
            "modality": _model_modality(row.family),
            "grain": row.grain,
            "outcome": row.outcome,
            "term": row.term,
            "estimate_controlled": row.estimate,
            "ci_low_controlled": row.ci_low,
            "ci_high_controlled": row.ci_high,
            "p_value": row.p_value,
            "p_fdr_bh": row.p_fdr_bh,
            "n_subjects": row.n_subjects,
            "n_trials": row.n_trials,
            "model_type": row.model_type,
            "fit_status": row.fit_status,
        })
    return _append_failed_model_status(
        pd.DataFrame(rows), diagnostics, "order_time_stability_sensitivity"
    )


def carryover_sensitivity(
    models: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    models = _merge_model_fit_status(models, diagnostics)
    subset = models.loc[
        models.get("scope", pd.Series(index=models.index, dtype=str)).eq(
            "carryover_sensitivity"
        )
    ].copy()
    if not subset.empty:
        subset.insert(0, "modality", subset["family"].map(_model_modality))
    return _append_failed_model_status(
        subset, diagnostics, "carryover_sensitivity"
    )


def _append_failed_model_status(
    table: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    scope: str,
) -> pd.DataFrame:
    if diagnostics is None or diagnostics.empty:
        return table
    failed = diagnostics.loc[
        diagnostics.get("scope", pd.Series(index=diagnostics.index, dtype=str)).eq(scope)
        & ~diagnostics.get("status", pd.Series(index=diagnostics.index, dtype=str)).astype(str).str.startswith("fit")
    ].copy()
    if failed.empty:
        return table
    failed["modality"] = failed.get("family", "").map(_model_modality)
    failed["fit_status"] = failed["status"]
    failed["row_type"] = "model_status"
    return pd.concat([table, failed], ignore_index=True, sort=False)


def _merge_model_fit_status(
    models: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
) -> pd.DataFrame:
    out = models.copy()
    out["fit_status"] = out.get("status", "fit")
    if diagnostics is None or diagnostics.empty:
        return out
    keys = ["grain", "family", "scope", "outcome"]
    available = [key for key in keys if key in out and key in diagnostics]
    if len(available) != len(keys):
        return out
    status = diagnostics[keys + ["status"]].drop_duplicates(keys)
    status = status.rename(columns={"status": "_diagnostic_status"})
    out = out.merge(status, on=keys, how="left")
    out["fit_status"] = out["_diagnostic_status"].fillna(out["fit_status"])
    return out.drop(columns=["_diagnostic_status"])


def _model_modality(family: object) -> str:
    value = str(family)
    if value == "questionnaire":
        return "questionnaire"
    if value == "eeg":
        return "eeg"
    if value.startswith("eye_"):
        return "eye"
    return value


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
        tmp = pd.DataFrame({"WWR": master["WWR"], "_y": values})
        means = tmp.groupby("WWR")["_y"].mean()
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
        tmp = pd.DataFrame({factor: master[factor], "_y": values})
        for level, sub in tmp.groupby(factor, dropna=False):
            rows.append({"factor": factor, "level": level, "outcome": outcome, "n": int(sub["_y"].notna().sum()), "mean": float(sub["_y"].mean()) if sub["_y"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def power_sensitivity(participants: pd.DataFrame) -> pd.DataFrame:
    group_col = "ExperienceGroup" if "ExperienceGroup" in participants.columns else "Experience"
    counts = participants[group_col].fillna("Unknown").astype(str).value_counts()
    rows = [{"grouping": group_col, "level": level, "n": int(n)} for level, n in counts.items()]
    if len(counts) >= 2:
        rows.append({"grouping": group_col, "level": "min_group_n", "n": int(counts.min()), "interpretation": "interaction_power_limited_if_small"})
    return pd.DataFrame(rows)


def datebatch_adjusted_core_models(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for outcome in [c for c in PRIMARY_QUESTIONNAIRE if c in master.columns]:
        data = _outcome_data(master, outcome)
        if "DateBatch" not in data.columns or data["DateBatch"].nunique(dropna=True) < 2:
            rows.append({"outcome": outcome, "status": "missing_or_single_DateBatch", "model_role": "datebatch_adjusted"})
            continue
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
        data = data.dropna(subset=[outcome])
        terms = _datebatch_model_terms(data)
        if len(data) < 8 or not terms:
            rows.append({"outcome": outcome, "status": "insufficient_data", "n": int(len(data)), "model_role": "datebatch_adjusted"})
            continue
        formula = f"{outcome} ~ " + " + ".join(terms)
        try:
            fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["participant_id"]})
            model_type = "ols_cluster_by_participant"
        except Exception as exc:
            fit = smf.ols(formula, data=data).fit()
            model_type = f"ols_fallback_after:{type(exc).__name__}"
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
                "n_subjects": int(data["participant_id"].nunique()) if "participant_id" in data.columns else np.nan,
                "formula": formula,
                "model_role": "datebatch_adjusted",
                "status": "fit",
                "interpretation_note": "Controls DateBatch, ExperienceGroup, Complexity, order, and available demographics; interaction terms are sensitivity checks.",
            })
    return pd.DataFrame(rows)


def experience_split_questionnaire(master: pd.DataFrame) -> pd.DataFrame:
    data = _scene_level(master)
    rows: list[dict] = []
    metrics = [c for c in PRIMARY_QUESTIONNAIRE if c in data.columns]
    specs = [
        ("ExperienceGroup", ["ExperienceGroup"]),
        ("WWR:ExperienceGroup", ["WWR", "ExperienceGroup"]),
        ("Complexity:ExperienceGroup", ["Complexity", "ExperienceGroup"]),
        ("DateBatch:ExperienceGroup", ["DateBatch", "ExperienceGroup"]),
        ("DateBatch:WWR:ExperienceGroup", ["DateBatch", "WWR", "ExperienceGroup"]),
    ]
    for metric in metrics:
        for factor, cols in specs:
            if any(col not in data.columns for col in cols):
                continue
            for keys, sub in data.groupby(cols, dropna=False, sort=True):
                keys = keys if isinstance(keys, tuple) else (keys,)
                base = dict(zip(cols, keys))
                rows.append({
                    "outcome": metric,
                    "factor": factor,
                    "level": ";".join(f"{col}={base[col]}" for col in cols),
                    "ExperienceGroup": base.get("ExperienceGroup", ""),
                    "WWR": base.get("WWR", ""),
                    "Complexity": base.get("Complexity", ""),
                    "DateBatch": base.get("DateBatch", ""),
                    **_summary_stats(sub, metric),
                    "interpretation_note": "Scene-level questionnaire split; use for descriptive moderation exploration, not as proof of interaction by itself.",
                })
    return pd.DataFrame(rows)


def effect_size_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    outcomes = [c for c in _candidate_outcomes(master) if c in master.columns]
    for outcome in outcomes:
        data = _outcome_data(master, outcome)
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
        data = data.dropna(subset=[outcome])
        if data.empty:
            continue
        for factor in ["WWR", "Complexity", "ExperienceGroup", "DateBatch"]:
            if factor not in data.columns or data[factor].nunique(dropna=True) < 2:
                continue
            means = data.groupby(factor, dropna=False)[outcome].mean().dropna()
            if means.empty:
                continue
            leader = means.idxmax()
            lowest = means.idxmin()
            overall_sd = float(data[outcome].std(ddof=1))
            raw_range = float(means.loc[leader] - means.loc[lowest])
            rows.append({
                "outcome": outcome,
                "factor": factor,
                "contrast": "leader_minus_lowest",
                "level_high": leader,
                "level_low": lowest,
                "mean_high": float(means.loc[leader]),
                "mean_low": float(means.loc[lowest]),
                "raw_difference": raw_range,
                "standardized_difference": float(raw_range / overall_sd) if overall_sd and np.isfinite(overall_sd) else np.nan,
                "n_subjects": _unique_subjects(data),
                "n_trials": _unique_trials(data),
                "n_rows": int(len(data)),
                "interpretation_note": "Descriptive effect-size screen; direction is not a causal claim.",
            })
            contrast = _planned_contrast(data, outcome, factor)
            if contrast:
                rows.append(contrast)
    return pd.DataFrame(rows)


def _datebatch_model_terms(data: pd.DataFrame) -> list[str]:
    terms: list[str] = []
    for col in ["WWR", "Complexity", "ExperienceGroup", "DateBatch", "Gender"]:
        if _categorical_available(data, col):
            terms.append(f"C({col})")
    if _categorical_available(data, "WWR") and _categorical_available(data, "DateBatch"):
        terms.append("C(WWR):C(DateBatch)")
    if _categorical_available(data, "WWR") and _categorical_available(data, "ExperienceGroup"):
        terms.append("C(WWR):C(ExperienceGroup)")
    for col in ["Age", "block", "position"]:
        if _numeric_available(data, col):
            terms.append(col)
    return terms


def _planned_contrast(data: pd.DataFrame, outcome: str, factor: str) -> dict | None:
    if factor == "WWR":
        means = data.groupby(factor)[outcome].mean()
        numeric = {float(k): float(v) for k, v in means.items() if pd.notna(v)}
        if not {15.0, 45.0, 75.0}.issubset(numeric):
            return None
        comparator = max(numeric[45.0], numeric[75.0])
        diff = numeric[15.0] - comparator
        sd = float(data[outcome].std(ddof=1))
        return {
            "outcome": outcome,
            "factor": factor,
            "contrast": "WWR15_minus_best_WWR45_or_WWR75",
            "level_high": "WWR15",
            "level_low": "best_non15",
            "mean_high": numeric[15.0],
            "mean_low": comparator,
            "raw_difference": float(diff),
            "standardized_difference": float(diff / sd) if sd and np.isfinite(sd) else np.nan,
            "n_subjects": _unique_subjects(data),
            "n_trials": _unique_trials(data),
            "n_rows": int(len(data)),
            "interpretation_note": "Positive values mean WWR15 exceeds the better of WWR45/WWR75; three-level descriptive contrast only.",
        }
    level_pairs = {
        "Complexity": (1, 0, "Complexity1_minus_0"),
        "ExperienceGroup": ("High", "Low", "High_minus_Low"),
        "DateBatch": ("Second_2026-05-01_or_later", "First_before_2026-05-01", "Second_minus_First"),
    }
    if factor not in level_pairs:
        return None
    high, low, label = level_pairs[factor]
    sub_high = data.loc[data[factor].astype(str).eq(str(high)), outcome].dropna()
    sub_low = data.loc[data[factor].astype(str).eq(str(low)), outcome].dropna()
    if sub_high.empty or sub_low.empty:
        return None
    diff = float(sub_high.mean() - sub_low.mean())
    pooled = _pooled_sd(sub_high, sub_low)
    return {
        "outcome": outcome,
        "factor": factor,
        "contrast": label,
        "level_high": high,
        "level_low": low,
        "mean_high": float(sub_high.mean()),
        "mean_low": float(sub_low.mean()),
        "raw_difference": diff,
        "standardized_difference": float(diff / pooled) if pooled and np.isfinite(pooled) else np.nan,
        "n_subjects": _unique_subjects(data),
        "n_trials": _unique_trials(data),
        "n_rows": int(len(data)),
        "interpretation_note": "Planned descriptive contrast; use with model and balance diagnostics.",
    }


def _pooled_sd(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float(np.sqrt(var)) if var >= 0 else np.nan


def _summary_stats(df: pd.DataFrame, metric: str) -> dict:
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    n = int(len(values))
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan
    return {
        "n_subjects": _unique_subjects(df.loc[values.index]) if n else 0,
        "n_trials": _unique_trials(df.loc[values.index]) if n else 0,
        "n_rows": n,
        "mean": float(values.mean()) if n else np.nan,
        "sd": sd,
        "se": se,
        "ci95_low": float(values.mean() - 1.96 * se) if np.isfinite(se) else np.nan,
        "ci95_high": float(values.mean() + 1.96 * se) if np.isfinite(se) else np.nan,
        "median": float(values.median()) if n else np.nan,
    }


def _outcome_data(master: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = _scene_level(master) if outcome.startswith(("q_", "eeg_")) else master.copy()
    return data.copy()


def _scene_level(df: pd.DataFrame) -> pd.DataFrame:
    if set(KEYS).issubset(df.columns):
        return df.drop_duplicates(KEYS).copy()
    return df.copy()


def _categorical_available(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any() and df[col].nunique(dropna=True) > 1


def _numeric_available(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    return values.notna().any() and values.nunique(dropna=True) > 1


def _unique_subjects(df: pd.DataFrame) -> int:
    return int(df["participant_id"].astype(str).nunique()) if "participant_id" in df.columns and not df.empty else 0


def _unique_trials(df: pd.DataFrame) -> int:
    if set(KEYS).issubset(df.columns) and not df.empty:
        return int(df[KEYS].drop_duplicates().shape[0])
    return 0


def _candidate_outcomes(master: pd.DataFrame) -> list[str]:
    tokens = ["q_S", "theta", "alpha", "beta", "FCR", "TFD", "TTFF", "attention_share"]
    return [c for c in master.columns if any(token.lower() in c.lower() for token in tokens)]
