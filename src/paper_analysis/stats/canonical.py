"""Canonical, modality-specific statistical analysis.

The eye-tracking sample is constructed from eye tables and eye QC only.  EEG
quality flags are never joined into an eye eligibility decision.  Every fitted
model is a participant-clustered GEE; failed or non-finite fits are diagnosed
and are never replaced by ordinary least squares.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from paper_analysis.utils.io import read_table, write_table


KEYS = ["participant_id", "scene_id"]
QUESTIONNAIRE_OUTCOMES = ["q_S1", "q_S2", "q_S3", "q_S4", "q_S5"]
EEG_OUTCOMES = [
    f"eeg_{roi}_{band}"
    for roi in ("F", "P", "O")
    for band in ("theta", "alpha", "beta")
]
EEG_ORDER_PRIMARY = {
    f"eeg_{roi}_{band}"
    for roi in ("F", "P", "O")
    for band in ("theta", "alpha")
}
PRIMARY_SCOPES = {"primary_eye_available", "primary_available", "eeg_qc_passed"}
PRIMARY_EYE_TRIAL = [
    "aoi_transition_rate_per_min",
    "transition_entropy_normalized",
    "angular_scanpath_deg_per_s",
    "saccade_rate_per_min",
    "first_window_fixation_latency_ms",
    "window_entry_count",
    "window_directed_transition_rate_per_min",
    "blink_count",
    "blink_rate_per_min",
    "pupil_post_early_delta_mm",
]
MODEL_COLUMNS = [
    "grain", "family", "scope", "hypothesis_block", "interpretation_tier",
    "outcome", "term", "estimate", "std_error", "effect_scale", "ci_low",
    "ci_high", "p_value", "p_fdr_bh", "p_fdr_bh_family", "p_fdr_bh_all",
    "significant_fdr_bh_0_05", "significant_fdr_bh_family_0_05", "n_obs",
    "n_subjects", "n_trials", "model_type", "formula", "status",
]
DIAGNOSTIC_COLUMNS = [
    "grain", "family", "scope", "outcome", "status", "n_obs",
    "n_subjects", "n_trials", "model_type", "formula", "error",
]


def run_canonical_analysis(
    questionnaire_csv: str | Path,
    eye_aoi_csv: str | Path,
    eye_dynamic_csv: str | Path,
    eye_qc_csv: str | Path,
    eeg_csv: str | Path,
    eeg_scene_qc_csv: str | Path,
    participants_csv: str | Path,
    outdir: str | Path = "outputs/06_models",
    mde_simulations: int = 1000,
    random_seed: int = 20260713,
    scene_manifest_csv: str | Path | None = None,
    trimodal_qc_csv: str | Path | None = None,
) -> dict[str, Path]:
    """Fit the single authoritative model package and supporting sample audits."""
    participants = read_table(participants_csv)
    scene = read_table(scene_manifest_csv) if scene_manifest_csv else pd.DataFrame()
    trimodal_qc = read_table(trimodal_qc_csv) if trimodal_qc_csv else pd.DataFrame()
    questionnaire = _attach_participants(_normalize_questionnaire(read_table(questionnaire_csv)), participants)
    eye_aoi = _attach_participants(read_table(eye_aoi_csv), participants)
    eye_dynamic = _attach_participants(read_table(eye_dynamic_csv), participants)
    if set(KEYS).issubset(eye_aoi.columns) and set(KEYS + ["valid_eye_duration_s"]).issubset(eye_dynamic.columns):
        eye_aoi = eye_aoi.merge(
            eye_dynamic[KEYS + ["valid_eye_duration_s"]].drop_duplicates(KEYS),
            on=KEYS, how="left", suffixes=("", "_dynamic"),
        )
    eye_qc = read_table(eye_qc_csv)
    eeg = _attach_participants(_normalize_eeg_columns(read_table(eeg_csv)), participants)
    eeg = _apply_eeg_qc(eeg, read_table(eeg_scene_qc_csv))
    questionnaire = _add_order_features(_attach_scene_design(questionnaire, scene), scene)
    eye_aoi = _add_order_features(_attach_scene_design(eye_aoi, scene), scene)
    eye_dynamic = _add_order_features(_attach_scene_design(eye_dynamic, scene), scene)
    eeg = _add_order_features(_attach_scene_design(eeg, scene), scene)

    rows: list[dict] = []
    diagnostics: list[dict] = []
    _fit_questionnaire(questionnaire, rows, diagnostics)
    _fit_eeg(eeg, rows, diagnostics)
    _fit_eye_aoi(eye_aoi, rows, diagnostics)
    _fit_eye_trial(eye_dynamic, rows, diagnostics, scope="primary_eye_available")

    sensitivity_rows: list[dict] = []
    for threshold in (0.5, 0.6, 0.7, 0.8):
        selected = _eye_threshold_subset(eye_dynamic, eye_qc, threshold)
        selected_aoi = _eye_threshold_subset(eye_aoi, eye_qc, threshold)
        before = len(rows)
        _fit_eye_aoi(selected_aoi, rows, diagnostics, scope=f"eye_valid_coordinates_ge_{threshold:.1f}")
        _fit_eye_trial(selected, rows, diagnostics, scope=f"eye_valid_coordinates_ge_{threshold:.1f}")
        sensitivity_rows.extend(rows[before:])

    models = _apply_fdr(pd.DataFrame(rows))
    sensitivity = models.loc[models.get("scope", pd.Series(dtype=str)).astype(str).str.startswith("eye_valid_coordinates_ge_")].copy()
    availability = metric_sample_summary(questionnaire, eye_aoi, eye_dynamic, eeg)
    sample_flow = modality_sample_flow(
        questionnaire, eye_aoi, eye_dynamic, eye_qc, eeg, trimodal_qc
    )
    mde = monte_carlo_mde(
        eye_dynamic,
        outcomes=[c for c in PRIMARY_EYE_TRIAL if c in eye_dynamic.columns],
        simulations=mde_simulations,
        seed=random_seed,
    )
    outdir = Path(outdir)
    return {
        "model_results": write_table(models.reindex(columns=MODEL_COLUMNS), outdir / "model_results.csv"),
        "model_diagnostics": write_table(pd.DataFrame(diagnostics).reindex(columns=DIAGNOSTIC_COLUMNS), outdir / "model_diagnostics.csv"),
        "eye_qc_sensitivity_models": write_table(sensitivity.reindex(columns=MODEL_COLUMNS), outdir / "eye_qc_sensitivity_models.csv"),
        "metric_sample_summary": write_table(availability, outdir / "metric_sample_summary.csv"),
        "modality_sample_flow": write_table(sample_flow, outdir / "modality_sample_flow.csv"),
        "eye_mde_monte_carlo": write_table(mde, outdir / "eye_mde_monte_carlo.csv"),
    }


def _fit_questionnaire(data: pd.DataFrame, rows: list[dict], diagnostics: list[dict]) -> None:
    for outcome in QUESTIONNAIRE_OUTCOMES:
        _fit_spec(data, outcome, "questionnaire_scene", "questionnaire", "primary_available", "gaussian", rows, diagnostics)


def _fit_eeg(data: pd.DataFrame, rows: list[dict], diagnostics: list[dict]) -> None:
    for outcome in EEG_OUTCOMES:
        _fit_spec(data, outcome, "eeg_scene", "eeg", "eeg_qc_passed", "gamma", rows, diagnostics, positive=True)


def _fit_eye_aoi(data: pd.DataFrame, rows: list[dict], diagnostics: list[dict], scope: str = "primary_eye_available") -> None:
    if data.empty:
        return
    _fit_spec(data, "visited", "aoi_trial", "eye_aoi_visited", scope, "binomial", rows, diagnostics, aoi=True)
    visited = data.loc[_truthy(data.get("visited", pd.Series(False, index=data.index)))].copy()
    for outcome, family in [
        ("FC", "negative_binomial"),
        ("TFD_ms", "gamma"),
        ("FCR", "gamma"),
        ("TTFF_ms", "gamma"),
        ("attention_share", "gaussian_logit"),
        ("median_revisit_latency_ms", "gamma"),
    ]:
        _fit_spec(
            visited, outcome, "aoi_trial_visited", f"eye_aoi_{outcome}",
            scope, family, rows, diagnostics, aoi=True,
            positive=family == "gamma", offset_col="valid_eye_duration_s" if outcome == "FC" else None,
        )


def _fit_eye_trial(data: pd.DataFrame, rows: list[dict], diagnostics: list[dict], scope: str) -> None:
    specs = [
        ("aoi_transition_count", "negative_binomial", "valid_eye_duration_s"),
        ("aoi_transition_rate_per_min", "gamma", None),
        ("transition_entropy_normalized", "gaussian", None),
        ("angular_scanpath_deg_per_s", "gamma", None),
        ("saccade_count", "negative_binomial", "valid_eye_duration_s"),
        ("saccade_rate_per_min", "gamma", None),
        ("saccade_amplitude_median_px", "gamma", None),
        ("saccade_velocity_average_median_px_per_ms", "gamma", None),
        ("saccade_velocity_peak_median_px_per_ms", "gamma", None),
        ("window_entry_count", "negative_binomial", "valid_eye_duration_s"),
        ("window_directed_transition_rate_per_min", "gamma", None),
        ("first_window_fixation_latency_ms", "gamma", None),
        ("blink_count", "negative_binomial", "valid_eye_duration_s"),
        ("blink_rate_per_min", "gamma", None),
        ("pupil_post_early_delta_mm", "gaussian", None),
    ]
    for outcome, family, offset in specs:
        _fit_spec(
            data, outcome, "scene_trial", f"eye_trial_{outcome}", scope,
            family, rows, diagnostics, positive=family == "gamma", offset_col=offset,
            exploratory=outcome.startswith("pupil_") or outcome not in PRIMARY_EYE_TRIAL,
        )


def _fit_spec(
    data: pd.DataFrame,
    outcome: str,
    grain: str,
    model_family: str,
    scope: str,
    family: str,
    rows: list[dict],
    diagnostics: list[dict],
    *,
    aoi: bool = False,
    positive: bool = False,
    offset_col: str | None = None,
    exploratory: bool = False,
) -> None:
    if outcome not in data.columns:
        diagnostics.append(_diag(grain, model_family, scope, outcome, "missing_outcome"))
        return
    work = data.copy()
    if family == "binomial":
        work[outcome] = _truthy(work[outcome]).astype(int)
    else:
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    if family == "gaussian_logit":
        clipped = work[outcome].clip(1e-6, 1 - 1e-6)
        work["_model_outcome"] = np.log(clipped / (1 - clipped))
        response = "_model_outcome"
    else:
        response = outcome
    if positive:
        work = work.loc[work[outcome] > 0].copy()
    formula = _formula(work, response, aoi=aoi, sensitivity=False, order_policy="primary")
    _fit_gee(work, response, outcome, formula, family, grain, model_family, scope, rows, diagnostics, offset_col, exploratory)
    if scope in PRIMARY_SCOPES:
        for sensitivity_scope, order_policy in [
            ("order_unadjusted_sensitivity", "unadjusted"),
            ("order_time_stability_sensitivity", "time_stability"),
            ("carryover_sensitivity", "carryover"),
        ]:
            sensitivity_formula = _formula(
                work,
                response,
                aoi=aoi,
                sensitivity=False,
                order_policy=order_policy,
            )
            if sensitivity_formula != formula:
                _fit_gee(
                    work, response, outcome, sensitivity_formula, family, grain,
                    model_family, sensitivity_scope, rows, diagnostics,
                    offset_col, True,
                )
    # Demographic covariates are a predefined sensitivity model, not part of
    # every primary model.
    if scope == "primary_eye_available":
        sensitivity_formula = _formula(
            work, response, aoi=aoi, sensitivity=True, order_policy="primary"
        )
        if sensitivity_formula != formula:
            _fit_gee(work, response, outcome, sensitivity_formula, family, grain, model_family, "demographic_sensitivity", rows, diagnostics, offset_col, True)


def _fit_gee(
    data: pd.DataFrame,
    response: str,
    reported_outcome: str,
    formula: str,
    family_name: str,
    grain: str,
    model_family: str,
    scope: str,
    rows: list[dict],
    diagnostics: list[dict],
    offset_col: str | None,
    exploratory: bool,
) -> None:
    needed = set(_formula_columns(formula)) | {response, "participant_id"}
    if offset_col:
        needed.add(offset_col)
    work = data.dropna(subset=[c for c in needed if c in data.columns]).copy()
    if offset_col:
        offset_raw = pd.to_numeric(work[offset_col], errors="coerce")
        work = work.loc[offset_raw > 0].copy()
        work["_log_offset"] = np.log(pd.to_numeric(work[offset_col], errors="coerce"))
    n_subjects = int(work["participant_id"].astype(str).nunique()) if "participant_id" in work else 0
    n_trials = int(work[KEYS].drop_duplicates().shape[0]) if set(KEYS).issubset(work.columns) else int(len(work))
    if len(work) < 12 or n_subjects < 4:
        diagnostics.append(_diag(grain, model_family, scope, reported_outcome, "insufficient_data", len(work), n_subjects, n_trials, formula=formula))
        return
    if family_name == "binomial" and work[response].nunique() < 2:
        diagnostics.append(_diag(grain, model_family, scope, reported_outcome, "no_binary_variation", len(work), n_subjects, n_trials, formula=formula))
        return
    families = {
        "binomial": sm.families.Binomial(),
        "negative_binomial": sm.families.NegativeBinomial(),
        "gamma": sm.families.Gamma(sm.families.links.Log()),
        "gaussian": sm.families.Gaussian(),
        "gaussian_logit": sm.families.Gaussian(),
    }
    try:
        model = smf.gee(
            formula, groups="participant_id", data=work,
            family=families[family_name],
            cov_struct=sm.cov_struct.Exchangeable(),
            offset=work["_log_offset"] if offset_col else None,
        )
        rank_deficient = (
            np.linalg.matrix_rank(np.asarray(model.exog, dtype=float))
            < model.exog.shape[1]
        )
        fit = model.fit()
        ci = fit.conf_int()
        numeric = [fit.params, fit.bse, fit.pvalues, ci.iloc[:, 0], ci.iloc[:, 1]]
        if any(not np.isfinite(pd.to_numeric(part, errors="coerce")).all() for part in numeric):
            raise FloatingPointError("nonfinite_covariance_or_estimate")
    except Exception as exc:
        status = "unstable_nonfinite_covariance" if isinstance(exc, FloatingPointError) else f"fit_failed:{type(exc).__name__}"
        diagnostics.append(_diag(grain, model_family, scope, reported_outcome, status, len(work), n_subjects, n_trials, formula=formula, error=str(exc)))
        return
    model_type = f"gee_{family_name}_participant_clustered"
    for term, estimate in fit.params.items():
        hypothesis_block = _hypothesis_block(
            reported_outcome, str(term), model_family=model_family, scope=scope
        )
        rows.append({
            "grain": grain, "family": model_family, "scope": scope,
            "hypothesis_block": hypothesis_block,
            "interpretation_tier": _term_tier(
                reported_outcome, str(term), model_family, scope, exploratory
            ),
            "outcome": reported_outcome,
            "term": str(term), "estimate": float(estimate),
            "std_error": float(fit.bse[term]),
            "effect_scale": _effect_scale(family_name),
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "p_value": float(fit.pvalues[term]), "n_obs": int(fit.nobs),
            "n_subjects": n_subjects, "n_trials": n_trials,
            "model_type": model_type, "formula": formula, "status": "fit",
        })
    fit_status = "fit_rank_deficient" if rank_deficient else "fit"
    diagnostics.append(_diag(grain, model_family, scope, reported_outcome, fit_status, int(fit.nobs), n_subjects, n_trials, model_type, formula))


def _formula(
    data: pd.DataFrame,
    response: str,
    aoi: bool,
    sensitivity: bool,
    order_policy: str = "primary",
) -> str:
    terms = []
    if _varying(data, "WWR") and _varying(data, "Complexity"):
        terms.append("C(WWR) * C(Complexity)")
    else:
        for name in ("WWR", "Complexity"):
            if _varying(data, name):
                terms.append(f"C({name})")
    if _varying(data, "ExperienceGroup"):
        interactions = [f"C(ExperienceGroup) * C({name})" for name in ("WWR", "Complexity") if _varying(data, name)]
        terms.extend(interactions or ["C(ExperienceGroup)"])
    if order_policy in {"primary", "carryover"}:
        terms.extend(name for name in ("block", "position") if _varying(data, name))
    elif order_policy == "time_stability" and _varying(data, "trial_index"):
        terms.append("trial_index")
        if _varying(data, "WWR"):
            terms.append("C(WWR):trial_index")
        if _varying(data, "Complexity"):
            terms.append("C(Complexity):trial_index")
    elif order_policy != "unadjusted":
        raise ValueError(f"Unknown order policy: {order_policy}")
    if order_policy == "carryover":
        if _varying(data, "previous_WWR"):
            terms.append("C(previous_WWR)")
        if _varying(data, "previous_Complexity"):
            terms.append("C(previous_Complexity)")
        if _varying(data, "order_scheme"):
            terms.append("C(order_scheme)")
        if _varying(data, "break_before_trial"):
            terms.append("break_before_trial")
    if aoi and _varying(data, "class_name"):
        terms.append("C(class_name)")
        if _varying(data, "WWR"):
            terms.append("C(WWR):C(class_name)")
    if sensitivity:
        for name in ("Gender",):
            if _varying(data, name):
                terms.append(f"C({name})")
        # RecruitmentBatch and the date-derived DateBatch are often the same
        # partition.  Never include both in one sensitivity model because that
        # produces an exactly singular design matrix.
        batch = "RecruitmentBatch" if _varying(data, "RecruitmentBatch") else "DateBatch" if _varying(data, "DateBatch") else None
        if batch:
            terms.append(f"C({batch})")
        if "Age" in data and pd.to_numeric(data["Age"], errors="coerce").notna().sum() > 0:
            terms.append("Age")
    return f"{response} ~ " + (" + ".join(dict.fromkeys(terms)) if terms else "1")


def _hypothesis_block(
    outcome: str,
    term: str,
    *,
    model_family: str,
    scope: str,
) -> str:
    exploration = {"aoi_transition_rate_per_min", "transition_entropy_normalized", "angular_scanpath_deg_per_s", "saccade_rate_per_min"}
    window = {"first_window_fixation_latency_ms", "window_entry_count", "window_directed_transition_rate_per_min"}
    fatigue = {"angular_scanpath_deg_per_s", "scanpath_length_px_per_s", "blink_rate_per_min", "blink_count", "pupil_post_early_delta_mm"}
    if scope == "primary_eye_available" and outcome in exploration and "Complexity" in term and "ExperienceGroup" not in term:
        return "H1_complexity_exploration"
    if scope == "primary_eye_available" and outcome in window and "WWR" in term and "ExperienceGroup" not in term:
        return "H2_wwr_window_direction"
    if scope == "primary_eye_available" and outcome in exploration | window and "ExperienceGroup" in term and ":" in term:
        return "H3_experience_moderation"
    if scope in PRIMARY_SCOPES and term in {"block", "position"}:
        if model_family == "eeg" and outcome in EEG_ORDER_PRIMARY:
            return "H4_eeg_theta_alpha_order"
        if model_family == "questionnaire":
            return "H4_questionnaire_order"
        if model_family.startswith("eye_") and outcome in fatigue:
            return "H4_eye_fatigue_order"
    if scope == "order_time_stability_sensitivity" and "trial_index" in term:
        if "WWR" in term or "Complexity" in term:
            return f"H4_{_modality_label(model_family)}_time_stability"
    if scope == "carryover_sensitivity" and (
        "previous_WWR" in term
        or "previous_Complexity" in term
        or term == "break_before_trial"
    ):
        return f"H4_{_modality_label(model_family)}_carryover"
    return "exploratory_not_in_primary_fdr"


def _modality_label(model_family: str) -> str:
    if model_family == "questionnaire":
        return "questionnaire"
    if model_family == "eeg":
        return "eeg"
    return "eye"


def _term_tier(
    outcome: str,
    term: str,
    model_family: str,
    scope: str,
    exploratory: bool,
) -> str:
    fatigue = {
        "angular_scanpath_deg_per_s", "blink_count", "blink_rate_per_min",
        "pupil_post_early_delta_mm",
    }
    if (
        scope == "primary_eye_available"
        and model_family.startswith("eye_")
        and term in {"block", "position"}
        and outcome in fatigue
    ):
        return "primary"
    if exploratory or scope not in PRIMARY_SCOPES:
        return "exploratory"
    if (
        model_family == "eeg"
        and term in {"block", "position"}
        and outcome not in EEG_ORDER_PRIMARY
    ):
        return "exploratory"
    return "primary"


def _apply_fdr(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame(columns=MODEL_COLUMNS)
    out = models.copy()
    out["p_fdr_bh"] = np.nan
    primary = ~out["hypothesis_block"].eq("exploratory_not_in_primary_fdr")
    for _, idx in out.loc[primary].groupby("hypothesis_block").groups.items():
        out.loc[idx, "p_fdr_bh"] = _bh(out.loc[idx, "p_value"])
    out["p_fdr_bh_family"] = out["p_fdr_bh"]
    out["p_fdr_bh_all"] = _bh(out["p_value"])
    out["significant_fdr_bh_0_05"] = out["p_fdr_bh"].lt(0.05)
    out["significant_fdr_bh_family_0_05"] = out["p_fdr_bh_family"].lt(0.05)
    return out


def metric_sample_summary(*frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    frame_specs = [
        ("questionnaire_scene", frames[0], QUESTIONNAIRE_OUTCOMES),
        ("eye_aoi", frames[1], ["visited", "FC", "FCR", "TFD_ms", "attention_share", "median_revisit_latency_ms"]),
        ("eye_scene_trial", frames[2], PRIMARY_EYE_TRIAL),
        ("eeg_scene", frames[3], [c for c in frames[3].columns if c.startswith("eeg_")]),
    ]
    for grain, frame, metrics in frame_specs:
        for metric in metrics:
            if metric not in frame:
                continue
            value = pd.to_numeric(frame[metric], errors="coerce")
            valid = value.notna()
            n = int(valid.sum())
            mean = float(value.loc[valid].mean()) if n else np.nan
            se = float(value.loc[valid].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            condition_cols = [c for c in ("WWR", "Complexity", "class_name") if c in frame]
            rows.append({
                "grain": grain, "metric": metric, "n_rows": n,
                "n_subjects": int(frame.loc[valid, "participant_id"].astype(str).nunique()) if "participant_id" in frame else 0,
                "n_trials": int(frame.loc[valid, KEYS].drop_duplicates().shape[0]) if set(KEYS).issubset(frame.columns) else n,
                "valid_condition_cells": int(frame.loc[valid, condition_cols].drop_duplicates().shape[0]) if condition_cols else 1,
                "missing_rate": float(1 - valid.mean()) if len(valid) else np.nan,
                "mean": mean, "ci95_low": mean - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_high": mean + 1.96 * se if np.isfinite(se) else np.nan,
            })
    return pd.DataFrame(rows)


def modality_sample_flow(
    questionnaire: pd.DataFrame,
    eye_aoi: pd.DataFrame,
    eye_dynamic: pd.DataFrame,
    eye_qc: pd.DataFrame,
    eeg: pd.DataFrame,
    trimodal_qc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for modality, frame, policy in [
        ("questionnaire", questionnaire, "questionnaire_available"),
        ("eye", eye_dynamic if not eye_dynamic.empty else eye_aoi, "eye_metric_specific_available_no_eeg_filter"),
        ("eeg", eeg, "eeg_qc_passed"),
    ]:
        rows.append(_sample_row(modality, policy, frame))
    eye_trials = set(map(tuple, eye_dynamic[KEYS].drop_duplicates().to_numpy())) if set(KEYS).issubset(eye_dynamic.columns) else set()
    eeg_trials = set(map(tuple, eeg[KEYS].drop_duplicates().to_numpy())) if set(KEYS).issubset(eeg.columns) else set()
    q_trials = set(map(tuple, questionnaire[KEYS].drop_duplicates().to_numpy())) if set(KEYS).issubset(questionnaire.columns) else set()
    intersection = eye_trials & eeg_trials & q_trials
    if trimodal_qc is not None and not trimodal_qc.empty and set(KEYS).issubset(trimodal_qc.columns):
        if "excluded_from_analysis" in trimodal_qc:
            keep = ~_truthy(trimodal_qc["excluded_from_analysis"])
            qc_kept = set(
                map(
                    tuple,
                    trimodal_qc.loc[keep, KEYS].drop_duplicates().to_numpy(),
                )
            )
            intersection &= qc_kept
    rows.append({"modality": "trimodal_intersection", "eligibility_policy": "descriptive_alignment_only", "n_subjects": len({x[0] for x in intersection}), "n_trials": len(intersection)})
    if not eye_qc.empty and "analysis_valid_ratio" in eye_qc:
        ratio = pd.to_numeric(eye_qc["analysis_valid_ratio"], errors="coerce")
        for threshold in (0.5, 0.6, 0.7, 0.8):
            rows.append(_sample_row("eye", f"sensitivity_valid_coordinates_ge_{threshold:.1f}", eye_qc.loc[ratio.ge(threshold)]))
    return pd.DataFrame(rows)


def monte_carlo_mde(data: pd.DataFrame, outcomes: list[str], simulations: int = 1000, seed: int = 20260713, alpha: float = 0.05, target_power: float = 0.8) -> pd.DataFrame:
    """Repeated-measures Monte Carlo MDE for the binary Complexity contrast.

    The simulation operates on each participant's condition means, preserving
    the empirical number of repeated observations.  It reports precision and
    does not label observed results publishable or unpublishable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    if not {"participant_id", "Complexity"}.issubset(data.columns):
        return pd.DataFrame()
    for outcome in outcomes:
        if outcome not in data:
            continue
        work = data[["participant_id", "Complexity", outcome]].copy()
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        work["Complexity"] = pd.to_numeric(work["Complexity"], errors="coerce")
        means = work.dropna().groupby(["participant_id", "Complexity"])[outcome].mean().unstack()
        if not {0, 1}.issubset(means.columns):
            rows.append({"metric": outcome, "status": "missing_complexity_pair", "simulations": simulations, "seed": seed})
            continue
        paired = means[[0, 1]].dropna()
        diffs = (paired[1] - paired[0]).to_numpy(dtype=float)
        n = len(diffs)
        if n < 4:
            rows.append({"metric": outcome, "status": "insufficient_paired_subjects", "n_subjects": n, "simulations": simulations, "seed": seed})
            continue
        sigma = float(np.std(diffs, ddof=1))
        raw_scale = float(pd.to_numeric(work[outcome], errors="coerce").std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            rows.append({"metric": outcome, "status": "zero_or_invalid_variance", "n_subjects": n, "simulations": simulations, "seed": seed})
            continue
        grid = np.linspace(0, 2.5 * sigma, 101)
        powers = []
        for effect in grid:
            sim = rng.normal(effect, sigma, size=(simulations, n))
            se = sim.std(axis=1, ddof=1) / np.sqrt(n)
            reject = np.abs(sim.mean(axis=1) / se) > 1.96
            powers.append(float(np.mean(reject)))
        eligible = np.flatnonzero(np.asarray(powers) >= target_power)
        mde = float(grid[eligible[0]]) if len(eligible) else np.nan
        precision = 1.96 * sigma / np.sqrt(n)
        rows.append({
            "metric": outcome, "contrast": "Complexity_1_minus_0", "status": "estimated",
            "n_subjects": n, "simulations": simulations, "alpha_two_sided": alpha,
            "target_power": target_power, "seed": seed, "mde_raw_units": mde,
            "mde_standardized": mde / raw_scale if raw_scale > 0 else np.nan,
            "estimated_precision_half_width_raw": precision,
            "note": "Monte Carlo design sensitivity; not retrospective observed power.",
        })
    return pd.DataFrame(rows)


def _apply_eeg_qc(eeg: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    if qc.empty or not set(KEYS).issubset(qc.columns):
        return eeg
    fields = KEYS + [c for c in ("bad_eeg_quality", "eeg_subject_quality_exclusion") if c in qc]
    merged = eeg.merge(qc[fields].drop_duplicates(KEYS), on=KEYS, how="left", suffixes=("", "_qc"))
    bad = pd.Series(False, index=merged.index)
    for col in fields[2:]:
        bad |= _truthy(merged[col])
    return merged.loc[~bad].copy()


def _attach_participants(data: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "participant_id" not in data or "participant_id" not in participants:
        return data
    demographic = [c for c in ("participant_id", "ExperienceGroup", "Gender", "Age", "RecruitmentBatch", "DateBatch") if c in participants]
    base = data.drop(columns=[c for c in demographic if c != "participant_id" and c in data], errors="ignore")
    return base.merge(participants[demographic].drop_duplicates("participant_id"), on="participant_id", how="left")


def _normalize_questionnaire(data: pd.DataFrame) -> pd.DataFrame:
    """Expose prepared S1-S5 columns under the canonical q_S1-q_S5 names."""
    out = data.copy()
    for index in range(1, 6):
        canonical = f"q_S{index}"
        source = f"S{index}"
        if canonical not in out and source in out:
            out[canonical] = out[source]
    return out


def _normalize_eeg_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize ROI-band EEG columns and reject conflicting duplicates."""
    out = data.copy()
    for canonical in EEG_OUTCOMES:
        source = canonical.removeprefix("eeg_")
        has_source = source in out.columns
        has_canonical = canonical in out.columns
        if has_source and has_canonical:
            left = pd.to_numeric(out[source], errors="coerce")
            right = pd.to_numeric(out[canonical], errors="coerce")
            equal = (left.isna() & right.isna()) | np.isclose(
                left, right, equal_nan=True
            )
            if not bool(np.asarray(equal).all()):
                raise ValueError(
                    f"Conflicting EEG columns {source!r} and {canonical!r}"
                )
            out = out.drop(columns=[source])
        elif has_source:
            out = out.rename(columns={source: canonical})
    return out


def _attach_scene_design(data: pd.DataFrame, scene: pd.DataFrame) -> pd.DataFrame:
    if data.empty or scene.empty or not set(KEYS).issubset(data.columns) or not set(KEYS).issubset(scene.columns):
        return data
    design_cols = [
        c
        for c in (
            "order_scheme",
            "participant_order",
            "block",
            "position",
            "round",
            "WWR",
            "Complexity",
            "condition_id",
        )
        if c in scene.columns
    ]
    missing = [c for c in design_cols if c not in data.columns]
    if not missing:
        return data
    return data.merge(
        scene[KEYS + missing].drop_duplicates(KEYS),
        on=KEYS,
        how="left",
        validate="many_to_one",
    )


def _add_order_features(
    data: pd.DataFrame,
    trial_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if data.empty or not set(KEYS).issubset(data.columns):
        return data
    out = data.copy()
    scene_id = pd.to_numeric(out["scene_id"], errors="coerce")
    if "block" in out and "position" in out:
        block = pd.to_numeric(out["block"], errors="coerce")
        position = pd.to_numeric(out["position"], errors="coerce")
        # The production experiment has six positions in each block. Small
        # synthetic fixtures may use fewer; only enforce the registered
        # 12-scene contract when the six-position design is present.
        if bool(position.eq(6).any()) or bool(scene_id.eq(12).any()):
            derived = (block - 1) * 6 + position
            mismatch = scene_id.notna() & derived.notna() & ~np.isclose(scene_id, derived)
            if bool(mismatch.any()):
                raise ValueError("scene_id does not match (block - 1) * 6 + position")
    out["trial_index"] = scene_id
    reference = (
        trial_reference.copy()
        if trial_reference is not None
        and not trial_reference.empty
        and set(KEYS).issubset(trial_reference.columns)
        else out
    )
    trial_cols = KEYS + [
        c
        for c in ("WWR", "Complexity", "block", "position", "order_scheme")
        if c in reference.columns
    ]
    trials = (
        reference[trial_cols]
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
        .copy()
    )
    trials["previous_WWR"] = trials.groupby("participant_id", sort=False)["WWR"].shift(1) if "WWR" in trials else np.nan
    trials["previous_Complexity"] = trials.groupby("participant_id", sort=False)["Complexity"].shift(1) if "Complexity" in trials else np.nan
    if "block" in trials and "position" in trials:
        trials["break_before_trial"] = (
            pd.to_numeric(trials["block"], errors="coerce").eq(2)
            & pd.to_numeric(trials["position"], errors="coerce").eq(1)
        ).astype(int)
    else:
        trials["break_before_trial"] = 0
    feature_cols = KEYS + [
        "previous_WWR",
        "previous_Complexity",
        "break_before_trial",
    ]
    out = out.drop(
        columns=[c for c in feature_cols[2:] if c in out.columns],
        errors="ignore",
    )
    return out.merge(
        trials[feature_cols],
        on=KEYS,
        how="left",
        validate="many_to_one",
    )


def _eye_threshold_subset(data: pd.DataFrame, qc: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if data.empty or qc.empty or "analysis_valid_ratio" not in qc or not set(KEYS).issubset(qc.columns):
        return data.iloc[0:0].copy()
    keep = qc.loc[pd.to_numeric(qc["analysis_valid_ratio"], errors="coerce").ge(threshold), KEYS].drop_duplicates()
    return data.merge(keep, on=KEYS, how="inner")


def _sample_row(modality: str, policy: str, frame: pd.DataFrame) -> dict:
    return {
        "modality": modality, "eligibility_policy": policy,
        "n_subjects": int(frame["participant_id"].astype(str).nunique()) if "participant_id" in frame else 0,
        "n_trials": int(frame[KEYS].drop_duplicates().shape[0]) if set(KEYS).issubset(frame.columns) else int(len(frame)),
    }


def _diag(grain: str, family: str, scope: str, outcome: str, status: str, n_obs: int = 0, n_subjects: int = 0, n_trials: int = 0, model_type: str = "", formula: str = "", error: str = "") -> dict:
    return {"grain": grain, "family": family, "scope": scope, "outcome": outcome, "status": status, "n_obs": int(n_obs), "n_subjects": int(n_subjects), "n_trials": int(n_trials), "model_type": model_type, "formula": formula, "error": error}


def _formula_columns(formula: str) -> list[str]:
    categorical = re.findall(r"C\(([^)]+)\)", formula)
    plain = re.findall(
        r"\b(?:block|position|trial_index|break_before_trial|Age|participant_id|_model_outcome)\b",
        formula,
    )
    response = formula.split("~", 1)[0].strip()
    return list(dict.fromkeys([response, *categorical, *plain]))


def _varying(data: pd.DataFrame, column: str) -> bool:
    return column in data and data[column].dropna().nunique() > 1


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "visited"})


def _effect_scale(family: str) -> str:
    if family == "binomial":
        return "log_odds"
    if family == "negative_binomial":
        return "log_rate_or_count_mean"
    if family == "gamma":
        return "log_mean"
    if family == "gaussian_logit":
        return "logit_transformed_share"
    return "outcome_units"


def _bh(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = (valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result
