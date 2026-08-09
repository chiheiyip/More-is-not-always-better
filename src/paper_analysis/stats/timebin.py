"""Co-primary models for clock-synchronized eye-tracking and EEG time bins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from paper_analysis.utils.io import read_table, write_table


EEG_OUTCOMES = [
    f"eeg_{roi}_{band}"
    for roi in ("F", "P", "O")
    for band in ("theta", "alpha", "beta")
]
EYE_OUTCOMES = ["visited", "TFD_ms", "attention_share", "FCR"]
KEYS = ["participant_id", "scene_id"]
PARALLEL_HYPOTHESES = {
    "time": "H_time_overall_change",
    "wwr_time": "H_time_wwr_trajectory",
    "complexity_time": "H_time_complexity_trajectory",
}
MODEL_COLUMNS = [
    "grain", "family", "scope", "hypothesis_block", "interpretation_tier",
    "outcome", "term", "estimate", "std_error", "effect_scale", "ci_low",
    "ci_high", "p_value", "p_fdr_bh", "p_fdr_bh_family",
    "p_fdr_bh_parallel", "p_fdr_bh_all",
    "significant_fdr_bh_0_05", "significant_fdr_bh_family_0_05", "n_obs",
    "n_subjects", "n_trials", "model_type", "formula", "status",
    "analysis_resolution", "analysis_status", "hypothesis_family",
    "clock_qc_policy",
    "onset_trim_s", "time_axis",
]


def run_timebin_models(
    synchronized_timebin_csv: str | Path,
    clock_scene_qc_csv: str | Path,
    outdir: str | Path = "outputs/06_models",
    scene_model_results_csv: str | Path | None = None,
    onset_trim_filter_s: float | None = None,
    expected_onset_trims_s: tuple[float, ...] | list[float] | None = None,
    time_column: str | None = None,
    require_common_trials: bool = True,
) -> dict[str, Path]:
    """Fit participant-clustered GEE models at synchronized 2-second resolution."""
    timebins = read_table(synchronized_timebin_csv)
    if onset_trim_filter_s is not None:
        if "onset_trim_s" not in timebins:
            raise ValueError("Synchronized time-bin table lacks onset_trim_s")
        timebins = timebins.loc[
            np.isclose(
                pd.to_numeric(timebins["onset_trim_s"], errors="coerce"),
                float(onset_trim_filter_s),
            )
        ].copy()
    qc = read_table(clock_scene_qc_csv)
    work = _eligible_timebins(timebins, qc)
    time_column = time_column or (
        "scene_time_norm" if "scene_time_norm" in work.columns else "time_norm"
    )
    if time_column not in work:
        raise ValueError(f"Synchronized time-bin table lacks {time_column}")
    expected = (
        tuple(sorted(float(value) for value in expected_onset_trims_s))
        if expected_onset_trims_s is not None else None
    )
    if expected is not None:
        observed = tuple(sorted(
            pd.to_numeric(work.get("onset_trim_s"), errors="coerce")
            .dropna().unique().astype(float).tolist()
        ))
        if observed != expected:
            raise ValueError(
                f"Expected parallel onset trims {list(expected)}; observed {list(observed)}"
            )
    work, sample_flow = _parallel_common_trial_filter(
        work,
        expected_onset_trims_s=expected,
        enabled=require_common_trials,
    )
    rows: list[dict] = []
    diagnostics: list[dict] = []

    grouped = (
        work.groupby("onset_trim_s", dropna=False)
        if "onset_trim_s" in work.columns
        else [(np.nan, work)]
    )
    for onset_trim_s, window in grouped:
        row_start = len(rows)
        diagnostic_start = len(diagnostics)
        eeg = window.drop_duplicates(KEYS + ["bin_index"]).copy()
        for outcome in EEG_OUTCOMES:
            _fit_outcome(
                eeg, outcome, "eeg", rows, diagnostics,
                time_column=time_column,
            )
        for outcome in EYE_OUTCOMES:
            _fit_outcome(
                window, outcome, "eye", rows, diagnostics, aoi=True,
                time_column=time_column,
            )
        for row in rows[row_start:]:
            row["onset_trim_s"] = onset_trim_s
        for row in diagnostics[diagnostic_start:]:
            row["onset_trim_s"] = onset_trim_s

    models = _apply_fdr(pd.DataFrame(rows)).reindex(columns=MODEL_COLUMNS)
    summaries = temporal_scene_summaries(work, time_column=time_column)
    stability = build_parallel_window_stability(models, expected)
    multiscale = build_multiscale_claim_support(
        read_table(scene_model_results_csv) if scene_model_results_csv else pd.DataFrame(),
        models,
    )
    outdir = Path(outdir)
    outputs = {
        "timebin_model_results": write_table(models, outdir / "timebin_model_results.csv"),
        "timebin_model_diagnostics": write_table(pd.DataFrame(diagnostics), outdir / "timebin_model_diagnostics.csv"),
        "temporal_scene_summaries": write_table(summaries, outdir / "temporal_scene_summaries.csv"),
        "parallel_window_sample_flow": write_table(
            sample_flow, outdir / "parallel_window_sample_flow.csv"
        ),
        "parallel_window_stability": write_table(
            stability, outdir / "parallel_window_stability.csv"
        ),
        "multiscale_claim_support": write_table(multiscale, outdir / "multiscale_claim_support.csv"),
    }
    if scene_model_results_csv:
        combined = combine_coprimary_model_results(read_table(scene_model_results_csv), models)
        outputs["model_results"] = write_table(combined, Path(scene_model_results_csv))
    return outputs


def _eligible_timebins(timebins: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "bin_index", "time_norm", "eeg_window_coverage"}
    missing = required - set(timebins.columns)
    if missing:
        raise ValueError(f"Synchronized time-bin table missing {sorted(missing)}")
    if not {*KEYS, "clock_alignment_pass"}.issubset(qc.columns):
        raise ValueError("Clock scene QC must contain participant_id, scene_id, clock_alignment_pass")
    qc_work = qc.copy()
    qc_work["_passed"] = _truthy(qc_work["clock_alignment_pass"])
    # Alignment eligibility is a Participant × Trial contract. A participant
    # does not need all 12 scenes to pass in order for their successfully
    # aligned scenes to contribute. Requiring complete clock QC here would
    # discard valid synchronized trials after the formal exact-intersection
    # table has already applied scene-level eye, EEG and clock QC.
    passed = qc_work.loc[qc_work["_passed"], KEYS].drop_duplicates()
    out = timebins.merge(passed, on=KEYS, how="inner")
    coverage = pd.to_numeric(out["eeg_window_coverage"], errors="coerce")
    return out.loc[coverage.ge(0.95)].copy()


def _parallel_common_trial_filter(
    data: pd.DataFrame,
    *,
    expected_onset_trims_s: tuple[float, ...] | None,
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "onset_trim_s" not in data or not enabled:
        return data.copy(), pd.DataFrame()
    work = data.copy()
    work["onset_trim_s"] = pd.to_numeric(work["onset_trim_s"], errors="coerce")
    observed = tuple(sorted(work["onset_trim_s"].dropna().unique().astype(float)))
    expected = expected_onset_trims_s or observed
    trial_windows = work[KEYS + ["onset_trim_s"]].drop_duplicates()
    counts = trial_windows.groupby(KEYS)["onset_trim_s"].nunique()
    common_keys = counts.loc[counts.eq(len(expected))].index.to_frame(index=False)
    filtered = work.merge(common_keys, on=KEYS, how="inner", validate="many_to_one")
    rows = []
    for trim in expected:
        before = work.loc[np.isclose(work["onset_trim_s"], trim)]
        after = filtered.loc[np.isclose(filtered["onset_trim_s"], trim)]
        rows.append({
            "onset_trim_s": trim,
            "participants_before_common_filter": int(before["participant_id"].nunique()),
            "trials_before_common_filter": int(before[KEYS].drop_duplicates().shape[0]),
            "timebin_rows_before_common_filter": int(len(before)),
            "participants_common": int(after["participant_id"].nunique()),
            "trials_common": int(after[KEYS].drop_duplicates().shape[0]),
            "timebin_rows_common": int(len(after)),
            "common_trial_rule": "present_in_every_parallel_onset_window",
        })
    return filtered, pd.DataFrame(rows)


def _fit_outcome(
    data: pd.DataFrame,
    outcome: str,
    modality: str,
    rows: list[dict],
    diagnostics: list[dict],
    *,
    aoi: bool = False,
    time_column: str = "time_norm",
) -> None:
    if outcome not in data.columns:
        diagnostics.append(_diag(outcome, modality, "missing_outcome"))
        return
    work = data.copy()
    if outcome == "visited":
        work[outcome] = _truthy(work[outcome]).astype(int)
        family = sm.families.Binomial()
        family_name = "binomial"
    else:
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        family = sm.families.Gaussian()
        family_name = "gaussian"
    formula = _timebin_formula(
        work, outcome, aoi=aoi, time_column=time_column
    )
    required = [outcome, "participant_id", time_column]
    work = work.dropna(subset=required).copy()
    n_subjects = int(work["participant_id"].astype(str).nunique())
    n_trials = int(work[KEYS].drop_duplicates().shape[0])
    if len(work) < 24 or n_subjects < 4:
        diagnostics.append(_diag(
            outcome, modality, "insufficient_data", len(work), n_subjects,
            n_trials, formula=formula,
        ))
        return
    if outcome == "visited" and work[outcome].nunique() < 2:
        diagnostics.append(_diag(
            outcome, modality, "no_binary_variation", len(work), n_subjects,
            n_trials, formula=formula,
        ))
        return
    try:
        model = smf.gee(
            formula,
            groups="participant_id",
            data=work,
            family=family,
            cov_struct=sm.cov_struct.Independence(),
        )
        fit = model.fit(cov_type="robust")
        ci = fit.conf_int()
        numeric = [fit.params, fit.bse, fit.pvalues, ci.iloc[:, 0], ci.iloc[:, 1]]
        if any(not np.isfinite(pd.to_numeric(part, errors="coerce")).all() for part in numeric):
            raise FloatingPointError("nonfinite_covariance_or_estimate")
    except Exception as exc:
        diagnostics.append(_diag(
            outcome, modality, f"fit_failed:{type(exc).__name__}", len(work),
            n_subjects, n_trials, formula=formula, error=str(exc),
        ))
        return

    for term, estimate in fit.params.items():
        normalized = _normalize_time_term(str(term), time_column=time_column)
        family_label = _parallel_hypothesis(normalized, time_column=time_column)
        rows.append({
            "grain": "synchronized_2s_timebin",
            "family": f"{modality}_synchronized_timebin",
            "scope": "clock_qc_passed",
            "hypothesis_block": family_label,
            "interpretation_tier": (
                "parallel" if family_label in PARALLEL_HYPOTHESES.values()
                else "adjustment_covariate"
            ),
            "outcome": outcome,
            "term": str(term),
            "estimate": float(estimate),
            "std_error": float(fit.bse[term]),
            "effect_scale": "log_odds" if family_name == "binomial" else "outcome_units",
            "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]),
            "p_value": float(fit.pvalues[term]),
            "n_obs": int(fit.nobs),
            "n_subjects": n_subjects,
            "n_trials": n_trials,
            "model_type": f"gee_{family_name}_participant_clustered_independent_robust",
            "formula": formula,
            "status": "fit",
            "analysis_resolution": "synchronized_timebin",
            "analysis_status": "parallel",
            "hypothesis_family": family_label,
            "clock_qc_policy": "scene_level_monotonic_coverage95_match95_p95delta2ms",
            "time_axis": time_column,
        })
    diagnostics.append(_diag(
        outcome, modality, "fit", int(fit.nobs), n_subjects, n_trials,
        model_type=f"gee_{family_name}_participant_clustered_independent_robust",
        formula=formula,
    ))


def _timebin_formula(
    data: pd.DataFrame,
    outcome: str,
    *,
    aoi: bool,
    time_column: str = "time_norm",
) -> str:
    terms: list[str] = []
    for name in ("WWR", "Complexity", "ExperienceGroup"):
        if _varying(data, name):
            terms.append(f"C({name})")
    for name in ("block", "position"):
        if _varying(data, name):
            terms.append(name)
    terms.append(time_column)
    if _varying(data, "WWR"):
        terms.append(f"C(WWR):{time_column}")
    if _varying(data, "Complexity"):
        terms.append(f"C(Complexity):{time_column}")
    if aoi and _varying(data, "class_name"):
        terms.append("C(class_name)")
    return f"{outcome} ~ " + " + ".join(terms)


def temporal_scene_summaries(
    data: pd.DataFrame,
    *,
    time_column: str = "time_norm",
) -> pd.DataFrame:
    rows: list[dict] = []
    onset = ["onset_trim_s"] if "onset_trim_s" in data.columns else []
    eeg = data.drop_duplicates(KEYS + onset + ["bin_index"]).copy()
    specs = [(eeg, outcome, "eeg", []) for outcome in EEG_OUTCOMES if outcome in eeg.columns]
    specs += [
        (data, outcome, "eye", ["class_name"] if "class_name" in data.columns else [])
        for outcome in EYE_OUTCOMES if outcome in data.columns
    ]
    for frame, outcome, modality, extra in specs:
        value = _truthy(frame[outcome]).astype(float) if outcome == "visited" else pd.to_numeric(frame[outcome], errors="coerce")
        work = frame[KEYS + onset + [time_column, *extra]].copy()
        work["value"] = value
        work["phase"] = pd.cut(
            pd.to_numeric(work[time_column], errors="coerce"),
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["early", "middle", "late"],
            right=False,
        )
        grouping = KEYS + onset + extra
        for key, sub in work.groupby(grouping, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            base = dict(zip(grouping, key_tuple))
            phase = sub.groupby("phase", observed=True)["value"].mean()
            valid = sub.dropna(subset=[time_column, "value"])
            slope = (
                float(np.polyfit(valid[time_column].astype(float), valid["value"].astype(float), 1)[0])
                if len(valid) >= 2 and valid[time_column].nunique() >= 2 else np.nan
            )
            rows.append({
                **base,
                "modality": modality,
                "outcome": outcome,
                "early_mean": phase.get("early", np.nan),
                "middle_mean": phase.get("middle", np.nan),
                "late_mean": phase.get("late", np.nan),
                "late_minus_early": phase.get("late", np.nan) - phase.get("early", np.nan),
                "scene_slope_per_time_norm": slope,
                "time_axis": time_column,
                "analysis_resolution": "synchronized_timebin",
                "analysis_status": "parallel_bridge_summary",
            })
    return pd.DataFrame(rows)


def combine_coprimary_model_results(scene: pd.DataFrame, timebin: pd.DataFrame) -> pd.DataFrame:
    scene = scene.copy()
    scene["analysis_resolution"] = scene.get("analysis_resolution", "scene_level")
    scene["analysis_status"] = scene.get("analysis_status", np.where(
        scene.get("interpretation_tier", pd.Series("", index=scene.index)).eq("primary"),
        "primary", "supporting",
    ))
    scene["hypothesis_family"] = scene.get(
        "hypothesis_family",
        scene.get("hypothesis_block", "scene_level"),
    )
    scene["clock_qc_policy"] = scene.get("clock_qc_policy", "not_applicable_scene_level")
    columns = list(dict.fromkeys([*scene.columns, *timebin.columns]))
    return pd.concat(
        [scene.reindex(columns=columns), timebin.reindex(columns=columns)],
        ignore_index=True,
    )


def build_multiscale_claim_support(scene: pd.DataFrame, timebin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for resolution, models in (("scene_level", scene), ("synchronized_timebin", timebin)):
        if models.empty:
            continue
        q = pd.to_numeric(
            models.get("p_fdr_bh_family", models.get("p_fdr_bh", pd.Series(np.nan, index=models.index))),
            errors="coerce",
        )
        for outcome, sub in models.assign(_q=q).groupby("outcome", dropna=False):
            rows.append({
                "analysis_resolution": resolution,
                "analysis_status": (
                    "parallel" if resolution == "synchronized_timebin"
                    else "primary"
                ),
                "outcome": outcome,
                "tested_terms": int(len(sub)),
                "fdr_significant_terms": int(sub["_q"].lt(0.05).sum()),
                "support_type": "overall_level" if resolution == "scene_level" else "time_dynamic",
                "interpretation_rule": "convergent_strengthens_evidence;discordant_is_scale_dependent",
            })
    return pd.DataFrame(rows)


def _apply_fdr(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty:
        return models
    out = models.copy()
    out["p_fdr_bh"] = np.nan
    parallel = out["hypothesis_family"].isin(PARALLEL_HYPOTHESES.values())
    window_groups = ["hypothesis_family"]
    if "onset_trim_s" in out:
        window_groups.insert(0, "onset_trim_s")
    for _, index in out.loc[parallel].groupby(window_groups).groups.items():
        out.loc[index, "p_fdr_bh"] = _bh(out.loc[index, "p_value"])
    out["p_fdr_bh_family"] = out["p_fdr_bh"]
    out["p_fdr_bh_parallel"] = np.nan
    for _, index in out.loc[parallel].groupby("hypothesis_family").groups.items():
        out.loc[index, "p_fdr_bh_parallel"] = _bh(out.loc[index, "p_value"])
    out["p_fdr_bh_all"] = _bh(out.loc[parallel, "p_value"]).reindex(out.index)
    out["significant_fdr_bh_0_05"] = out["p_fdr_bh"].lt(0.05)
    out["significant_fdr_bh_family_0_05"] = out["p_fdr_bh_family"].lt(0.05)
    return out


def build_parallel_window_stability(
    models: pd.DataFrame,
    expected_onset_trims_s: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    if models.empty or "onset_trim_s" not in models:
        return pd.DataFrame()
    work = models.loc[
        models["hypothesis_family"].isin(PARALLEL_HYPOTHESES.values())
    ].copy()
    if work.empty:
        return pd.DataFrame()
    expected = expected_onset_trims_s or tuple(sorted(
        pd.to_numeric(work["onset_trim_s"], errors="coerce")
        .dropna().unique().astype(float)
    ))
    rows = []
    grouping = ["family", "outcome", "term", "hypothesis_family"]
    for key, sub in work.groupby(grouping, dropna=False):
        estimates = pd.to_numeric(sub["estimate"], errors="coerce")
        signs = set(np.sign(estimates.dropna()).astype(int)) - {0}
        windows = set(
            pd.to_numeric(sub["onset_trim_s"], errors="coerce")
            .dropna().astype(float)
        )
        all_windows = set(expected).issubset(windows)
        direction_consistent = all_windows and len(signs) == 1
        window_sig = pd.to_numeric(
            sub["p_fdr_bh_family"], errors="coerce"
        ).lt(0.05)
        parallel_sig = pd.to_numeric(
            sub["p_fdr_bh_parallel"], errors="coerce"
        ).lt(0.05)
        if not all_windows:
            classification = "not_estimable_all_windows"
        elif direction_consistent and parallel_sig.all():
            classification = "parallel_stable_evidence"
        elif window_sig.any():
            classification = "window_specific_evidence"
        elif direction_consistent:
            classification = "direction_consistent_insufficient_evidence"
        else:
            classification = "window_unstable"
        rows.append({
            **dict(zip(grouping, key if isinstance(key, tuple) else (key,))),
            "expected_onset_trims_s": ",".join(f"{value:g}" for value in expected),
            "windows_estimated": int(len(windows)),
            "direction_consistent": bool(direction_consistent),
            "window_significant_count": int(window_sig.sum()),
            "parallel_significant_count": int(parallel_sig.sum()),
            "estimate_min": float(estimates.min()) if estimates.notna().any() else np.nan,
            "estimate_max": float(estimates.max()) if estimates.notna().any() else np.nan,
            "classification": classification,
        })
    return pd.DataFrame(rows)


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def _normalize_time_term(term: str, *, time_column: str = "time_norm") -> str:
    if term == time_column:
        return term
    if time_column not in term:
        return term
    if "WWR" in term:
        return f"C(WWR):{time_column}"
    if "Complexity" in term:
        return f"C(Complexity):{time_column}"
    return term


def _parallel_hypothesis(term: str, *, time_column: str) -> str:
    if term == time_column:
        return PARALLEL_HYPOTHESES["time"]
    if term == f"C(WWR):{time_column}":
        return PARALLEL_HYPOTHESES["wwr_time"]
    if term == f"C(Complexity):{time_column}":
        return PARALLEL_HYPOTHESES["complexity_time"]
    return "model_covariate_not_fdr"


def _varying(data: pd.DataFrame, column: str) -> bool:
    return column in data.columns and data[column].dropna().nunique() > 1


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _diag(
    outcome: str,
    modality: str,
    status: str,
    n_obs: int = 0,
    n_subjects: int = 0,
    n_trials: int = 0,
    *,
    model_type: str = "gee_participant_clustered_independent_robust",
    formula: str = "",
    error: str = "",
) -> dict:
    return {
        "grain": "synchronized_2s_timebin",
        "family": f"{modality}_synchronized_timebin",
        "scope": "clock_qc_passed",
        "outcome": outcome,
        "status": status,
        "n_obs": n_obs,
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "model_type": model_type,
        "formula": formula,
        "error": error,
        "analysis_resolution": "synchronized_timebin",
        "analysis_status": "parallel",
        "clock_qc_policy": "scene_level_monotonic_coverage95_match95_p95delta2ms",
    }
