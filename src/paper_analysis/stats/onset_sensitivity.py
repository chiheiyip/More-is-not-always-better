"""Scene-onset window sensitivity and equivalence analysis for EEG.

This module consumes already re-extracted waveform metrics.  It cannot and does
not trim scene-level EEG values.  Formal equivalence is limited to the paired
5/10/15-s common-QC set and the predeclared 10-vs-15-s comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from paper_analysis.eeg.onset import load_eeg_analysis_config
from paper_analysis.utils.io import read_table, write_table


KEYS = ["participant_id", "scene_id"]
DEFAULT_OUTCOMES = [
    f"eeg_{roi}_{band}"
    for roi in ("F", "P", "O")
    for band in ("theta", "alpha", "beta")
]


def run_onset_sensitivity_analysis(
    sensitivity_trial_csv: str | Path,
    common_qc_csv: str | Path,
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path,
    eeg_analysis_config: str | Path | dict = "configs/eeg_analysis.json",
) -> dict[str, Path]:
    config = load_eeg_analysis_config(eeg_analysis_config)
    trials = _prepare_design(
        read_table(sensitivity_trial_csv),
        read_table(participants_csv),
        read_table(scene_manifest_csv),
    )
    common = read_table(common_qc_csv)
    trials = trials.merge(
        common[KEYS + ["onset_common_qc_pass"]], on=KEYS, how="left",
        validate="many_to_one",
    )
    outcomes = [outcome for outcome in DEFAULT_OUTCOMES if outcome in trials]
    if not outcomes:
        raise ValueError("No registered EEG ROI-band outcomes found in onset sensitivity table")

    models, diagnostics = fit_variant_model_suite(trials, outcomes)
    common_trials = trials.loc[_truthy(trials["onset_common_qc_pass"])].copy()
    equivalence = paired_cluster_bootstrap_equivalence(
        common_trials,
        outcomes=outcomes,
        trim_a=10.0,
        trim_b=15.0,
        bound_sd=config["equivalence_bound_sd"],
        iterations=config["bootstrap_iterations"],
        seed=config["random_seed"],
    )
    comparisons = descriptive_window_comparisons(models, reference_trim_s=10.0)
    bootstrap_failures = equivalence.loc[
        ~equivalence.get("status", pd.Series(index=equivalence.index, dtype=str)).eq("tested")
        | pd.to_numeric(
            equivalence.get("bootstrap_success_rate", pd.Series(index=equivalence.index, dtype=float)),
            errors="coerce",
        ).lt(0.90)
    ].copy()
    readiness = onset_reviewer_readiness(
        trials=trials,
        common=common,
        models=models,
        equivalence=equivalence,
        expected_variants=config["onset_trim_variants_s"],
    )
    methods = (
        "为估计场景进入后的持续稳态活动，在功率谱密度（PSD）与质量控制（QC）计算前，"
        "剔除每个场景前10秒；以15秒作为主要稳健性窗口，并以5秒和未截断（0秒）版本"
        "审计窗口敏感性。5、10和15秒版本分别重算QC，并在三者共同通过QC的参与者–场景"
        "试次上正式比较10秒与15秒结果。"
    )
    limitations = (
        "固定起始截断只能降低场景切换及问卷后运动伪迹的影响，不能证明残余carryover完全"
        "消失。最初10秒的真实场景反应不属于本研究的持续稳态估计对象；5秒灰屏未被视为"
        "独立干净基线；前序条件模型只覆盖可观测的一阶PreviousWWR和PreviousComplexity。"
    )
    outdir = Path(outdir)
    outputs = {
        "onset_variant_model_results": write_table(models, outdir / "onset_variant_model_results.csv"),
        "onset_variant_model_diagnostics": write_table(diagnostics, outdir / "onset_variant_model_diagnostics.csv"),
        "onset_window_comparisons": write_table(comparisons, outdir / "onset_window_comparisons.csv"),
        "onset_equivalence_10_vs_15": write_table(equivalence, outdir / "onset_equivalence_10_vs_15.csv"),
        "onset_bootstrap_failures": write_table(
            bootstrap_failures, outdir / "onset_bootstrap_failures.csv"
        ),
        "onset_reviewer_readiness": write_table(readiness, outdir / "onset_reviewer_readiness.csv"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    methods_path = outdir / "onset_methods_and_limitations.md"
    methods_path.write_text(
        "# EEG 场景起始截断方法\n\n## Methods\n\n"
        + methods + "\n\n## Limitations\n\n" + limitations + "\n",
        encoding="utf-8",
    )
    outputs["onset_methods_and_limitations"] = methods_path
    return outputs


def fit_variant_model_suite(
    trials: pd.DataFrame,
    outcomes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    diagnostics: list[dict] = []
    for trim, variant in trials.groupby("onset_trim_s", sort=True):
        eligible = variant.loc[
            ~_truthy(variant.get("bad_eeg_quality", pd.Series(False, index=variant.index)))
            & ~_truthy(variant.get("eeg_subject_quality_exclusion", pd.Series(False, index=variant.index)))
        ].copy()
        _fit_scope(eligible, outcomes, float(trim), "variant_specific_qc", False, rows, diagnostics)
        _fit_scope(
            eligible.loc[pd.to_numeric(eligible.get("position"), errors="coerce").gt(1)],
            outcomes, float(trim), "previous_scene", True, rows, diagnostics,
        )
        _fit_scope(
            eligible.loc[pd.to_numeric(eligible.get("block"), errors="coerce").eq(1)],
            outcomes, float(trim), "block1", False, rows, diagnostics,
        )
        if float(trim) in {5.0, 10.0, 15.0}:
            common = eligible.loc[_truthy(eligible.get("onset_common_qc_pass", pd.Series(False, index=eligible.index)))]
            _fit_scope(common, outcomes, float(trim), "common_5_10_15_qc", False, rows, diagnostics)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _fit_scope(
    data: pd.DataFrame,
    outcomes: Iterable[str],
    trim: float,
    scope: str,
    previous: bool,
    rows: list[dict],
    diagnostics: list[dict],
) -> None:
    terms = _design_terms(data, previous=previous)
    for outcome in outcomes:
        work = data.copy()
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        needed = [outcome, "participant_id"]
        work = work.dropna(subset=needed)
        formula = f"{outcome} ~ " + (" + ".join(terms) if terms else "1")
        if len(work) < 12 or work["participant_id"].nunique() < 3:
            diagnostics.append(_diagnostic(trim, scope, outcome, formula, "insufficient_data", work))
            continue
        try:
            fit = smf.gee(
                formula, groups="participant_id", data=work,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Independence(),
            ).fit(cov_type="robust")
            ci = fit.conf_int()
            if not np.isfinite(fit.params).all():
                raise FloatingPointError("nonfinite_coefficients")
        except Exception as exc:
            diagnostics.append(_diagnostic(
                trim, scope, outcome, formula, f"fit_failed:{type(exc).__name__}",
                work, str(exc),
            ))
            continue
        for term, estimate in fit.params.items():
            rows.append({
                "onset_trim_s": trim,
                "sample_strategy": scope,
                "outcome": outcome,
                "term": str(term),
                "estimate": float(estimate),
                "std_error": float(fit.bse[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "p_value": float(fit.pvalues[term]),
                "n_obs": int(fit.nobs),
                "n_subjects": int(work["participant_id"].nunique()),
                "formula": formula,
                "model_type": "gee_gaussian_participant_clustered_robust",
            })
        diagnostics.append(_diagnostic(trim, scope, outcome, formula, "fit", work))


def paired_cluster_bootstrap_equivalence(
    data: pd.DataFrame,
    *,
    outcomes: Iterable[str],
    trim_a: float = 10.0,
    trim_b: float = 15.0,
    bound_sd: float = 0.20,
    iterations: int = 5000,
    seed: int = 20260802,
    formula_terms: list[str] | None = None,
) -> pd.DataFrame:
    """Compare paired window coefficients after scaling by the 10-s outcome SD."""
    required = {*KEYS, "participant_id", "onset_trim_s"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Equivalence input missing {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for outcome in outcomes:
        if outcome not in data:
            continue
        pair = _paired_outcome_data(data, outcome, trim_a, trim_b)
        scale_values = pd.to_numeric(
            pair.loc[np.isclose(pair["onset_trim_s"], trim_a), outcome], errors="coerce"
        )
        scale_mean = float(scale_values.mean())
        scale_sd = float(scale_values.std(ddof=1))
        if not np.isfinite(scale_sd) or scale_sd <= 0:
            rows.append(_failed_equivalence_row(outcome, "invalid_primary_scale", iterations, bound_sd))
            continue
        pair["_standardized_outcome"] = (pd.to_numeric(pair[outcome], errors="coerce") - scale_mean) / scale_sd
        terms = formula_terms or _design_terms(pair, previous=False)
        formula = "_standardized_outcome ~ " + (" + ".join(terms) if terms else "1")
        original_a = _fit_coefficients(pair.loc[np.isclose(pair["onset_trim_s"], trim_a)], formula)
        original_b = _fit_coefficients(pair.loc[np.isclose(pair["onset_trim_s"], trim_b)], formula)
        registered = [
            term for term in original_a.index.intersection(original_b.index)
            if term != "Intercept" and _registered_condition_term(term)
        ]
        if formula_terms is not None:
            registered = [term for term in original_a.index.intersection(original_b.index) if term != "Intercept"]
        estimates = {term: [] for term in registered}
        paired_participants = sorted(pair["participant_id"].astype(str).unique())
        for _ in range(iterations):
            sampled = rng.choice(paired_participants, size=len(paired_participants), replace=True)
            pieces = []
            for index, participant in enumerate(sampled):
                piece = pair.loc[pair["participant_id"].astype(str).eq(participant)].copy()
                piece["participant_id"] = f"bootstrap_{index}"
                pieces.append(piece)
            boot = pd.concat(pieces, ignore_index=True)
            try:
                coef_a = _fit_coefficients(boot.loc[np.isclose(boot["onset_trim_s"], trim_a)], formula)
                coef_b = _fit_coefficients(boot.loc[np.isclose(boot["onset_trim_s"], trim_b)], formula)
            except Exception:
                continue
            for term in registered:
                if term in coef_a and term in coef_b:
                    difference = float(coef_b[term] - coef_a[term])
                    if np.isfinite(difference):
                        estimates[term].append(difference)
        for term in registered:
            values = np.asarray(estimates[term], dtype=float)
            success = int(len(values))
            success_rate = success / iterations
            difference = float(original_b[term] - original_a[term])
            ci90_low, ci90_high = _quantiles(values, (0.05, 0.95))
            ci95_low, ci95_high = _quantiles(values, (0.025, 0.975))
            se = float(np.std(values, ddof=1)) if success > 1 else np.nan
            if success_rate < 0.90:
                status = "insufficient_bootstrap_success"
                p_tost = np.nan
            elif not np.isfinite(se) or se <= 0:
                status = "invalid_bootstrap_se"
                p_tost = np.nan
            else:
                p_lower = 1 - norm.cdf((difference + bound_sd) / se)
                p_upper = norm.cdf((difference - bound_sd) / se)
                p_tost = float(max(p_lower, p_upper))
                status = "tested"
            rows.append({
                "outcome": outcome,
                "term": term,
                "comparison": f"{trim_b:g}s_minus_{trim_a:g}s",
                "standardization": f"{trim_a:g}s_outcome_mean_sd",
                "estimate_difference_sd": difference,
                "ci90_low_sd": ci90_low,
                "ci90_high_sd": ci90_high,
                "ci95_low_sd": ci95_low,
                "ci95_high_sd": ci95_high,
                "equivalence_bound_low_sd": -bound_sd,
                "equivalence_bound_high_sd": bound_sd,
                "p_tost": p_tost,
                "bootstrap_requested": iterations,
                "bootstrap_successful": success,
                "bootstrap_success_rate": success_rate,
                "status": status,
                "n_participants": len(paired_participants),
                "n_paired_trials": int(pair[KEYS].drop_duplicates().shape[0]),
                "formula": formula,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_tost_fdr_bh"] = _bh(out["p_tost"])
    within = out["ci90_low_sd"].gt(-bound_sd) & out["ci90_high_sd"].lt(bound_sd)
    out["equivalent"] = (
        out["status"].eq("tested") & within & out["p_tost_fdr_bh"].lt(0.05)
    )
    return out


def descriptive_window_comparisons(models: pd.DataFrame, reference_trim_s: float = 10.0) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    primary = models.loc[
        models["sample_strategy"].eq("common_5_10_15_qc")
        & np.isclose(models["onset_trim_s"], reference_trim_s)
    ]
    rows = []
    for trim in (0.0, 5.0, 15.0):
        strategy = "common_5_10_15_qc" if trim in {5.0, 15.0} else "variant_specific_qc"
        comparison = models.loc[
            models["sample_strategy"].eq(strategy) & np.isclose(models["onset_trim_s"], trim)
        ]
        reference = primary if strategy == "common_5_10_15_qc" else models.loc[
            models["sample_strategy"].eq("variant_specific_qc")
            & np.isclose(models["onset_trim_s"], reference_trim_s)
        ]
        merged = comparison.merge(reference, on=["outcome", "term"], suffixes=("_comparison", "_reference"))
        for _, row in merged.iterrows():
            difference = row["estimate_comparison"] - row["estimate_reference"]
            se = np.sqrt(row["std_error_comparison"] ** 2 + row["std_error_reference"] ** 2)
            rows.append({
                "comparison": f"{trim:g}s_minus_{reference_trim_s:g}s",
                "sample_strategy": strategy,
                "outcome": row["outcome"], "term": row["term"],
                "estimate_difference": difference,
                "ci95_low": difference - 1.96 * se,
                "ci95_high": difference + 1.96 * se,
                "direction_consistent": np.sign(row["estimate_comparison"]) == np.sign(row["estimate_reference"]),
                "inference_role": "formal_equivalence_reported_separately" if trim == 15 else "descriptive_only",
            })
    return pd.DataFrame(rows)


def onset_reviewer_readiness(
    *,
    trials: pd.DataFrame,
    common: pd.DataFrame,
    models: pd.DataFrame,
    equivalence: pd.DataFrame,
    expected_variants: Iterable[float],
) -> pd.DataFrame:
    observed = sorted(pd.to_numeric(trials["onset_trim_s"], errors="coerce").dropna().unique())
    required_scopes = {
        (trim, scope)
        for trim in (0.0, 5.0, 10.0, 15.0)
        for scope in ("variant_specific_qc", "previous_scene", "block1")
    } | {
        (trim, "common_5_10_15_qc") for trim in (5.0, 10.0, 15.0)
    }
    observed_scopes = set(zip(
        pd.to_numeric(models.get("onset_trim_s"), errors="coerce"),
        models.get("sample_strategy", pd.Series(dtype=str)).astype(str),
    )) if not models.empty else set()
    checks = [
        ("all_onset_variants_exported", observed == sorted(float(v) for v in expected_variants)),
        ("variant_specific_qc_present", not trials.empty and "bad_eeg_quality" in trials),
        ("common_5_10_15_qc_present", not common.empty and "onset_common_qc_pass" in common),
        ("model_suite_present", required_scopes.issubset(observed_scopes)),
        ("formal_10_vs_15_equivalence_present", not equivalence.empty),
        (
            "formal_equivalence_rows_tested",
            not equivalence.empty and equivalence["status"].eq("tested").all(),
        ),
        (
            "bootstrap_success_at_least_90pct",
            not equivalence.empty and equivalence["bootstrap_success_rate"].ge(0.90).all(),
        ),
    ]
    complete = all(passed for _, passed in checks)
    rows = [{"check": name, "pass": bool(passed)} for name, passed in checks]
    rows.append({
        "check": "reviewer_evidence_complete",
        "pass": complete,
        "allowed_wording": (
            "transition influence mitigated and window sensitivity evaluated"
            if complete else "onset-window evidence incomplete"
        ),
        "forbidden_wording": "carryover eliminated; carryover completely excluded",
    })
    return pd.DataFrame(rows)


def _prepare_design(trials: pd.DataFrame, participants: pd.DataFrame, scene: pd.DataFrame) -> pd.DataFrame:
    out = trials.copy()
    for canonical in DEFAULT_OUTCOMES:
        raw = canonical.removeprefix("eeg_")
        if canonical not in out and raw in out:
            out[canonical] = out[raw]
    participant_columns = [
        column for column in (
            "Gender", "ExperienceGroup", "OrderGroup", "Order", "Age"
        )
        if column in participants and column not in out
    ]
    if participant_columns:
        out = out.merge(
            participants[["participant_id", *participant_columns]].drop_duplicates("participant_id"),
            on="participant_id", how="left", validate="many_to_one",
        )
    scene_columns = [
        column for column in (
            "WWR", "Complexity", "block", "position", "OrderGroup",
            "participant_order", "order_scheme",
        )
        if column in scene and column not in out
    ]
    if scene_columns:
        out = out.merge(
            scene[KEYS + scene_columns].drop_duplicates(KEYS),
            on=KEYS, how="left", validate="many_to_one",
        )
    if "position" in out:
        out["position_centered"] = pd.to_numeric(out["position"], errors="coerce") - 3.5
    if "OrderGroup" not in out:
        for source in ("participant_order", "Order", "order_scheme"):
            if source in out:
                out["OrderGroup"] = out[source].astype(str)
                break
    sort = [column for column in ["participant_id", "block", "position", "scene_id", "onset_trim_s"] if column in out]
    out = out.sort_values(sort).reset_index(drop=True)
    lag_group = ["participant_id", "onset_trim_s"] + (["block"] if "block" in out else [])
    if "WWR" in out:
        out["previous_WWR"] = out.groupby(lag_group, sort=False)["WWR"].shift(1)
    if "Complexity" in out:
        out["previous_Complexity"] = out.groupby(lag_group, sort=False)["Complexity"].shift(1)
    if "position" in out:
        first = pd.to_numeric(out["position"], errors="coerce").eq(1)
        for column in ("previous_WWR", "previous_Complexity"):
            if column in out:
                out.loc[first, column] = np.nan
    return out


def _paired_outcome_data(data: pd.DataFrame, outcome: str, trim_a: float, trim_b: float) -> pd.DataFrame:
    work = data.loc[
        np.isclose(pd.to_numeric(data["onset_trim_s"], errors="coerce"), trim_a)
        | np.isclose(pd.to_numeric(data["onset_trim_s"], errors="coerce"), trim_b)
    ].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    complete = (
        work.dropna(subset=[outcome])
        .groupby(KEYS)["onset_trim_s"].nunique()
        .loc[lambda value: value.eq(2)].index
    )
    complete_frame = pd.DataFrame(complete.tolist(), columns=KEYS)
    return work.merge(complete_frame, on=KEYS, how="inner", validate="many_to_one")


def _fit_coefficients(data: pd.DataFrame, formula: str) -> pd.Series:
    fit = smf.gee(
        formula, groups="participant_id", data=data,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit(cov_type="robust", maxiter=100)
    if not np.isfinite(fit.params).all():
        raise FloatingPointError("nonfinite_coefficients")
    return fit.params


def _design_terms(data: pd.DataFrame, *, previous: bool) -> list[str]:
    terms: list[str] = []
    if _varying(data, "WWR") and _varying(data, "Complexity"):
        terms.append("C(WWR) * C(Complexity)")
    else:
        terms.extend(f"C({name})" for name in ("WWR", "Complexity") if _varying(data, name))
    if _varying(data, "ExperienceGroup"):
        for condition in ("WWR", "Complexity"):
            if _varying(data, condition):
                terms.append(f"C({condition}) * C(ExperienceGroup)")
    if _varying(data, "Gender"):
        terms.append("C(Gender)")
    terms.extend(name for name in ("block", "position_centered") if _varying(data, name))
    if _varying(data, "OrderGroup"):
        terms.append("C(OrderGroup)")
    if previous:
        terms.extend(
            f"C({name})" for name in ("previous_WWR", "previous_Complexity")
            if _varying(data, name)
        )
    return list(dict.fromkeys(terms))


def _registered_condition_term(term: str) -> bool:
    return any(token in term for token in ("WWR", "Complexity", "ExperienceGroup"))


def _varying(data: pd.DataFrame, column: str) -> bool:
    return column in data and data[column].dropna().nunique() > 1


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _quantiles(values: np.ndarray, probs: tuple[float, float]) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    result = np.quantile(values, probs)
    return float(result[0]), float(result[1])


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = (valid * len(valid) / np.arange(1, len(valid) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def _diagnostic(trim: float, scope: str, outcome: str, formula: str, status: str, data: pd.DataFrame, error: str = "") -> dict:
    return {
        "onset_trim_s": trim, "sample_strategy": scope, "outcome": outcome,
        "status": status, "n_obs": len(data),
        "n_subjects": data["participant_id"].nunique() if "participant_id" in data else 0,
        "formula": formula, "error": error,
    }


def _failed_equivalence_row(outcome: str, status: str, iterations: int, bound: float) -> dict:
    return {
        "outcome": outcome, "term": "", "comparison": "15s_minus_10s",
        "estimate_difference_sd": np.nan, "ci90_low_sd": np.nan,
        "ci90_high_sd": np.nan, "ci95_low_sd": np.nan, "ci95_high_sd": np.nan,
        "equivalence_bound_low_sd": -bound, "equivalence_bound_high_sd": bound,
        "p_tost": np.nan, "bootstrap_requested": iterations,
        "bootstrap_successful": 0, "bootstrap_success_rate": 0.0,
        "status": status, "n_participants": 0, "n_paired_trials": 0,
        "formula": "",
    }
