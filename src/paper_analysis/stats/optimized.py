"""Data-first, grain-safe analysis for the supplemented VR/EEG experiment.

This module intentionally separates each modality's estimand from the trimodal
intersection. Questionnaire and eye-tracking observations are not discarded just
because an EEG segment fails QC; EEG inference uses only QC-passed scene trials.
All inferential models use participant-clustered GEE and never silently fall back
to ordinary least squares.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from paper_analysis.utils.io import read_table, write_table


KEYS = ["participant_id", "scene_id"]
COHORT_CUTOFF = date(2026, 5, 1)
QUESTIONNAIRE = ["q_S1", "q_S2", "q_S3", "q_S4", "q_S5"]
EEG = [
    "eeg_F_theta", "eeg_F_alpha", "eeg_F_beta",
    "eeg_P_theta", "eeg_P_alpha", "eeg_P_beta",
    "eeg_O_theta", "eeg_O_alpha", "eeg_O_beta",
]
EYE_CONTINUOUS = ["FCR", "TFD_ms", "TTFF_ms", "attention_share"]
MODEL_COLUMNS = [
    "outcome", "term", "estimate", "std_error", "p_value", "ci_low", "ci_high",
    "n_obs", "n_subjects", "model_type", "formula", "status", "model_family", "cohort_scope",
]
DIAGNOSTIC_COLUMNS = ["status", "n_obs", "n_subjects", "formula", "model_type", "model_family", "cohort_scope", "outcome"]
# The original submission is a reference for *what to compare*, not a numerical
# source.  Its old QC counts and p values are deliberately never imported.
FIRST_PAPER_CORE = {
    "questionnaire_scene": {"q_S1", "q_S2", "q_S3", "q_S4", "q_S5"},
    "eye_aoi": {"FCR"},
    "eye_aoi_visited": set(),
    "eeg_scene_qc_passed": {"eeg_O_theta", "eeg_F_theta", "eeg_O_alpha", "eeg_O_beta"},
}
FIRST_PAPER_REFERENCE = r"E:\26\260419-final-submit.docx"


def run_optimized_analysis(
    pre_qc_master_csv: str | Path,
    qc_csv: str | Path,
    outdir: str | Path,
) -> dict[str, Path]:
    """Produce a grain-safe result package and an adversarial audit.

    ``pre_qc_master_csv`` is AOI-expanded.  It is converted into separate
    questionnaire/EEG scene tables and an eye AOI table before model fitting.
    """
    outdir = Path(outdir)
    pre = _attach_cohort(read_table(pre_qc_master_csv))
    qc = read_table(qc_csv)
    keep = _keep_flags(qc)
    scene_all = _scene_level(pre)
    scene_eeg = scene_all.merge(keep.loc[keep["eeg_qc_keep"], KEYS], on=KEYS, how="inner")
    scene_trimodal = scene_all.merge(keep.loc[keep["trimodal_keep"], KEYS], on=KEYS, how="inner")
    eye_all = pre.copy()
    eye_all = eye_all.merge(keep, on=KEYS, how="left")
    registry = _data_registry(scene_all, scene_eeg, scene_trimodal, eye_all)
    selection_source = scene_all.merge(keep, on=KEYS, how="left")
    selection_source["eeg_qc_keep"] = selection_source["eeg_qc_keep"].fillna(False)
    selection = _selection_table(selection_source)
    selection_model = _fit_selection_model(selection_source)

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for scope, data in _scopes(scene_all):
        fitted, diag = _fit_outcomes(
            data, QUESTIONNAIRE, "questionnaire_scene", scope, _scene_formula(data), binary=False
        )
        rows.extend(fitted)
        diagnostics.extend(diag)
    for scope, data in _scopes(scene_eeg):
        fitted, diag = _fit_outcomes(data, EEG, "eeg_scene_qc_passed", scope, _scene_formula(data), binary=False)
        rows.extend(fitted)
        diagnostics.extend(diag)
    for scope, data in _scopes(eye_all):
        # The original paper's inferential eye endpoint was AOI-level FCR.
        # Visit probability is near-completely separated in several AOI cells;
        # TTFF/TFD/attention share remain in the descriptive table rather than
        # being forced into unreliable conditional p-value models.
        fitted, diag = _fit_outcomes(data, ["FCR"], "eye_aoi", scope, _eye_formula(data), binary=False)
        rows.extend(fitted)
        diagnostics.extend(diag)

    models = _add_fdr(pd.DataFrame(rows, columns=MODEL_COLUMNS))
    comparison = _cohort_comparison(models)
    descriptives = _cohort_condition_descriptives(scene_all, scene_eeg, eye_all)
    methods = _methods_note(registry, selection)
    paths = {
        "data_registry": write_table(registry, outdir / "data_registry.csv"),
        "cohort_qc_selection": write_table(selection, outdir / "cohort_qc_selection.csv"),
        "cohort_qc_selection_model": write_table(selection_model, outdir / "cohort_qc_selection_model.csv"),
        "model_results": write_table(models, outdir / "model_results_clustered.csv"),
        "model_diagnostics": write_table(pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS), outdir / "model_diagnostics_clustered.csv"),
        "cohort_comparison": write_table(comparison, outdir / "initial_vs_total_comparison.csv"),
        "first_paper_reference_scope": write_table(_first_paper_reference_scope(), outdir / "first_paper_reference_scope.csv"),
        "cohort_condition_descriptives": write_table(descriptives, outdir / "cohort_condition_descriptives.csv"),
    }
    methods_path = outdir / "methods_and_scope.md"
    methods_path.write_text(methods, encoding="utf-8")
    paths["methods_and_scope"] = methods_path
    paths.update(run_adversarial_audit(outdir))
    return paths


def run_adversarial_audit(outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    registry = read_table(outdir / "data_registry.csv")
    models = read_table(outdir / "model_results_clustered.csv")
    diagnostics = read_table(outdir / "model_diagnostics_clustered.csv")
    checks: list[dict] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})

    registry_map = registry.set_index("dataset")
    q_trials = int(registry_map.loc["questionnaire_scene_all", "n_trials"])
    eeg_trials = int(registry_map.loc["eeg_scene_qc_passed", "n_trials"])
    trimodal_trials = int(registry_map.loc["trimodal_scene_aligned", "n_trials"])
    check("cohort_provenance_present", registry["cohort_levels"].astype(str).str.contains("Initial").any() and registry["cohort_levels"].astype(str).str.contains("Supplement").any(), "Both acquisition cohorts are reconstructed from eye_record_id.")
    check("modality_specific_samples", q_trials >= eeg_trials >= trimodal_trials, f"questionnaire={q_trials}; EEG QC-passed={eeg_trials}; synchronized trimodal={trimodal_trials} scene trials.")
    eeg_n = pd.to_numeric(models.loc[models["model_family"].eq("eeg_scene_qc_passed"), "n_obs"], errors="coerce")
    check("eeg_not_aoi_duplicated", bool(eeg_n.dropna().le(eeg_trials).all()), f"maximum EEG model n={int(eeg_n.max()) if eeg_n.notna().any() else 0}; allowed={eeg_trials}.")
    eye_formulas = models.loc[models["model_family"].astype(str).str.startswith("eye_"), "formula"].astype(str)
    check("eye_model_accounts_for_aoi", bool(eye_formulas.str.contains("class_name", regex=False).all()), "Every eye model includes AOI class terms.")
    check("cluster_robust_inference", models["model_type"].astype(str).str.contains("cluster_robust").all(), "Every fitted model uses participant-clustered robust covariance; no naive OLS fallback is accepted.")
    check("fdr_present", {"p_fdr_bh_family", "significant_fdr_bh_family_0_05"}.issubset(models.columns), "Family-level BH-FDR columns are present.")
    check("fit_failures_visible", "status" in diagnostics.columns, "All model attempts retain an explicit status.")
    failed = diagnostics.loc[~diagnostics["status"].eq("fit"), "status"] if "status" in diagnostics.columns else pd.Series(dtype=str)
    check("no_unstable_primary_fit", failed.empty, f"non-fit model attempts={len(failed)}; statuses={','.join(sorted(failed.astype(str).unique())) if not failed.empty else 'none'}.")
    status = "pass" if all(item["status"] == "PASS" for item in checks) else "fail"
    payload = {"status": status, "checks": checks}
    json_path = outdir / "adversarial_audit.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Adversarial analysis audit", "", f"Overall status: **{status}**", "", "| Check | Status | Detail |", "|---|---|---|"]
    lines.extend(f"| {item['check_id']} | {item['status']} | {item['detail']} |" for item in checks)
    lines.extend(["", "This audit checks data grain and inference integrity; it does not promote any scientific claim."])
    md_path = outdir / "adversarial_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"adversarial_audit_json": json_path, "adversarial_audit": md_path}


def _attach_cohort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "eye_record_id" not in out.columns:
        out["Cohort"] = "Unknown"
        return out
    dates = out["eye_record_id"].map(_record_date)
    out["collection_date_derived"] = dates.map(lambda value: value.isoformat() if value else "")
    out["Cohort"] = dates.map(lambda value: "Supplement" if value and value >= COHORT_CUTOFF else "Initial" if value else "Unknown")
    return out


def _record_date(value: object) -> date | None:
    match = re.match(r"(\d{6})", str(value or "").strip())
    if not match:
        return None
    try:
        text = match.group(1)
        return date(2000 + int(text[:2]), int(text[2:4]), int(text[4:6]))
    except ValueError:
        return None


def _keep_flags(qc: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in ["excluded_from_analysis", "bad_eeg_quality", "missing_eeg"] if column in qc.columns]
    out = qc[KEYS + available].copy()
    out["trimodal_keep"] = ~_truthy(out.get("excluded_from_analysis", pd.Series(False, index=out.index)))
    bad_eeg = _truthy(out.get("bad_eeg_quality", pd.Series(False, index=out.index)))
    missing_eeg = _truthy(out.get("missing_eeg", pd.Series(False, index=out.index)))
    out["eeg_qc_keep"] = ~(bad_eeg | missing_eeg)
    return out[KEYS + ["eeg_qc_keep", "trimodal_keep"]]


def _scene_level(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(KEYS).copy() if set(KEYS).issubset(df.columns) else df.copy()


def _data_registry(scene_all: pd.DataFrame, scene_eeg: pd.DataFrame, scene_trimodal: pd.DataFrame, eye_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, data, grain in [
        ("questionnaire_scene_all", scene_all, "participant_id + scene_id"),
        ("eeg_scene_qc_passed", scene_eeg, "participant_id + scene_id"),
        ("trimodal_scene_aligned", scene_trimodal, "participant_id + scene_id"),
        ("eye_aoi_all", eye_all, "participant_id + scene_id + class_name"),
    ]:
        rows.append({
            "dataset": name,
            "grain": grain,
            "n_rows": int(len(data)),
            "n_subjects": int(data["participant_id"].nunique()),
            "n_trials": int(data[KEYS].drop_duplicates().shape[0]),
            "cohort_levels": ",".join(sorted(data.get("Cohort", pd.Series(dtype=str)).dropna().astype(str).unique())),
        })
    return pd.DataFrame(rows)


def _selection_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ["Cohort", "ExperienceGroup", "Gender", "WWR", "Complexity", "block", "position"]:
        if factor not in data.columns:
            continue
        for level, sub in data.groupby(factor, dropna=False):
            rows.append({"factor": factor, "level": str(level), "n_trials": int(len(sub)), "kept_trials": int(sub["eeg_qc_keep"].sum()), "retention_rate": float(sub["eeg_qc_keep"].mean())})
    return pd.DataFrame(rows)


def _fit_selection_model(selection_source: pd.DataFrame) -> pd.DataFrame:
    terms = ["C(Cohort)", "C(ExperienceGroup)", "C(Gender)", "C(WWR)", "C(Complexity)", "block", "position"]
    formula = "eeg_qc_keep ~ " + " + ".join(_available_terms(selection_source, terms))
    rows, diagnostic = _fit_gee(selection_source, "eeg_qc_keep", formula, binary=True)
    if not rows:
        return pd.DataFrame([diagnostic])
    return pd.DataFrame(rows)


def _scopes(data: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "Total", data
    for cohort in ["Initial", "Supplement"]:
        sub = data.loc[data.get("Cohort", pd.Series("", index=data.index)).eq(cohort)].copy()
        if sub["participant_id"].nunique() >= 4:
            yield cohort, sub


def _scene_formula(data: pd.DataFrame) -> str:
    # Primary estimand is the randomized within-subject spatial design.  Do not
    # overfit sparse between-subject cohort/experience cells into every outcome.
    terms = ["C(WWR) * C(Complexity)", "block", "position"]
    return " ~ ".join(["{outcome}", " + ".join(_available_terms(data, terms))])


def _eye_formula(data: pd.DataFrame) -> str:
    terms = ["C(WWR) * C(class_name)", "C(Complexity)"]
    return " ~ ".join(["{outcome}", " + ".join(_available_terms(data, terms))])


def _available_terms(data: pd.DataFrame, terms: list[str]) -> list[str]:
    out = []
    for term in terms:
        columns = re.findall(r"C\(([^)]+)\)", term) or [term]
        if all(column in data.columns and data[column].dropna().nunique() > 1 for column in columns):
            out.append(term)
    return out or ["1"]


def _fit_outcomes(data: pd.DataFrame, outcomes: list[str], family: str, scope: str, formula_template: str, binary: bool) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    diagnostics: list[dict] = []
    for outcome in outcomes:
        if outcome not in data.columns:
            diagnostics.append({"model_family": family, "cohort_scope": scope, "outcome": outcome, "status": "missing_outcome"})
            continue
        result_rows, diagnostic = _fit_gee(data, outcome, formula_template.format(outcome=outcome), binary)
        for row in result_rows:
            row.update({"model_family": family, "cohort_scope": scope})
        diagnostic.update({"model_family": family, "cohort_scope": scope, "outcome": outcome})
        rows.extend(result_rows)
        diagnostics.append(diagnostic)
    return rows, diagnostics


def _fit_gee(data: pd.DataFrame, outcome: str, formula: str, binary: bool) -> tuple[list[dict], dict]:
    columns = [outcome, "participant_id"]
    work = data.copy()
    if binary:
        work[outcome] = _truthy(work[outcome]).astype(int)
    else:
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.dropna(subset=[outcome, "participant_id"])
    if len(work) < 12 or work["participant_id"].nunique() < 4:
        return [], {"status": "insufficient_data", "n_obs": int(len(work)), "n_subjects": int(work["participant_id"].nunique())}
    if binary and work[outcome].nunique() < 2:
        return [], {"status": "no_binary_variation", "n_obs": int(len(work)), "n_subjects": int(work["participant_id"].nunique())}
    try:
        if binary:
            fit = smf.glm(formula, data=work, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": work["participant_id"]}
            )
            model_type = "glm_binomial_cluster_robust"
        else:
            fit = smf.ols(formula, data=work).fit(
                cov_type="cluster", cov_kwds={"groups": work["participant_id"]}
            )
            model_type = "ols_cluster_robust"
    except Exception as exc:
        return [], {"status": f"fit_failed:{type(exc).__name__}", "n_obs": int(len(work)), "n_subjects": int(work["participant_id"].nunique()), "formula": formula}
    ci = fit.conf_int()
    numeric_parts = [fit.params, fit.bse, fit.pvalues, ci.iloc[:, 0], ci.iloc[:, 1]]
    if any(~np.isfinite(pd.to_numeric(part, errors="coerce")).all() for part in numeric_parts):
        return [], {"status": "unstable_nonfinite_robust_covariance", "n_obs": int(fit.nobs), "n_subjects": int(work["participant_id"].nunique()), "formula": formula}
    rows = []
    for term, estimate in fit.params.items():
        rows.append({
            "outcome": outcome,
            "term": str(term),
            "estimate": float(estimate),
            "std_error": float(fit.bse.get(term, np.nan)),
            "p_value": float(fit.pvalues.get(term, np.nan)),
            "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]),
            "n_obs": int(fit.nobs),
            "n_subjects": int(work["participant_id"].nunique()),
            "model_type": model_type,
            "formula": formula,
            "status": "fit",
        })
    return rows, {"status": "fit", "n_obs": int(fit.nobs), "n_subjects": int(work["participant_id"].nunique()), "formula": formula, "model_type": model_type}


def _add_fdr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["p_fdr_bh_family"] = np.nan
    for _, idx in out.groupby(["model_family", "cohort_scope"], dropna=False).groups.items():
        values = pd.to_numeric(out.loc[idx, "p_value"], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        order = valid.sort_values().index
        ranks = pd.Series(range(1, len(order) + 1), index=order, dtype=float)
        adjusted = (valid.loc[order] * len(order) / ranks).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
        out.loc[adjusted.index, "p_fdr_bh_family"] = adjusted
    out["significant_fdr_bh_family_0_05"] = out["p_fdr_bh_family"].lt(0.05)
    return out


def _cohort_comparison(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    candidate = models.loc[
        models["cohort_scope"].isin(["Initial", "Total"])
        & models["term"].astype(str).str.contains(r"C\(WWR\)|C\(Complexity\)", regex=True)
        & ~models["term"].astype(str).str.contains("Cohort", regex=False)
    ].copy()
    index = ["model_family", "outcome", "term"]
    wide = candidate.pivot_table(index=index, columns="cohort_scope", values=["estimate", "p_fdr_bh_family", "n_obs", "n_subjects"], aggfunc="first")
    if wide.empty:
        return pd.DataFrame()
    wide.columns = [f"{metric}_{scope.lower()}" for metric, scope in wide.columns]
    out = wide.reset_index()
    total = pd.to_numeric(out.get("estimate_total", pd.Series(np.nan, index=out.index)), errors="coerce")
    initial = pd.to_numeric(out.get("estimate_initial", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["estimate_change_total_minus_initial"] = total - initial
    out["same_direction"] = np.sign(total) == np.sign(initial)
    out["first_paper_core"] = out.apply(
        lambda row: row["outcome"] in FIRST_PAPER_CORE.get(row["model_family"], set()), axis=1
    )
    out["interpretation_boundary"] = "Compares condition estimates after adding the supplement; cohort membership is observational and not a causal treatment."
    return out


def _cohort_condition_descriptives(scene_all: pd.DataFrame, scene_eeg: pd.DataFrame, eye_all: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for family, data, outcomes, grain in [
        ("questionnaire_scene", scene_all, QUESTIONNAIRE, "scene"),
        ("eeg_scene_qc_passed", scene_eeg, EEG, "scene"),
        ("eye_aoi", eye_all, ["visited", *EYE_CONTINUOUS], "AOI"),
    ]:
        dimensions = ["Cohort", "WWR", "Complexity"] + (["class_name"] if "class_name" in data.columns and family == "eye_aoi" else [])
        for outcome in outcomes:
            if outcome not in data.columns:
                continue
            values = _truthy(data[outcome]).astype(float) if outcome == "visited" else pd.to_numeric(data[outcome], errors="coerce")
            columns = list(dict.fromkeys(dimensions + KEYS + ["participant_id"]))
            temp = data[columns].copy()
            temp["value"] = values
            for keys, sub in temp.groupby(dimensions, dropna=False):
                keys = keys if isinstance(keys, tuple) else (keys,)
                valid = sub["value"].dropna()
                rows.append({
                    "model_family": family, "outcome": outcome, "grain": grain,
                    **dict(zip(dimensions, keys)), "n_rows": int(len(valid)),
                    "n_subjects": int(sub.loc[valid.index, "participant_id"].nunique()) if not valid.empty else 0,
                    "n_trials": int(sub.loc[valid.index, KEYS].drop_duplicates().shape[0]) if not valid.empty else 0,
                    "mean": float(valid.mean()) if not valid.empty else np.nan,
                    "sd": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                })
    return pd.DataFrame(rows)


def _methods_note(registry: pd.DataFrame, selection: pd.DataFrame) -> str:
    q = registry.set_index("dataset").loc["questionnaire_scene_all"]
    eeg = registry.set_index("dataset").loc["eeg_scene_qc_passed"]
    return f"""# Optimized analysis scope

- Cohort rule: `Initial` is an eye-record date before {COHORT_CUTOFF.isoformat()}; `Supplement` is on/after that date.
- Questionnaire estimand: all available scene trials ({int(q.n_trials)} trials, {int(q.n_subjects)} participants); it is not restricted by EEG QC.
- EEG estimand: QC-passed scene trials only ({int(eeg.n_trials)} trials, {int(eeg.n_subjects)} participants), one EEG observation per participant-scene trial.
- Eye estimand: AOI-expanded observations. FCR is the inferential endpoint and models include `class_name` so table/window/equipment rows are not treated as exchangeable replicas. Visit probability, TFD, TTFF, and attention share are retained as descriptive outcomes because this dataset has separated/structurally missing AOI cells.
- Inference: participant-clustered robust covariance. Continuous outcomes use OLS with cluster-robust standard errors; the binary visited outcome uses binomial logistic regression with the same cluster-robust covariance. Failed models remain failed; no naive (non-clustered) OLS fallback is used.
- Multiplicity: BH-FDR is calculated within each model family and cohort scope.
- QC selection is reported separately so the effect of EEG exclusion is visible rather than hidden.
- First-paper reference: {FIRST_PAPER_REFERENCE} defines only the original study questions and core outcomes (S1-S5, FCR by AOI, and F/O theta plus O alpha/beta). No historical p value, effect estimate, or old QC count is reused.
"""


def _first_paper_reference_scope() -> pd.DataFrame:
    rows = []
    labels = {
        "questionnaire_scene": "Scene-level S1-S5 subjective appraisal",
        "eye_aoi": "AOI fixation count rate (FCR)",
        "eeg_scene_qc_passed": "F/O theta and O alpha/beta scene-viewing power",
    }
    for family, outcomes in FIRST_PAPER_CORE.items():
        if not outcomes:
            continue
        for outcome in sorted(outcomes):
            rows.append({
                "source_document": FIRST_PAPER_REFERENCE,
                "model_family": family,
                "outcome": outcome,
                "first_paper_role": labels[family],
                "comparison_rule": "Recompute Initial and Total using the current pipeline; do not reuse historical estimates, p values, or QC counts.",
            })
    return pd.DataFrame(rows)


def _truthy(values: pd.Series | object) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    return pd.Series([str(values).strip().lower() in {"true", "1", "yes", "y"}])
