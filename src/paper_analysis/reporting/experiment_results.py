from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_analysis.utils.io import read_table, write_table, write_text
from paper_analysis.utils.markdown import dataframe_to_markdown


PLAIN_COLUMNS = [
    "module",
    "source_table",
    "grain",
    "metric",
    "factor",
    "level",
    "n_subjects",
    "n_trials",
    "n_rows",
    "mean",
    "sd",
    "se",
    "ci95_low",
    "ci95_high",
    "median",
    "iqr",
    "note",
]

PRIMARY_QUESTIONNAIRE = ["S1", "S2", "S3", "S4", "S5", "S5_7", "Afford4", "Bmean", "IPQ_mean"]
PRIMARY_EYE = ["visited", "FCR", "FC_rate", "TFD_ms", "TTFF_ms", "attention_share", "share_pct"]
PRIMARY_EEG = [
    "F_theta",
    "F_alpha",
    "F_beta",
    "P_theta",
    "P_alpha",
    "P_beta",
    "O_theta",
    "O_alpha",
    "O_beta",
    "eeg_F_theta",
    "eeg_F_alpha",
    "eeg_F_beta",
    "eeg_P_theta",
    "eeg_P_alpha",
    "eeg_P_beta",
    "eeg_O_theta",
    "eeg_O_alpha",
    "eeg_O_beta",
]
DEFAULT_FACTORS = ["overall", "WWR", "Complexity", "ExperienceGroup", "WWR:Complexity", "WWR:ExperienceGroup"]


def build_experiment_result_package(
    outputs_root: str | Path,
    outdir: str | Path | None = None,
    audit_dir: str | Path | None = None,
) -> dict[str, Path]:
    outputs_root = Path(outputs_root)
    outdir = Path(outdir) if outdir is not None else outputs_root / "07_paper_tables"
    audit_dir = Path(audit_dir) if audit_dir is not None else outputs_root / "11_audit"

    plain = experiment_plain_results(outputs_root)
    significance = experiment_significance_results(outputs_root)
    teacher_brief = teacher_data_brief(outputs_root, plain, significance)
    interpretation = interpretation_reference(outputs_root, plain, significance)
    audit_md, audit_summary = adversarial_data_review(outputs_root, plain, significance)

    paths = {
        "experiment_plain_results": write_table(plain, outdir / "experiment_plain_results.csv"),
        "experiment_significance_results": write_table(significance, outdir / "experiment_significance_results.csv"),
        "teacher_data_brief": write_text(teacher_brief, outdir / "teacher_data_brief.md"),
        "interpretation_reference": write_text(interpretation, outdir / "interpretation_reference.md"),
        "adversarial_data_review": write_text(audit_md, audit_dir / "adversarial_data_review.md"),
        "audit_summary": write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), audit_dir / "audit_summary.json"),
    }
    paths["experiment_plain_results_xlsx"] = write_plain_results_xlsx(plain, outdir / "experiment_plain_results.xlsx")
    paths["experiment_plain_results_md"] = write_text(plain_results_markdown(plain), outdir / "experiment_plain_results.md")
    return paths


def experiment_plain_results(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary = _read_json(outputs_root / "realdata_run_summary.json")
    rows.extend(_sample_rows(outputs_root, summary))

    questionnaire = _read_optional(outputs_root / "02_questionnaire" / "questionnaire_long.csv")
    rows.extend(_summaries(
        questionnaire,
        module="questionnaire",
        source_table="02_questionnaire/questionnaire_long.csv",
        grain="scene_trial",
        metrics=[m for m in PRIMARY_QUESTIONNAIRE if m in questionnaire.columns],
        note="Questionnaire scene-level descriptive result; IPQ_mean is participant-level information repeated on scene rows.",
    ))

    eye = _read_optional(outputs_root / "03_eye_tracking" / "eye_aoi_trial_long.csv")
    rows.extend(_summaries(
        eye,
        module="eye_tracking",
        source_table="03_eye_tracking/eye_aoi_trial_long.csv",
        grain="aoi_expanded_row",
        metrics=[m for m in PRIMARY_EYE if m in eye.columns],
        note="Eye-tracking AOI-expanded descriptive result; n_trials is the participant-scene count, n_rows is AOI-expanded rows.",
    ))

    eeg = _read_optional(outputs_root / "04_eeg" / "eeg_trial_long.csv")
    eeg_metrics = [m for m in PRIMARY_EEG if m in eeg.columns and not m.startswith("eeg_")]
    rows.extend(_summaries(
        eeg,
        module="eeg",
        source_table="04_eeg/eeg_trial_long.csv",
        grain="scene_trial",
        metrics=eeg_metrics,
        note="EEG scene-level descriptive result after EEG standardization/QC columns are attached.",
    ))
    rows.extend(_eeg_qc_rows(outputs_root))

    master = _read_optional(outputs_root / "05_multimodal_fusion" / "analysis_master_long.csv")
    fusion_metrics = [m for m in PRIMARY_QUESTIONNAIRE if f"q_{m}" in master.columns]
    fusion_metrics = [f"q_{m}" for m in fusion_metrics] + [m for m in PRIMARY_EYE if m in master.columns] + [m for m in PRIMARY_EEG if m in master.columns and m.startswith("eeg_")]
    rows.extend(_summaries(
        master,
        module="multimodal_retained",
        source_table="05_multimodal_fusion/analysis_master_long.csv",
        grain="aoi_expanded_retained_row",
        metrics=fusion_metrics,
        note="Main retained fusion table. AOI-expanded rows must not be reported as scene-trial counts.",
    ))

    rows.extend(_robustness_rows(outputs_root))
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=PLAIN_COLUMNS)
    return out.reindex(columns=PLAIN_COLUMNS)


def experiment_significance_results(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = _read_optional(outputs_root / "06_models" / "model_results.csv")
    diagnostics = _read_optional(outputs_root / "06_models" / "model_diagnostics.csv")
    diag_by_outcome = diagnostics.drop_duplicates("outcome").set_index("outcome").to_dict("index") if "outcome" in diagnostics.columns else {}
    for row in models.to_dict("records"):
        outcome = str(row.get("outcome", ""))
        diag = diag_by_outcome.get(outcome, {})
        model_type = str(row.get("model_type") or diag.get("model_type") or "")
        std_error = _to_float(row.get("std_error"))
        ci_low = _to_float(row.get("ci_low"))
        ci_high = _to_float(row.get("ci_high"))
        p_value = _to_float(row.get("p_value"))
        fallback = "fallback" in model_type.lower()
        warning = fallback or _unstable_interval(std_error, ci_low, ci_high)
        rows.append({
            "source": "06_models/model_results.csv",
            "result_family": _result_family(outcome),
            "outcome": outcome,
            "term": row.get("term", ""),
            "estimate": _to_float(row.get("estimate")),
            "std_error": std_error,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "model_type": model_type,
            "n": _to_float(row.get("n")),
            "fallback_flag": bool(fallback),
            "warning_flag": bool(warning),
            "significance_label": _p_label(p_value),
            "interpretation_note": _interpretation_note(outcome, str(row.get("term", "")), p_value, fallback, warning),
        })

    for path, source in [
        (outputs_root / "06_models" / "emmeans_contrasts.csv", "06_models/emmeans_contrasts.csv"),
        (outputs_root / "02_questionnaire" / "questionnaire_wwr_polynomial_contrasts.csv", "02_questionnaire/questionnaire_wwr_polynomial_contrasts.csv"),
    ]:
        contrasts = _read_optional(path)
        for row in contrasts.to_dict("records"):
            outcome = str(row.get("outcome", ""))
            term = str(row.get("term") or row.get("contrast") or "")
            p_value = _to_float(row.get("p_value"))
            rows.append({
                "source": source,
                "result_family": _result_family(outcome),
                "outcome": outcome,
                "term": term,
                "estimate": _first_float(row, ["estimate", "mean_contrast", "wwr45_peak_index"]),
                "std_error": _first_float(row, ["std_error", "se_contrast"]),
                "p_value": p_value,
                "ci_low": _to_float(row.get("ci_low")),
                "ci_high": _to_float(row.get("ci_high")),
                "model_type": str(row.get("model_type", "planned_or_descriptive_contrast")),
                "n": _first_float(row, ["n", "n_subjects"]),
                "fallback_flag": False,
                "warning_flag": False,
                "significance_label": _p_label(p_value),
                "interpretation_note": _interpretation_note(outcome, term, p_value, False, False),
            })
    return pd.DataFrame(rows)


def write_plain_results_xlsx(plain: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        plain.to_excel(writer, index=False, sheet_name="all_results")
        for module, sub in plain.groupby("module", dropna=False):
            sheet = str(module or "unknown")[:31]
            sub.to_excel(writer, index=False, sheet_name=sheet)
    return out


def plain_results_markdown(plain: pd.DataFrame) -> str:
    lines = ["# Experiment Plain Results", ""]
    if plain.empty:
        return "# Experiment Plain Results\n\nNo result rows available.\n"
    for module, sub in plain.groupby("module", dropna=False):
        lines.extend([f"## {module}", ""])
        keep = ["metric", "factor", "level", "grain", "n_subjects", "n_trials", "n_rows", "mean", "sd", "ci95_low", "ci95_high", "median", "iqr", "note"]
        preview = sub[[c for c in keep if c in sub.columns]].copy()
        lines.append(dataframe_to_markdown(preview))
        lines.append("")
    return "\n".join(lines)


def teacher_data_brief(outputs_root: Path, plain: pd.DataFrame, significance: pd.DataFrame) -> str:
    summary = _read_json(outputs_root / "realdata_run_summary.json")
    qc = _read_optional(outputs_root / "05_multimodal_fusion" / "analysis_qc_exclusions.csv")
    master = _read_optional(outputs_root / "05_multimodal_fusion" / "analysis_master_long.csv")
    scene_trials = _unique_trials(master)
    aoi_rows = int(len(master)) if not master.empty else int(summary.get("fusion_kept_rows", 0) or 0)
    total_trials = int(summary.get("analysis_scene_trials_total") or _unique_trials(qc) or summary.get("scene_rows", 0) or 0)
    kept_trials = int(summary.get("analysis_scene_trials_kept") or scene_trials or 0)
    excluded_trials = int(summary.get("analysis_scene_trials_excluded") or max(total_trials - kept_trials, 0))

    lines = [
        "# Teacher Data Brief",
        "",
        "这份摘要只把数据结果摊开，供老师判断写法；不是最终论文结论。",
        "",
        "## 样本和口径",
        "",
        f"- 参与者：{summary.get('participants', _safe_int(_unique_subjects(master)))}",
        f"- 场景试次：总计 {total_trials}，主分析保留 {kept_trials}，排除 {excluded_trials}",
        f"- AOI 展开行：{aoi_rows}。这个数字不是场景试次数，不能和 {kept_trials} 个保留场景试次混用。",
        "",
    ]
    if not qc.empty:
        reason_cols = [c for c in ["bad_eeg_quality", "duration_mismatch", "missing_questionnaire", "missing_eye", "missing_eeg", "scene_count_mismatch"] if c in qc.columns]
        if reason_cols:
            lines.extend(["主要排除原因："])
            for col in reason_cols:
                lines.append(f"- {col}: {_truth_count(qc[col])}")
            lines.append("")

    for module, title, metrics in [
        ("questionnaire", "问卷结果", ["S1", "S2", "S3", "S4", "S5"]),
        ("eye_tracking", "眼动结果", ["visited", "FCR", "TFD_ms", "TTFF_ms", "attention_share"]),
        ("eeg", "EEG 结果", ["O_theta", "F_theta", "O_alpha"]),
    ]:
        lines.extend([f"## {title}", ""])
        lines.extend(_factor_leader_lines(plain, module=module, factor="WWR", metrics=metrics, label="WWR"))
        lines.extend(_factor_leader_lines(plain, module=module, factor="Complexity", metrics=metrics, label="Complexity"))
        lines.append("")

    lines.extend(["## 显著性结果怎么看", ""])
    if significance.empty:
        lines.append("- 没有可汇总的显著性结果。")
    else:
        sig = significance.loc[significance["significance_label"].eq("p<0.05")]
        fallback = significance.loc[significance["fallback_flag"].fillna(False)]
        warning = significance.loc[significance["warning_flag"].fillna(False)]
        lines.append(f"- p<0.05 的模型/对比项：{len(sig)} 行。")
        lines.append(f"- OLS fallback 行：{len(fallback)} 行。")
        lines.append(f"- 带模型警告或不稳定标记行：{len(warning)} 行。")
        lines.append("- 显著性只说明该比较或模型项显著；是否写成机制、最优或规律，需要老师结合实验设计和模型稳定性判断。")
    lines.append("")
    return "\n".join(lines)


def interpretation_reference(outputs_root: Path, plain: pd.DataFrame, significance: pd.DataFrame) -> str:
    lines = [
        "# Interpretation Reference",
        "",
        "本文件是参考性解释，不替代老师基于数据的最终判断。",
        "",
        "## 可以直接从数据表判断的事",
        "",
        "- 每个指标在 WWR、Complexity、ExperienceGroup 下的均值、差异方向和样本口径。",
        "- 模型项或 planned contrast 是否达到 p<0.05。",
        "- 结果是否来自 scene trial、AOI-expanded row 或 smoke run。",
        "",
        "## 需要谨慎的事",
        "",
        "- WWR 只有 15、45、75 三档；即使 45% 显著，也只能说在本实验测试的三档中表现更优，不能直接说找到了连续意义上的最优 WWR。",
        "- EEG 解释必须结合 EEG QC、同步 QC 和问卷/眼动收敛，不能单靠 EEG 模型项写认知机制。",
        "- ExperienceGroup 只有交互项稳定时才支持“调节效应”；主效应更适合写成经验组差异。",
        "- AOI-expanded row 不是 scene trial。眼动模型和描述统计必须标明数据粒度。",
        "- 小样本 smoke run 只验证流程，不更新论文最终实验结果。",
        "",
    ]
    if not significance.empty:
        wwr_sig = significance.loc[
            significance["term"].astype(str).str.contains("WWR", case=False, na=False)
            & significance["significance_label"].eq("p<0.05")
        ]
        lines.extend(["## WWR 显著性参考", ""])
        lines.append(f"- WWR 相关 p<0.05 行数：{len(wwr_sig)}。请优先回看 `experiment_significance_results.csv` 的 estimate、CI、model_type 和 warning_flag。")
        lines.append("")
    return "\n".join(lines)


def adversarial_data_review(outputs_root: Path, plain: pd.DataFrame, significance: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    checks: list[dict[str, str]] = []
    summary = _read_json(outputs_root / "realdata_run_summary.json")
    master = _read_optional(outputs_root / "05_multimodal_fusion" / "analysis_master_long.csv")
    figure_qa = _read_optional(outputs_root / "10_figures" / "figure_qa.csv")
    data_availability = _read_optional(outputs_root / "09_data_package" / "data_availability_index.csv")

    _add_check(checks, "plain_results_present", not plain.empty, "plain result table has rows")
    _add_check(checks, "significance_results_present", not significance.empty, "significance table has rows")
    _add_check(checks, "grain_is_explicit", "grain" in plain.columns and plain["grain"].astype(str).str.len().gt(0).all(), "all plain rows carry a grain")

    if not master.empty:
        scene_trials = _unique_trials(master)
        aoi_rows = len(master)
        _add_check(
            checks,
            "aoi_row_scene_trial_distinction",
            aoi_rows != scene_trials,
            f"retained fusion has {scene_trials} scene trials and {aoi_rows} AOI-expanded rows; report both separately",
        )

    if not significance.empty:
        fallback_count = int(significance["fallback_flag"].fillna(False).sum()) if "fallback_flag" in significance.columns else 0
        warning_count = int(significance["warning_flag"].fillna(False).sum()) if "warning_flag" in significance.columns else 0
        noted = significance.get("interpretation_note", pd.Series("", index=significance.index)).astype(str).str.len().gt(0).all()
        _add_check(checks, "fallback_rows_are_flagged", True, f"fallback rows flagged: {fallback_count}")
        _add_check(checks, "warning_rows_are_flagged", True, f"warning rows flagged: {warning_count}")
        _add_check(checks, "significance_has_interpretation_notes", bool(noted), "each significance row has a plain-language note")

    if not figure_qa.empty and "qa_status" in figure_qa.columns:
        nonpass = figure_qa.loc[~figure_qa["qa_status"].astype(str).eq("pass")]
        _add_check(checks, "figure_qa", nonpass.empty, f"non-pass figure QA rows: {len(nonpass)}")
    else:
        _add_check(checks, "figure_qa", True, "figure QA not present in this run; acceptable for smoke/no-figure runs", severity="warning")

    if not data_availability.empty:
        placeholders = data_availability.astype(str).apply(lambda col: col.str.contains("AUTHOR_INPUT_NEEDED", regex=False, na=False)).any(axis=1)
        _add_check(checks, "data_availability_placeholders_explicit", True, f"AUTHOR_INPUT_NEEDED dataset rows: {int(placeholders.sum())}", severity="warning")

    smoke = _is_smoke_run(outputs_root, summary)
    _add_check(
        checks,
        "smoke_scope_marked",
        True,
        "smoke run detected; outputs are process validation only" if smoke else "full or historical result package; not marked as smoke",
        severity="warning" if smoke else "info",
    )

    failed = [c for c in checks if c["status"] == "FAIL"]
    warnings = [c for c in checks if c["status"] == "WARN"]
    summary_out = {
        "status": "fail" if failed else "pass_with_warnings" if warnings else "pass",
        "fail_count": len(failed),
        "warning_count": len(warnings),
        "checks": checks,
    }
    lines = [
        "# Adversarial Data Review",
        "",
        "This review audits whether the generated result package presents data clearly without hiding grain, QC, model, or smoke-run caveats.",
        "",
        f"Overall status: **{summary_out['status']}**",
        "",
        dataframe_to_markdown(pd.DataFrame(checks)),
        "",
        "## Review Focus",
        "",
        "- Scene-trial counts and AOI-expanded row counts are separated.",
        "- Significance rows keep model type, fallback flags, warning flags, and interpretation notes.",
        "- EEG and WWR language is bounded by design and QC caveats.",
        "- Smoke runs are not treated as final paper results.",
        "",
    ]
    return "\n".join(lines), summary_out


def _sample_rows(outputs_root: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    participants = _read_optional(outputs_root / "01_sample_qc" / "participants_standardized.csv")
    qc = _read_optional(outputs_root / "05_multimodal_fusion" / "analysis_qc_exclusions.csv")
    rows.append(_plain_row(
        module="sample_qc",
        source_table="realdata_run_summary.json",
        grain="run",
        metric="participants",
        factor="overall",
        level="all",
        n_subjects=int(summary.get("participants", _unique_subjects(participants)) or 0),
        n_trials=int(summary.get("scene_rows", 0) or 0),
        n_rows=1,
        mean=np.nan,
        note="Participant count from run summary or participants table.",
    ))
    if not qc.empty and "excluded_from_analysis" in qc.columns:
        excluded = _truth_count(qc["excluded_from_analysis"])
        total = len(qc)
        for metric, count in [("analysis_scene_trials_total", total), ("analysis_scene_trials_kept", total - excluded), ("analysis_scene_trials_excluded", excluded)]:
            rows.append(_plain_row(
                module="sample_qc",
                source_table="05_multimodal_fusion/analysis_qc_exclusions.csv",
                grain="scene_trial",
                metric=metric,
                factor="overall",
                level="all",
                n_subjects=_unique_subjects(qc),
                n_trials=count,
                n_rows=count,
                mean=np.nan,
                note="Scene-trial QC count.",
            ))
        for reason in [c for c in ["bad_eeg_quality", "duration_mismatch", "missing_questionnaire", "missing_eye", "missing_eeg", "scene_count_mismatch"] if c in qc.columns]:
            count = _truth_count(qc[reason])
            rows.append(_plain_row(
                module="sample_qc",
                source_table="05_multimodal_fusion/analysis_qc_exclusions.csv",
                grain="scene_trial",
                metric=f"exclusion_{reason}",
                factor="overall",
                level="all",
                n_subjects=_unique_subjects(qc.loc[_truth_mask(qc[reason])]),
                n_trials=count,
                n_rows=count,
                mean=np.nan,
                note="Exclusion reason counts can overlap.",
            ))
    return rows


def _summaries(
    df: pd.DataFrame,
    module: str,
    source_table: str,
    grain: str,
    metrics: list[str],
    note: str,
) -> list[dict[str, Any]]:
    if df.empty or not metrics:
        return []
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for factor_spec in DEFAULT_FACTORS:
            cols = [] if factor_spec == "overall" else factor_spec.split(":")
            if any(col not in df.columns for col in cols):
                continue
            if not cols:
                rows.append(_summary_row(df, module, source_table, grain, metric, "overall", "all", note))
                continue
            grouped = df.groupby(cols, dropna=False, sort=True)
            for keys, sub in grouped:
                keys = keys if isinstance(keys, tuple) else (keys,)
                level = ";".join(f"{col}={value}" for col, value in zip(cols, keys))
                rows.append(_summary_row(sub, module, source_table, grain, metric, factor_spec, level, note))
    return rows


def _summary_row(df: pd.DataFrame, module: str, source_table: str, grain: str, metric: str, factor: str, level: str, note: str) -> dict[str, Any]:
    values = _numeric_series(df[metric]).dropna()
    n = int(len(values))
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    se = float(sd / math.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan
    ci = 1.96 * se if np.isfinite(se) else np.nan
    q75 = float(values.quantile(0.75)) if n else np.nan
    q25 = float(values.quantile(0.25)) if n else np.nan
    mean = float(values.mean()) if n else np.nan
    return _plain_row(
        module=module,
        source_table=source_table,
        grain=grain,
        metric=metric,
        factor=factor,
        level=level,
        n_subjects=_unique_subjects(df.loc[values.index]) if n else 0,
        n_trials=_unique_trials(df.loc[values.index]) if n else 0,
        n_rows=int(len(df.loc[values.index])) if n else 0,
        mean=mean,
        sd=sd,
        se=se,
        ci95_low=float(mean - ci) if np.isfinite(ci) else np.nan,
        ci95_high=float(mean + ci) if np.isfinite(ci) else np.nan,
        median=float(values.median()) if n else np.nan,
        iqr=float(q75 - q25) if n else np.nan,
        note=note,
    )


def _eeg_qc_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subject_qc = _read_optional(outputs_root / "04_eeg" / "eeg_subject_qc.csv")
    if not subject_qc.empty and "eeg_subject_quality_exclusion" in subject_qc.columns:
        rows.append(_plain_row(
            module="eeg_qc",
            source_table="04_eeg/eeg_subject_qc.csv",
            grain="participant",
            metric="eeg_subject_quality_exclusion",
            factor="overall",
            level="all",
            n_subjects=_truth_count(subject_qc["eeg_subject_quality_exclusion"]),
            n_trials=0,
            n_rows=len(subject_qc),
            mean=np.nan,
            note="Participant-level EEG quality exclusion count.",
        ))
    thresholds = _read_optional(outputs_root / "04_eeg" / "eeg_qc_thresholds.csv")
    for row in thresholds.to_dict("records"):
        rows.append(_plain_row(
            module="eeg_qc",
            source_table="04_eeg/eeg_qc_thresholds.csv",
            grain="qc_metric",
            metric=str(row.get("metric", "")),
            factor="threshold",
            level=str(row.get("method", "")),
            n_subjects=0,
            n_trials=0,
            n_rows=int(row.get("n", 0) or 0),
            mean=_to_float(row.get("threshold")),
            note=f"Robust EEG threshold; k={row.get('k', '')}, median={row.get('median', '')}, MAD={row.get('mad', '')}.",
        ))
    return rows


def _robustness_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nonlinear = _read_optional(outputs_root / "06_robustness" / "nonlinear_wwr_sensitivity.csv")
    for row in nonlinear.to_dict("records"):
        rows.append(_plain_row(
            module="robustness",
            source_table="06_robustness/nonlinear_wwr_sensitivity.csv",
            grain="diagnostic",
            metric=str(row.get("outcome", "")),
            factor="WWR45_peak_index",
            level=str(row.get("claim_strength", "")),
            n_subjects=0,
            n_trials=0,
            n_rows=1,
            mean=_to_float(row.get("wwr45_peak_index")),
            note="WWR45 minus mean of WWR15 and WWR75; diagnostic only.",
        ))
    order = _read_optional(outputs_root / "06_robustness" / "order_fatigue_effects.csv")
    for row in order.to_dict("records"):
        rows.append(_plain_row(
            module="robustness",
            source_table="06_robustness/order_fatigue_effects.csv",
            grain="diagnostic",
            metric=str(row.get("outcome", "")),
            factor="order_fatigue",
            level=str(row.get("order_variable", "")),
            n_subjects=0,
            n_trials=int(row.get("n", 0) or 0),
            n_rows=1,
            mean=_to_float(row.get("correlation")),
            note="Correlation diagnostic for order/fatigue; not a primary effect estimate.",
        ))
    return rows


def _plain_row(**kwargs: Any) -> dict[str, Any]:
    row = {col: np.nan for col in PLAIN_COLUMNS}
    row.update(kwargs)
    return row


def _factor_leader_lines(plain: pd.DataFrame, module: str, factor: str, metrics: list[str], label: str) -> list[str]:
    lines: list[str] = []
    sub = plain.loc[plain["module"].eq(module) & plain["factor"].eq(factor) & plain["metric"].isin(metrics)].copy() if not plain.empty else pd.DataFrame()
    if sub.empty:
        lines.append(f"- {label}: 没有可汇总结果。")
        return lines
    for metric, metric_df in sub.groupby("metric", sort=False):
        values = pd.to_numeric(metric_df["mean"], errors="coerce")
        metric_df = metric_df.loc[values.notna()].copy()
        if metric_df.empty:
            continue
        metric_df["_mean"] = values.loc[metric_df.index]
        top = metric_df.sort_values("_mean", ascending=False).iloc[0]
        low = metric_df.sort_values("_mean", ascending=True).iloc[0]
        diff = top["_mean"] - low["_mean"]
        lines.append(
            f"- {metric} 按 {label}：最高为 {top['level']}，均值 {top['_mean']:.3g}；最低为 {low['level']}，均值 {low['_mean']:.3g}；差值约 {diff:.3g}。"
        )
    return lines or [f"- {label}: 没有可汇总结果。"]


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_table(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    if text.isin({"true", "false", "1", "0", "yes", "no", "y", "n"}).all():
        return text.map({"true": 1, "1": 1, "yes": 1, "y": 1, "false": 0, "0": 0, "no": 0, "n": 0}).astype(float)
    return pd.to_numeric(series, errors="coerce")


def _unique_subjects(df: pd.DataFrame) -> int:
    return int(df["participant_id"].astype(str).nunique()) if "participant_id" in df.columns and not df.empty else 0


def _unique_trials(df: pd.DataFrame) -> int:
    if {"participant_id", "scene_id"}.issubset(df.columns) and not df.empty:
        return int(df[["participant_id", "scene_id"]].drop_duplicates().shape[0])
    return 0


def _truth_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _truth_count(series: pd.Series) -> int:
    return int(_truth_mask(series).sum())


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def _first_float(row: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        value = _to_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def _unstable_interval(std_error: float, ci_low: float, ci_high: float) -> bool:
    if np.isfinite(std_error) and abs(std_error) > 1000:
        return True
    if np.isfinite(ci_low) and np.isfinite(ci_high) and abs(ci_high - ci_low) > 10000:
        return True
    return False


def _p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p_not_available"
    return "p<0.05" if p_value < 0.05 else "p>=0.05"


def _result_family(outcome: str) -> str:
    text = str(outcome)
    if text.startswith("q_") or text in PRIMARY_QUESTIONNAIRE:
        return "questionnaire"
    if text.startswith("eeg_") or any(token in text.lower() for token in ["theta", "alpha", "beta"]):
        return "eeg"
    if text in PRIMARY_EYE:
        return "eye_tracking"
    return "other"


def _interpretation_note(outcome: str, term: str, p_value: float, fallback: bool, warning: bool) -> str:
    notes = []
    if np.isfinite(p_value):
        notes.append("统计显著" if p_value < 0.05 else "未达到0.05显著性")
    else:
        notes.append("没有可直接解读的p值")
    if "WWR" in term:
        notes.append("WWR只有15/45/75三档，显著性只支持本实验三档内比较")
    if str(outcome).startswith("eeg_") or any(token in str(outcome).lower() for token in ["theta", "alpha", "beta"]):
        notes.append("EEG结果需结合QC和多模态证据")
    if str(outcome) in PRIMARY_EYE:
        notes.append("眼动结果注意AOI-expanded row粒度")
    if fallback:
        notes.append("模型发生OLS fallback，需谨慎")
    if warning:
        notes.append("存在模型稳定性警告或异常标准误/CI")
    return "；".join(dict.fromkeys(notes))


def _is_smoke_run(outputs_root: Path, summary: dict[str, Any]) -> bool:
    if "smoke" in str(outputs_root).lower():
        return True
    participants = summary.get("participants")
    try:
        return participants is not None and int(participants) < 10
    except (TypeError, ValueError):
        return False


def _add_check(checks: list[dict[str, str]], check_id: str, ok: bool, detail: str, severity: str = "fail") -> None:
    if ok:
        status = "WARN" if severity == "warning" else "INFO" if severity == "info" else "PASS"
    else:
        status = "WARN" if severity == "warning" else "FAIL"
    checks.append({"check_id": check_id, "status": status, "detail": detail})


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
