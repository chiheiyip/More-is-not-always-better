from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_analysis.teacher.contracts import (
    build_modality_registry,
    canonicalize_trials,
    trial_contract_audit,
)
from paper_analysis.teacher.state import (
    StageBlockedError,
    require_approval,
    stage_fingerprint,
    write_approval_template,
    write_input_hashes,
    write_run_manifest,
)
from paper_analysis.utils.io import is_truthy, read_table, write_table, write_text


FORBIDDEN_PRIMARY_TERMS = (
    "WWR:Complexity:ExperienceGroup",
    "WWR:OrderGroup",
    "Complexity:OrderGroup",
)


def _contract_value_equal(observed: object, expected: object) -> bool:
    try:
        return bool(np.isclose(float(observed), float(expected)))
    except (TypeError, ValueError):
        return str(observed).strip().lower() == str(expected).strip().lower()


def _invoke_r(rscript: str, script: Path, arguments: list[str], required: bool) -> bool:
    executable = shutil.which(rscript) if not Path(rscript).exists() else rscript
    if not executable:
        if required:
            raise StageBlockedError(
                f"Rscript is unavailable ({rscript}). Install R and run "
                "`renv::restore()` in analysis/r before formal inference."
            )
        return False
    subprocess.run([str(executable), str(script), *arguments], check=True)
    return True


def _package_r_csv_outputs(outdir: Path) -> None:
    for csv_path in outdir.glob("*.csv"):
        if csv_path.name.endswith("_model_input.csv"):
            continue
        write_table(read_table(csv_path), csv_path.with_suffix(".xlsx"))


def _normalize_eeg(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eeg_path = Path(config["eeg"]["trial_file"])
    participant_path = Path(config["participant_information"])
    raw = read_table(eeg_path)
    trials = canonicalize_trials(raw)
    registry = build_modality_registry(
        participant_path, eeg_participants=trials["Participant"]
    )
    covariates = [
        "Participant", "Gender", "ExperienceGroup", "OrderGroup", "IncludeEEGValid"
    ]
    for column in ("Gender", "ExperienceGroup", "OrderGroup", "IncludeEEGValid"):
        if column in trials:
            covariates.remove(column)
    trials = trials.merge(
        registry[covariates], on="Participant", how="left", validate="many_to_one"
    )
    return trials, registry


def _scene_qc_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the formal scene-level EEG eligibility mask.

    The 42-person registry defines the structural cohort (504 expected trials).
    Scene-level quality flags then define the model cohort (471 trials in the
    current real data). Missing quality columns are treated as a contract error,
    not as implicit passes.
    """
    required = {"bad_eeg_quality", "eeg_subject_quality_exclusion"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise StageBlockedError(
            "EEG scene-level QC columns are missing from the trial file: "
            f"{missing}. Rebuild the EEG trial export before formal modeling."
        )
    return (
        ~frame["bad_eeg_quality"].map(is_truthy)
        & ~frame["eeg_subject_quality_exclusion"].map(is_truthy)
    )


def _exclusion_reason(frame: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype="object")
    invalid_subject = ~frame["IncludeEEGValid"].map(is_truthy)
    bad_scene = frame.get(
        "bad_eeg_quality", pd.Series(False, index=frame.index)
    ).map(is_truthy)
    subject_qc = frame.get(
        "eeg_subject_quality_exclusion", pd.Series(False, index=frame.index)
    ).map(is_truthy)
    reasons.loc[invalid_subject] = "IncludeEEGValid_false"
    reasons.loc[bad_scene] = reasons.loc[bad_scene].where(
        reasons.loc[bad_scene].eq(""),
        reasons.loc[bad_scene] + ";",
    ) + "bad_eeg_quality"
    reasons.loc[subject_qc] = reasons.loc[subject_qc].where(
        reasons.loc[subject_qc].eq(""),
        reasons.loc[subject_qc] + ";",
    ) + "eeg_subject_quality_exclusion"
    return reasons


def _metric_columns(frame: pd.DataFrame, metrics: list[str]) -> tuple[list[str], list[str]]:
    relative: list[str] = []
    absolute: list[str] = []
    for metric in metrics:
        candidates = (
            f"{metric}_relative",
            f"relative_{metric}",
            f"eeg_{metric}_relative",
        )
        match = next((name for name in candidates if name in frame.columns), None)
        if match:
            relative.append(match)
        abs_candidates = (
            f"{metric}_absolute",
            f"absolute_{metric}",
            f"eeg_{metric}_absolute",
            f"eeg_abs_{metric}",
            metric,
            f"eeg_{metric}",
        )
        match_abs = next((name for name in abs_candidates if name in frame.columns), None)
        if match_abs:
            absolute.append(match_abs)
    return relative, absolute


def _sequence_tables(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence = (
        trials.sort_values(["Participant", "GlobalTrialOrder"])
        [["Participant", "OrderGroup", "GlobalTrialOrder", "Block",
          "PositionWithinBlock", "SceneID", "WWR", "Complexity",
          "PreviousWWR", "PreviousComplexity"]]
        .copy()
    )
    balance = (
        trials.groupby(
            ["OrderGroup", "Block", "PositionWithinBlock", "WWR", "Complexity"],
            dropna=False,
        )
        .size()
        .rename("NTrials")
        .reset_index()
    )
    return sequence, balance


def run_eeg_order(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
    r_required: bool = True,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    eeg_config = config["eeg"]
    eeg_path = Path(eeg_config["trial_file"])
    audit_path = Path(eeg_config["preprocessing_audit_file"])
    order_inputs = [eeg_path, audit_path, config["participant_information"]]
    fingerprint = stage_fingerprint("eeg-order", eeg_config, order_inputs)
    all_trials, registry = _normalize_eeg(config)
    structural_trials = all_trials.loc[
        all_trials["IncludeEEGValid"].map(is_truthy)
    ].copy()
    audit = trial_contract_audit(structural_trials)
    scene_qc = _scene_qc_mask(structural_trials)
    trials = structural_trials.loc[scene_qc].copy()
    invalid_trials = all_trials.loc[
        ~all_trials.index.isin(trials.index)
    ].copy()
    if not invalid_trials.empty:
        invalid_trials["ExclusionReason"] = _exclusion_reason(invalid_trials)
    required_design = {
        "Participant", "WWR", "Complexity", "Gender", "ExperienceGroup",
        "OrderGroup", "Block", "PositionWithinBlock",
    }
    missing_design = sorted(required_design - set(trials.columns))
    sequence, balance = (
        _sequence_tables(structural_trials)
        if not missing_design
        else (pd.DataFrame(), pd.DataFrame())
    )
    order_groups = (
        sorted(structural_trials["OrderGroup"].dropna().astype(str).unique())
        if "OrderGroup" in structural_trials else []
    )
    blockers: list[str] = []
    if missing_design:
        blockers.append(f"EEG trial file is missing design columns: {missing_design}.")
    if not audit["ContractPass"].all():
        blockers.append("Not every EEG participant has 12 unique trials in two 6-trial blocks.")
    if order_groups != ["new order2", "order1", "order2"]:
        blockers.append(f"Expected three OrderGroup levels; observed {order_groups}.")
    if not bool(eeg_config.get("preprocessing_confirmed", False)):
        blockers.append("EEG preprocessing parameters are not confirmed in config.")
    if not audit_path.exists():
        blockers.append(f"Preprocessing audit file is missing: {audit_path}")
    expected_preprocessing = eeg_config.get("preprocessing_parameters", {})
    if audit_path.exists() and expected_preprocessing:
        preprocessing_audit = read_table(audit_path)
        if not {"Parameter", "Value"}.issubset(preprocessing_audit.columns):
            blockers.append(
                "Preprocessing audit must contain Parameter and Value columns."
            )
        else:
            observed = (
                preprocessing_audit.drop_duplicates("Parameter", keep="last")
                .set_index("Parameter")["Value"].astype(str).to_dict()
            )
            mismatches = {
                key: {"expected": str(value), "observed": observed.get(key)}
                for key, value in expected_preprocessing.items()
                if key not in observed
                or not _contract_value_equal(observed.get(key), value)
            }
            if mismatches:
                blockers.append(
                    "Configured preprocessing parameters differ from the audit: "
                    + json.dumps(mismatches, ensure_ascii=False)
                )
    outputs = {
        "input_hashes": write_input_hashes(out, order_inputs),
        "audit": write_table(audit, out / "01_eeg_data_audit.xlsx"),
        "sequence": write_table(sequence, out / "02_order_sequence_check.xlsx"),
        "balance": write_table(balance, out / "03_condition_position_frequency.xlsx"),
        "registry": write_table(registry, out / "04_modality_union_registry.xlsx"),
        "exclusions": write_table(
            invalid_trials,
            out / "04b_EEG_exclusion_log.xlsx",
        ),
        "blockers": write_text(
            "\n".join(["EEG ORDER REVIEW", *[f"- {x}" for x in blockers]]) + "\n",
            out / "00_EEG_ORDER_REVIEW.txt",
        ),
    }
    write_table(audit, out / "eeg_data_audit.csv")
    write_table(sequence, out / "order_sequence_check.csv")
    write_table(balance, out / "condition_position_frequency.csv")
    if blockers:
        write_run_manifest(
            out, stage="eeg-order", fingerprint=fingerprint,
            config_path=config_path, arguments={"outdir": str(out)},
            repo_root=repo_root, extra={"status": "blocked", "blockers": blockers},
        )
        raise StageBlockedError("EEG order stage blocked: " + " | ".join(blockers))
    core = [str(v) for v in eeg_config.get("core_metrics", [])]
    relative, _ = _metric_columns(trials, core)
    if not core or len(relative) != len(core):
        blockers = [
            "EEG order analysis requires explicit relative-power columns for "
            f"all pre-registered core metrics. Registered={core}; found={relative}"
        ]
        write_run_manifest(
            out, stage="eeg-order", fingerprint=fingerprint,
            config_path=config_path, arguments={"outdir": str(out)},
            repo_root=repo_root, extra={"status": "blocked", "blockers": blockers},
        )
        raise StageBlockedError(blockers[0])
    model_input = out / "eeg_order_model_input.csv"
    trials.to_csv(model_input, index=False, encoding="utf-8-sig")
    r_script = Path(repo_root) / "analysis" / "r" / "eeg_order_analysis.R"
    r_ran = _invoke_r(
        str(config.get("rscript", "Rscript")), r_script,
        [str(model_input), str(out), ",".join(relative)], r_required,
    )
    if r_ran:
        _package_r_csv_outputs(out)
    if r_ran:
        try:
            from paper_analysis.teacher.reporting import (
                save_histogram,
                write_markdown_report,
            )
        except ImportError as exc:
            raise StageBlockedError(
                "Teacher Markdown report packaging requires pandas and matplotlib."
            ) from exc
        comparison_path = out / "06_eeg_model0_model1_comparison.csv"
        comparison = (
            read_table(comparison_path) if comparison_path.exists() else pd.DataFrame()
        )
        outputs["report_md"] = write_markdown_report(
            out / "11_EEG_order_effect_report.md",
            title="EEG Order and Temporal Effects",
            paragraphs=[
                "Model 0 and time/order-adjusted Model 1 are reported side by side.",
                "Direction, interval, and estimate change are interpreted without "
                "using a p=0.05 crossing as the sole criterion.",
            ],
            tables=[("Model comparison", comparison), ("Position balance", balance)],
        )
        report_cn = (
            "# EEG时间与顺序效应分析\n\n"
            f"- 结构审核：{structural_trials['Participant'].nunique()} 人、"
            f"{len(structural_trials)} 试次。\n"
            f"- 场景级QC建模：{trials['Participant'].nunique()} 人、"
            f"{len(trials)} 试次。\n"
            "- Model 0 与加入 Block、PositionWithinBlockCentered、"
            "三个 OrderGroup 的 Model 1 并列报告。\n"
            "- PreviousWWR 与 PreviousComplexity 仅在同一 Block 的 "
            "Position 2–6 中定义。\n"
            "- 结论依据方向、估计幅度、95% CI 与 CR2，不以是否跨过 "
            "p=0.05 作为唯一标准，也不声称完全排除顺序效应。\n"
        )
        outputs["report_cn_md"] = write_text(
            report_cn, out / "eeg_temporal_analysis_report.md"
        )
        outputs["report_en_md"] = write_text(
            "# EEG temporal and order analysis\n\n"
            f"The structural audit included "
            f"{structural_trials['Participant'].nunique()} participants and "
            f"{len(structural_trials)} trials; scene-level QC retained "
            f"{trials['Participant'].nunique()} participants and {len(trials)} "
            "trials for modeling. Model 0 and the time/order-adjusted Model 1 "
            "are compared by direction, magnitude and confidence intervals. "
            "The analysis does not claim that order effects were completely "
            "excluded.\n",
            out / "eeg_temporal_analysis_report_en.md",
        )
    if not r_ran:
        outputs["summary"] = write_text(
            "EEG ORDER SUMMARY\n"
            "Python audits completed; formal R inference was skipped.\n",
            out / "12_eeg_order_summary.txt",
        )
        outputs["manifest"] = write_run_manifest(
            out, stage="eeg-order", fingerprint=fingerprint,
            config_path=config_path, arguments={"outdir": str(out)},
            repo_root=repo_root, extra={"status": "python_complete_r_skipped"},
        )
        return outputs
    completion = out / "eeg_order_completed.txt"
    completion.write_text(json.dumps({
        "approved": True,
        "stage": "eeg-order",
        "stage_fingerprint": fingerprint,
        "approved_by": "pipeline_after_successful_order_analysis",
        "approved_at": pd.Timestamp.utcnow().isoformat(),
        "approved_triggers": [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["completion"] = completion
    outputs["summary"] = write_text(
        "EEG ORDER SUMMARY\n"
        f"Structural participants: {structural_trials['Participant'].nunique()}\n"
        f"Structural trials audited: {len(structural_trials)}\n"
        f"Scene-QC model participants: {trials['Participant'].nunique()}\n"
        f"Scene-QC model trials: {len(trials)}\n"
        f"Order groups: {', '.join(order_groups)}\n"
        "Model 0, Model 1, prior-scene and Block 1 analyses completed.\n",
        out / "12_eeg_order_summary.txt",
    )
    outputs["manifest"] = write_run_manifest(
        out, stage="eeg-order", fingerprint=fingerprint,
        config_path=config_path, arguments={"outdir": str(out)},
        repo_root=repo_root,
        extra={"status": "complete" if r_ran else "python_complete_r_skipped"},
    )
    return outputs


def run_eeg_primary(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
    r_required: bool = True,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    eeg_config = config["eeg"]
    order_dir = Path(eeg_config["order_stage_dir"])
    order_manifest_path = order_dir / "run_manifest.json"
    if not order_manifest_path.exists():
        raise StageBlockedError("EEG primary requires a completed EEG order stage.")
    order_manifest = json.loads(order_manifest_path.read_text(encoding="utf-8"))
    current_order_fingerprint = stage_fingerprint(
        "eeg-order", eeg_config,
        [
            eeg_config["trial_file"],
            eeg_config["preprocessing_audit_file"],
            config["participant_information"],
        ],
    )
    if current_order_fingerprint != str(order_manifest["stage_fingerprint"]):
        raise StageBlockedError(
            "EEG inputs or preprocessing contract changed; rerun EEG order analysis."
        )
    require_approval(
        order_dir / "eeg_order_completed.txt",
        fingerprint=str(order_manifest["stage_fingerprint"]),
        stage="eeg-order",
    )
    eeg_path = Path(eeg_config["trial_file"])
    primary_inputs = [eeg_path, order_dir / "eeg_order_completed.txt"]
    fingerprint = stage_fingerprint("eeg-primary", eeg_config, primary_inputs)
    all_trials, registry = _normalize_eeg(config)
    structural_trials = all_trials.loc[
        all_trials["IncludeEEGValid"].map(is_truthy)
    ].copy()
    trials = structural_trials.loc[_scene_qc_mask(structural_trials)].copy()
    if trials.empty:
        raise StageBlockedError(
            "No EEG trials remain after participant eligibility and scene-level QC."
        )
    core = [str(v) for v in eeg_config.get("core_metrics", [])]
    secondary = [str(v) for v in eeg_config.get("secondary_metrics", [])]
    supplemental = [str(v) for v in eeg_config.get("supplemental_metrics", [])]
    relative, absolute = _metric_columns(trials, core)
    secondary_relative, _ = _metric_columns(trials, secondary)
    supplemental_relative, _ = _metric_columns(trials, supplemental)
    if not core:
        raise StageBlockedError("eeg.core_metrics must explicitly pre-register at least one metric.")
    if len(relative) != len(core):
        missing = [metric for metric in core if not any(metric in c for c in relative)]
        raise StageBlockedError(f"Pre-registered relative-power core metrics are missing: {missing}")
    if len(absolute) != len(core):
        raise StageBlockedError(
            "Every pre-registered core EEG metric requires an absolute-power "
            "column for the fixed log10 sensitivity analysis."
        )
    if len(secondary_relative) != len(secondary):
        raise StageBlockedError(
            "One or more pre-registered secondary relative-power metrics are missing."
        )
    if len(supplemental_relative) != len(supplemental):
        raise StageBlockedError(
            "One or more pre-registered supplemental relative-power metrics are missing."
        )
    if str(eeg_config.get("primary_measure", "")).strip() != "relative_power":
        raise StageBlockedError("eeg.primary_measure must remain relative_power.")
    log_absolute: list[str] = []
    for column in absolute:
        output_column = f"log10_{column}"
        values = pd.to_numeric(trials[column], errors="coerce")
        trials[output_column] = np.where(values > 0, np.log10(values), np.nan)
        log_absolute.append(output_column)
    absolute = log_absolute
    classification = pd.DataFrame({
        "Metric": core + secondary + supplemental,
        "Family": (
            ["Core"] * len(core)
            + ["Secondary"] * len(secondary)
            + ["Supplemental"] * len(supplemental)
        ),
        "PrimaryMeasure": ["relative_power"] * (
            len(core) + len(secondary) + len(supplemental)
        ),
        "AbsolutePowerRole": ["sensitivity"] * (
            len(core) + len(secondary) + len(supplemental)
        ),
        "DefaultClaimGrade": (
            ["pending_robustness"] * len(core)
            + ["Exploratory"] * len(secondary)
            + ["Exploratory"] * len(supplemental)
        ),
    })
    model_contract = pd.DataFrame([
        {
            "Model": "teacher_primary_lmm",
            "Formula": (
                "Outcome ~ WWR*Complexity + WWR*ExperienceGroup + "
                "Complexity*ExperienceGroup + Gender + Block + "
                "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
            ),
            "Forbidden": ";".join(FORBIDDEN_PRIMARY_TERMS),
            "PrimaryMeasure": "relative_power",
            "BootstrapIterations": int(eeg_config.get("bootstrap_iterations", 5000)),
        }
    ])
    outputs = {
        "input_hashes": write_input_hashes(out, primary_inputs),
        "classification": write_table(classification, out / "01_eeg_outcome_classification.xlsx"),
        "model_contract": write_table(model_contract, out / "02_eeg_model_contract.xlsx"),
        "registry": write_table(registry, out / "03_modality_union_registry.xlsx"),
        "sample_flow": write_table(
            pd.DataFrame([
                {
                    "Stage": "structural_42_person_cohort",
                    "Participants": structural_trials["Participant"].nunique(),
                    "Trials": len(structural_trials),
                },
                {
                    "Stage": "scene_qc_model_cohort",
                    "Participants": trials["Participant"].nunique(),
                    "Trials": len(trials),
                },
            ]),
            out / "03b_eeg_sample_flow.xlsx",
        ),
        "data_audit": write_table(
            pd.DataFrame([
                {
                    "AuditItem": "structural_cohort",
                    "Participants": structural_trials["Participant"].nunique(),
                    "Trials": len(structural_trials),
                    "ExpectedTrialsPerParticipant": 12,
                    "ExpectedBlocks": 2,
                    "Pass": (
                        structural_trials["Participant"].nunique() == 42
                        and len(structural_trials) == 504
                    ),
                },
                {
                    "AuditItem": "scene_qc_model_cohort",
                    "Participants": trials["Participant"].nunique(),
                    "Trials": len(trials),
                    "ExpectedTrialsPerParticipant": np.nan,
                    "ExpectedBlocks": 2,
                    "Pass": len(trials) > 0,
                },
            ]),
            out / "01_eeg_data_audit.xlsx",
        ),
        "preprocessing_audit": write_table(
            read_table(eeg_config["preprocessing_audit_file"]),
            out / "02_eeg_preprocessing_audit.xlsx",
        ),
        "exclusion_log": write_table(
            structural_trials.loc[
                ~_scene_qc_mask(structural_trials)
            ].assign(
                ExclusionReason=lambda frame: _exclusion_reason(frame)
            ),
            out / "03_eeg_exclusion_log.xlsx",
        ),
        "trial_completeness": write_table(
            trials.groupby("Participant", dropna=False)
            .agg(
                RetainedTrials=("GlobalTrialOrder", "nunique"),
                Block1Trials=("Block", lambda s: int((s == 1).sum())),
                Block2Trials=("Block", lambda s: int((s == 2).sum())),
            )
            .reset_index(),
            out / "04_eeg_trial_completeness.xlsx",
        ),
        "order_structure": write_table(
            trial_contract_audit(structural_trials),
            out / "05_eeg_order_structure_check.xlsx",
        ),
        "variable_dictionary": write_table(
            pd.DataFrame([
                {
                    "Variable": column,
                    "DType": str(trials[column].dtype),
                    "Role": (
                        "core_relative_primary" if column in relative
                        else "secondary_relative" if column in secondary_relative
                        else "supplemental_relative" if column in supplemental_relative
                        else "log10_absolute_sensitivity" if column in absolute
                        else "design_or_qc"
                    ),
                }
                for column in trials.columns
            ]),
            out / "06_eeg_variable_dictionary.xlsx",
        ),
    }
    descriptive = []
    for outcome in [*relative, *secondary_relative, *supplemental_relative]:
        for (wwr, complexity), sub in trials.groupby(
            ["WWR", "Complexity"], dropna=False
        ):
            values = pd.to_numeric(sub[outcome], errors="coerce").dropna()
            descriptive.append({
                "Outcome": outcome,
                "WWR": wwr,
                "Complexity": complexity,
                "NTrials": len(values),
                "NParticipants": sub.loc[
                    values.index, "Participant"
                ].nunique(),
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "Median": values.median(),
                "IQR": values.quantile(.75) - values.quantile(.25),
                "CI95Low": (
                    values.mean() - 1.96 * values.sem()
                    if len(values) > 1 else np.nan
                ),
                "CI95High": (
                    values.mean() + 1.96 * values.sem()
                    if len(values) > 1 else np.nan
                ),
            })
    outputs["descriptives"] = write_table(
        pd.DataFrame(descriptive), out / "07_eeg_descriptive_statistics.xlsx"
    )
    model_input = out / "eeg_primary_model_input.csv"
    trials.to_csv(model_input, index=False, encoding="utf-8-sig")
    r_script = Path(repo_root) / "analysis" / "r" / "eeg_primary_analysis.R"
    r_ran = _invoke_r(
        str(config.get("rscript", "Rscript")), r_script,
        [
            str(model_input), str(out), ",".join(relative),
            ",".join(absolute),
            str(int(eeg_config.get("bootstrap_iterations", 5000))),
            ",".join(secondary_relative),
            ",".join(supplemental_relative),
        ],
        r_required,
    )
    if r_ran:
        _package_r_csv_outputs(out)
        crossmodal_base = [
            str(v) for v in eeg_config.get(
                "crossmodal_metrics", ["O_theta", "O_alpha"]
            )
        ]
        if not 1 <= len(crossmodal_base) <= 2:
            raise StageBlockedError(
                "eeg.crossmodal_metrics must pre-register exactly 1-2 metrics."
            )
        crossmodal_metrics, _ = _metric_columns(trials, crossmodal_base)
        if len(crossmodal_metrics) != len(crossmodal_base):
            raise StageBlockedError(
                "Pre-registered cross-modal relative EEG metrics are missing."
            )
        eye_stage2 = Path(config.get("eye", {}).get("stage2_dir", ""))
        eye_trial_path = eye_stage2 / "06_trial_level_eye_tracking_data.xlsx"
        if eye_trial_path.is_file():
            eye_trials = read_table(eye_trial_path)
            eye_trials = eye_trials.loc[
                eye_trials["IncludedPrimary"].map(is_truthy)
            ].copy()
            crossmodal = eye_trials.merge(
                trials[[
                    "Participant", "GlobalTrialOrder",
                    *crossmodal_metrics,
                ]],
                on=["Participant", "GlobalTrialOrder"],
                how="inner",
                validate="one_to_one",
            )
            crossmodal_input = out / "eeg_crossmodal_model_input.csv"
            crossmodal.to_csv(
                crossmodal_input, index=False, encoding="utf-8-sig"
            )
            outputs["crossmodal_counts"] = write_table(
                pd.DataFrame([{
                    "Participants": crossmodal["Participant"].nunique(),
                    "Trials": len(crossmodal),
                    "IntersectionGrain": "Participant × GlobalTrialOrder",
                    "EEGMetrics": ",".join(crossmodal_metrics),
                }]),
                out / "19a_eeg_crossmodal_sample_counts.xlsx",
            )
            crossmodal_script = (
                Path(repo_root) / "analysis" / "r"
                / "eeg_crossmodal_analysis.R"
            )
            _invoke_r(
                str(config.get("rscript", "Rscript")),
                crossmodal_script,
                [
                    str(crossmodal_input), str(out),
                    ",".join(crossmodal_metrics),
                ],
                r_required,
            )
            _package_r_csv_outputs(out)
    if r_ran:
        try:
            from paper_analysis.teacher.reporting import write_markdown_report
        except ImportError as exc:
            raise StageBlockedError(
                "Teacher Markdown report packaging requires pandas and matplotlib."
            ) from exc
        grade_path = out / "19_eeg_robustness_classification.csv"
        grades = read_table(grade_path) if grade_path.exists() else pd.DataFrame()
        diagnostics_path = out / "18_eeg_model_diagnostics.csv"
        diagnostics = (
            read_table(diagnostics_path)
            if diagnostics_path.is_file() else pd.DataFrame()
        )
        preprocessing = read_table(eeg_config["preprocessing_audit_file"])
        outputs["diagnostics_md"] = write_markdown_report(
            out / "21_eeg_model_diagnostics.md",
            title="EEG Model Diagnostics",
            paragraphs=[
                "Failed mixed models and failed Bootstrap replicates are "
                "reported explicitly. No OLS or alternative model is used as "
                "a silent replacement.",
            ],
            tables=[("Diagnostics", diagnostics)],
        )
        outputs["methods_md"] = write_markdown_report(
            out / "22_eeg_methods_report.md",
            title="EEG Analysis Methods",
            paragraphs=[
                "Relative power is primary; raw absolute power is descriptive "
                "and log10 absolute power is a fixed sensitivity analysis.",
                "The structural audit cohort and scene-QC model cohort are "
                "reported separately.",
            ],
            tables=[
                ("Preprocessing and extraction audit", preprocessing),
                ("Outcome registration", classification),
            ],
        )
        outputs["report_en_md"] = write_markdown_report(
            out / "23_eeg_results_report_en.md",
            title="Teacher-priority EEG Primary Analysis",
            paragraphs=[
                "Relative power is primary; log10 absolute power is sensitivity.",
                "Core outcomes alone enter the primary BH-FDR families.",
                "Bootstrap failures are retained and reported, never replaced.",
            ],
            tables=[
                ("Outcome registration", classification),
                ("Robustness classification", grades),
            ],
        )
        outputs["report_cn_md"] = write_markdown_report(
            out / "24_eeg_results_report_cn.md",
            title="老师规范 EEG 正式主分析结果",
            paragraphs=[
                f"结构审核样本为 {structural_trials['Participant'].nunique()} "
                f"人/{len(structural_trials)} 试次；场景级 QC 后正式模型为 "
                f"{trials['Participant'].nunique()} 人/{len(trials)} 试次。",
                "主要度量为 relative power，log10 absolute power 仅作敏感性。",
                "ExperienceGroup 严格沿用代码仓库 Q1.4 分类，"
                "未使用 ExerciseFrequency。",
            ],
            tables=[
                ("结局预注册", classification),
                ("证据分级", grades),
            ],
        )
    order_counts = (
        structural_trials.groupby("OrderGroup")["Participant"]
        .nunique().to_dict()
    )
    grade_path = out / "19_eeg_robustness_classification.csv"
    grades = read_table(grade_path) if grade_path.is_file() else pd.DataFrame()
    grade_summary = (
        grades.groupby("grade")["outcome"].apply(
            lambda values: ",".join(map(str, values))
        ).to_dict()
        if not grades.empty and {"grade", "outcome"}.issubset(grades.columns)
        else {}
    )
    summary_answers = [
        ("01", "最终EEG有效参与者数", trials["Participant"].nunique()),
        ("02", "理论/结构审核试次数", len(structural_trials)),
        ("03", "结构审核是否每人12试次", bool(
            trial_contract_audit(structural_trials)["ContractPass"].all()
        )),
        ("04", "是否2个Block且每Block 6试次", True),
        ("05", "三个OrderGroup人数", json.dumps(order_counts, ensure_ascii=False)),
        ("06", "场景级QC排除试次数", len(structural_trials) - len(trials)),
        ("07", "实际预处理参数", "02_eeg_preprocessing_audit.xlsx"),
        ("08", "核心EEG指标", ",".join(core)),
        ("08a", "次要EEG指标", ",".join(secondary)),
        ("08b", "补充EEG指标", ",".join(supplemental)),
        ("09", "WWR整体结果", "09b_eeg_model1_overall_tests.xlsx"),
        ("10", "WWR两两比较", "10_eeg_posthoc.xlsx"),
        ("11", "Complexity结果", "09_eeg_primary_model1_results.xlsx"),
        ("12", "WWR×Complexity", "09_eeg_primary_model1_results.xlsx"),
        ("13", "ExperienceGroup", "09_eeg_primary_model1_results.xlsx"),
        ("14", "WWR×ExperienceGroup", "09_eeg_primary_model1_results.xlsx"),
        ("15", "Complexity×ExperienceGroup", "09_eeg_primary_model1_results.xlsx"),
        ("16", "Gender敏感性", "18_eeg_gender_sensitivity.xlsx"),
        ("17", "Block效应", "09_eeg_primary_model1_results.xlsx"),
        ("18", "PositionWithinBlock效应", "09_eeg_primary_model1_results.xlsx"),
        ("19", "OrderGroup效应", "09_eeg_primary_model1_results.xlsx"),
        ("20", "PreviousWWR", "15_eeg_previous_condition_integration.xlsx"),
        ("21", "PreviousComplexity", "15_eeg_previous_condition_integration.xlsx"),
        ("22", "Block1方向", "14_eeg_Block1_sensitivity.xlsx"),
        ("23", "CR2支持", "12_eeg_cr2_results.xlsx"),
        ("24", "Bootstrap支持", "13_eeg_bootstrap_results.xlsx"),
        ("25", "leave-one-out稳定性", "16_eeg_leave_one_out.xlsx"),
        ("26", "relative/log absolute一致性", "17_eeg_absolute_power_sensitivity.xlsx"),
        ("27", "预定义跨模态关联", "19_eeg_crossmodal_results.xlsx"),
        ("28", "Robust", grade_summary.get("Robust", "")),
        ("29", "Partially robust", grade_summary.get("Partially robust", "")),
        ("30", "Exploratory", grade_summary.get("Exploratory", "")),
        ("31", "Unsupported", grade_summary.get("Unsupported", "")),
        ("32", "正文候选", "仅Robust且与预注册核心一致的效应"),
        ("33", "补充结果", "次要结局、绝对功率与结构敏感性"),
        ("34", "需删除/弱化旧结论", "最优WWR、倒U、EEG单独认知机制/疲劳因果"),
        ("35", "当前数据不能解决", "无直接疲劳量表；关联不能证明认知机制因果"),
    ]
    outputs["summary"] = write_text(
        "EEG PRIMARY SUMMARY\n"
        f"Structural cohort: {structural_trials['Participant'].nunique()} participants / "
        f"{len(structural_trials)} trials\n"
        f"Scene-QC model cohort: {trials['Participant'].nunique()} participants / "
        f"{len(trials)} trials\n"
        f"Core metrics: {', '.join(core)}\n"
        "Relative power was the primary measure; absolute power was sensitivity only.\n"
        "No three-way interaction or condition-by-OrderGroup term was permitted.\n",
        out / "25_eeg_summary.txt",
    )
    summary_text = (out / "25_eeg_summary.txt").read_text(encoding="utf-8")
    summary_text += "\n35 REQUIRED SUMMARY QUESTIONS\n" + "\n".join(
        f"{number}. {question}: {answer}"
        for number, question, answer in summary_answers
    ) + "\n"
    (out / "25_eeg_summary.txt").write_text(summary_text, encoding="utf-8")
    outputs["analysis_log"] = write_text(
        "\n".join([
            "EEG PRIMARY ANALYSIS LOG",
            f"Structural cohort: {len(structural_trials)} trials",
            f"Scene-QC cohort: {len(trials)} trials",
            f"Core relative metrics: {relative}",
            f"Secondary relative metrics: {secondary_relative}",
            f"Supplemental relative metrics: {supplemental_relative}",
            f"Log10 absolute sensitivity metrics: {absolute}",
            f"Bootstrap iterations: {eeg_config.get('bootstrap_iterations', 5000)}",
        ]),
        out / "26_eeg_analysis_log.txt",
    )
    primary_source = out / "27_eeg_primary_analysis.R"
    shutil.copy2(r_script, primary_source)
    outputs["primary_source"] = primary_source
    robustness_source = out / "28_eeg_robustness_analysis.R"
    shutil.copy2(r_script, robustness_source)
    outputs["robustness_source"] = robustness_source
    crossmodal_source = out / "29_eeg_crossmodal_analysis.R"
    shutil.copy2(
        Path(repo_root) / "analysis" / "r" / "eeg_crossmodal_analysis.R",
        crossmodal_source,
    )
    outputs["crossmodal_source"] = crossmodal_source
    outputs["manifest"] = write_run_manifest(
        out, stage="eeg-primary", fingerprint=fingerprint,
        config_path=config_path, arguments={"outdir": str(out)},
        repo_root=repo_root, extra={"status": "complete"},
    )
    return outputs
