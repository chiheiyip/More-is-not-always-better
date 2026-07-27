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
    "WWR:Complexity:ExerciseFrequency",
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
        "Participant", "Gender", "ExerciseFrequency", "OrderGroup", "IncludeEEGValid"
    ]
    for column in ("Gender", "ExerciseFrequency", "OrderGroup", "IncludeEEGValid"):
        if column in trials:
            covariates.remove(column)
    trials = trials.merge(
        registry[covariates], on="Participant", how="left", validate="many_to_one"
    )
    return trials, registry


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
    trials = all_trials.loc[
        all_trials["IncludeEEGValid"].map(is_truthy)
    ].copy()
    invalid_trials = all_trials.loc[
        ~all_trials["IncludeEEGValid"].map(is_truthy)
    ].copy()
    if not invalid_trials.empty:
        invalid_trials["ExclusionReason"] = "IncludeEEGValid_false"
    audit = trial_contract_audit(trials)
    required_design = {
        "Participant", "WWR", "Complexity", "Gender", "ExerciseFrequency",
        "OrderGroup", "Block", "PositionWithinBlock",
    }
    missing_design = sorted(required_design - set(trials.columns))
    sequence, balance = (
        _sequence_tables(trials)
        if not missing_design
        else (pd.DataFrame(), pd.DataFrame())
    )
    order_groups = (
        sorted(trials["OrderGroup"].dropna().astype(str).unique())
        if "OrderGroup" in trials else []
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
    if blockers:
        write_run_manifest(
            out, stage="eeg-order", fingerprint=fingerprint,
            config_path=config_path, arguments={"outdir": str(out)},
            repo_root=repo_root, extra={"status": "blocked", "blockers": blockers},
        )
        raise StageBlockedError("EEG order stage blocked: " + " | ".join(blockers))
    model_input = out / "eeg_order_model_input.csv"
    trials.to_csv(model_input, index=False, encoding="utf-8-sig")
    r_script = Path(repo_root) / "analysis" / "r" / "eeg_order_analysis.R"
    r_ran = _invoke_r(
        str(config.get("rscript", "Rscript")), r_script,
        [str(model_input), str(out)], r_required,
    )
    if r_ran:
        _package_r_csv_outputs(out)
    if r_ran:
        try:
            from paper_analysis.teacher.reporting import (
                save_histogram,
                write_docx_report,
            )
        except ImportError as exc:
            raise StageBlockedError(
                "Teacher report packaging requires the declared python-docx dependency."
            ) from exc
        outputs["position_figure"] = save_histogram(
            trials["PositionWithinBlock"],
            path=out / "Figure1_EEG_condition_position_audit.png",
            title="EEG trial positions",
            xlabel="Position within block",
        )
        comparison_path = out / "06_eeg_model0_model1_comparison.csv"
        comparison = (
            read_table(comparison_path) if comparison_path.exists() else pd.DataFrame()
        )
        outputs["report_docx"] = write_docx_report(
            out / "11_EEG_order_effect_report.docx",
            title="EEG Order and Temporal Effects",
            paragraphs=[
                "Model 0 and time/order-adjusted Model 1 are reported side by side.",
                "Direction, interval, and estimate change are interpreted without "
                "using a p=0.05 crossing as the sole criterion.",
            ],
            tables=[("Model comparison", comparison), ("Position balance", balance)],
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
        f"Participants: {trials['Participant'].nunique()}\n"
        f"Trials: {len(trials)}\n"
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
    trials = all_trials.loc[
        all_trials["IncludeEEGValid"].map(is_truthy)
    ].copy()
    if trials.empty:
        raise StageBlockedError("No EEG-valid trials remain after IncludeEEGValid.")
    core = [str(v) for v in eeg_config.get("core_metrics", [])]
    secondary = [str(v) for v in eeg_config.get("secondary_metrics", [])]
    supplemental = [str(v) for v in eeg_config.get("supplemental_metrics", [])]
    relative, absolute = _metric_columns(trials, core)
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
                "Outcome ~ WWR*Complexity + WWR*ExerciseFrequency + "
                "Complexity*ExerciseFrequency + Gender + Block + "
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
    }
    model_input = out / "eeg_primary_model_input.csv"
    trials.to_csv(model_input, index=False, encoding="utf-8-sig")
    r_script = Path(repo_root) / "analysis" / "r" / "eeg_primary_analysis.R"
    r_ran = _invoke_r(
        str(config.get("rscript", "Rscript")), r_script,
        [
            str(model_input), str(out), ",".join(relative),
            ",".join(absolute),
            str(int(eeg_config.get("bootstrap_iterations", 5000))),
        ],
        r_required,
    )
    if r_ran:
        _package_r_csv_outputs(out)
    if r_ran:
        try:
            from paper_analysis.teacher.reporting import write_docx_report
        except ImportError as exc:
            raise StageBlockedError(
                "Teacher report packaging requires the declared python-docx dependency."
            ) from exc
        grade_path = out / "19_eeg_robustness_classification.csv"
        grades = read_table(grade_path) if grade_path.exists() else pd.DataFrame()
        outputs["report_docx"] = write_docx_report(
            out / "19_EEG_primary_results_report.docx",
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
    outputs["summary"] = write_text(
        "EEG PRIMARY SUMMARY\n"
        f"Core metrics: {', '.join(core)}\n"
        "Relative power was the primary measure; absolute power was sensitivity only.\n"
        "No three-way interaction or condition-by-OrderGroup term was permitted.\n",
        out / "20_eeg_primary_summary.txt",
    )
    outputs["manifest"] = write_run_manifest(
        out, stage="eeg-primary", fingerprint=fingerprint,
        config_path=config_path, arguments={"outdir": str(out)},
        repo_root=repo_root, extra={"status": "complete"},
    )
    return outputs
