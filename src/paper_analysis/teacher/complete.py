from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from more_is_not_always_better.discovery import load_questionnaire_long_from_wjx
from paper_analysis.teacher.eeg import (
    _scene_qc_mask,
    run_eeg_order,
    run_eeg_primary,
)
from paper_analysis.teacher.contracts import canonicalize_trials
from paper_analysis.teacher.eye import (
    run_eye_stage1,
    run_eye_stage2,
    run_eye_stage3,
    run_eye_stage3_plan,
)
from paper_analysis.teacher.eye_figures import build_eye_scene_figures
from paper_analysis.teacher.state import (
    file_sha256,
    legacy_stage_methods_unchanged,
    method_contract_hash,
    stage_method_contract_hash,
    write_run_manifest,
)
from paper_analysis.stats.timebin import run_timebin_models
from paper_analysis.utils.io import is_truthy, read_table, write_table, write_text


STAGE_FOLDERS = {
    "eye-stage1": "01_eye_stage1",
    "eye-stage2": "02_eye_stage2",
    "eye-stage3-plan": "03_eye_stage3_plan",
    "eye-stage3-run": "04_eye_stage3",
    "eeg-order": "05_eeg_order",
    "eeg-primary": "06_eeg_primary",
}

STAGE_REUSABLE_STATUSES = {
    "eye-stage1": {"review_required"},
    "eye-stage2": {"complete", "python_complete_r_skipped"},
    "eye-stage3-plan": {"approval_required"},
    "eye-stage3-run": {"complete", "python_complete_r_skipped"},
    "eeg-order": {"complete", "python_complete_r_skipped"},
    "eeg-primary": {"complete", "python_complete_r_skipped"},
}

STAGE_REUSE_REQUIRED_FILES = {
    "eye-stage1": ("AOI_masks_approved.txt",),
    "eye-stage3-plan": ("stage3_plan.txt",),
}


def _input_hashes_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    for record in records:
        source = Path(str(record.get("path", "")))
        expected_exists = bool(record.get("exists"))
        if source.exists() != expected_exists:
            return False
        expected_hash = record.get("sha256")
        if expected_hash and (
            not source.is_file() or file_sha256(source) != expected_hash
        ):
            return False
    return True


def _manifest_complete(path: Path, repo_root: Path, stage: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
    except (json.JSONDecodeError, OSError):
        return False
    accepted_statuses = STAGE_REUSABLE_STATUSES.get(
        stage, {"complete", "python_complete_r_skipped"}
    )
    if status not in accepted_statuses:
        return False
    if any(
        not (path.parent / required).is_file()
        for required in STAGE_REUSE_REQUIRED_FILES.get(stage, ())
    ):
        return False
    current_stage_hash = stage_method_contract_hash(repo_root, stage)
    stored_stage_hash = payload.get("stage_method_contract_hash")
    methods_current = (
        stored_stage_hash == current_stage_hash
        if stored_stage_hash
        else legacy_stage_methods_unchanged(
            repo_root, stage, str(payload.get("git_commit", ""))
        )
    )
    return bool(
        methods_current
        and _input_hashes_current(path.with_name("input_hashes.json"))
    )


def _run_or_resume(
    *,
    name: str,
    function: Callable[..., dict[str, Path]],
    config: dict[str, Any],
    config_path: Path,
    outdir: Path,
    repo_root: Path,
    resume: bool,
    r_required: bool | None = None,
) -> dict[str, Path]:
    if resume and _manifest_complete(
        outdir / "run_manifest.json", repo_root, name
    ):
        return {"manifest": outdir / "run_manifest.json"}
    kwargs: dict[str, Any] = {
        "config_path": config_path,
        "outdir": outdir,
        "repo_root": repo_root,
    }
    if r_required is not None:
        kwargs["r_required"] = r_required
    return function(config, **kwargs)


def _import_reusable_stages(
    outputs_root: Path,
    run_root: Path,
    repo_root: Path,
) -> pd.DataFrame:
    rows = []
    promoted = outputs_root / "12_teacher_analysis"
    for stage, folder in STAGE_FOLDERS.items():
        source = promoted / folder
        destination = run_root / folder
        valid = (
            source.is_dir()
            and _manifest_complete(source / "run_manifest.json", repo_root, stage)
        )
        if valid and not destination.exists():
            shutil.copytree(source, destination)
        rows.append({
            "Stage": stage,
            "Source": str(source),
            "Destination": str(destination),
            "ReuseDecision": "reused" if valid else "rerun",
            "ReuseReason": (
                "stage methods unchanged; recorded inputs hash-identical"
                if valid else
                "stage methods or recorded inputs changed/incomplete"
            ),
        })
    return pd.DataFrame(rows)


def _approve_stage3(plan_dir: Path) -> Path:
    manifest = json.loads(
        (plan_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    candidates = [str(value).upper() for value in manifest.get(
        "candidate_triggers", []
    )]
    target = plan_dir / "stage3_plan_approved.txt"
    target.write_text(
        json.dumps({
            "approved": True,
            "stage": "eye-stage3-plan",
            "stage_fingerprint": manifest["stage_fingerprint"],
            "approved_by": "codex_self_review_authorized_by_user",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approved_triggers": candidates,
            "notes": [
                "The user explicitly authorized Codex self-review and requested "
                "completion of all evidence-triggered analyses.",
                "Non-triggered analyses are documented in the plan and do not "
                "receive blank placeholder outputs.",
            ],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def _reuse_manifest(outputs_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Audit high-value compatibility artifacts before allowing reuse.

    Historical artifacts without a stored run fingerprint are never labelled
    hash-identical. They can be registered once only after a source/schema/key
    count check, which is recorded as a distinct validation mode.
    """
    candidates = [
        (
            outputs_root / "02_questionnaire" / "questionnaire_long.csv",
            "questionnaire results",
            {"participant_id", "scene_id"},
        ),
        (
            outputs_root / "05_multimodal_fusion" / "sync_qc.csv",
            "synchronization QC",
            {"participant_id", "scene_id"},
        ),
        (
            outputs_root / "05_multimodal_fusion" / "analysis_master_long.csv",
            "legacy fusion table",
            {"participant_id", "scene_id"},
        ),
        (
            outputs_root / "05_multimodal_fusion" / "aligned_timebin_table.csv",
            "legacy scene-repeated time-bin table",
            {"participant_id", "scene_id"},
        ),
    ]
    rows: list[dict[str, Any]] = []
    questionnaire_source = Path(
        str(config.get("questionnaire_file", "")).strip()
    )
    current_input_hash = (
        file_sha256(questionnaire_source)
        if questionnaire_source.is_file() else ""
    )
    for path, purpose, key_columns in candidates:
        record: dict[str, Any] = {
            "Artifact": str(path),
            "Purpose": purpose,
            "Exists": path.is_file(),
            "ArtifactSHA256": file_sha256(path) if path.is_file() else "",
            "OriginalRunTime": (
                datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                if path.is_file() else ""
            ),
            "OriginalInputSHA256": "",
            "CurrentInputSHA256": (
                current_input_hash if purpose == "questionnaire results" else ""
            ),
            "ConfigurationCheck": "not_available",
            "MethodContractCheck": "not_available",
            "SchemaAndCriticalNumbersCheck": "failed",
            "ReuseDecision": "rerun_or_not_used",
            "ReuseReason": "artifact missing",
        }
        if path.is_file():
            try:
                frame = read_table(path)
                schema_ok = key_columns.issubset(frame.columns)
                key_count = (
                    frame[list(key_columns)].drop_duplicates().shape[0]
                    if schema_ok else 0
                )
                record["SchemaAndCriticalNumbersCheck"] = (
                    f"passed; rows={len(frame)}; unique_keys={key_count}"
                    if schema_ok else
                    f"failed; missing={sorted(key_columns - set(frame.columns))}"
                )
                if schema_ok:
                    record["ReuseDecision"] = (
                        "compatibility_only"
                        if purpose == "legacy scene-repeated time-bin table"
                        else "validated_once_current_run"
                    )
                    record["ReuseReason"] = (
                        "No historical input fingerprint was available; source "
                        "schema and canonical key counts were recomputed once. "
                        "Artifact is compatibility/supplementary only."
                    )
                    record["ConfigurationCheck"] = "compatible_supplementary_role"
                    record["MethodContractCheck"] = (
                        "scene_repeated; prohibited for temporal inference"
                        if purpose == "legacy scene-repeated time-bin table"
                        else "does_not_replace_teacher_models"
                    )
                    if purpose == "questionnaire results":
                        current_prepared_value = str(
                            config.get("stage3", {}).get(
                                "s3_trial_file", ""
                            )
                        ).strip()
                        current_prepared = (
                            Path(current_prepared_value)
                            if current_prepared_value else None
                        )
                        if (
                            current_prepared is not None
                            and current_prepared.is_file()
                        ):
                            current = read_table(current_prepared)
                            left = frame.rename(columns={
                                "Participant": "participant_id",
                                "GlobalTrialOrder": "scene_id",
                            })
                            right = current.rename(columns={
                                "Participant": "participant_id",
                                "GlobalTrialOrder": "scene_id",
                            })
                            common_items = [
                                item for item in ("S1", "S2", "S3", "S4", "S5")
                                if item in left and item in right
                            ]
                            joined = left[[
                                "participant_id", "scene_id", *common_items
                            ]].merge(
                                right[[
                                    "participant_id", "scene_id", *common_items
                                ]],
                                on=["participant_id", "scene_id"],
                                suffixes=(".historical", ".current"),
                                how="inner",
                            )
                            equal = True
                            for item in common_items:
                                old = pd.to_numeric(
                                    joined[f"{item}.historical"],
                                    errors="coerce",
                                )
                                new = pd.to_numeric(
                                    joined[f"{item}.current"],
                                    errors="coerce",
                                )
                                equal &= bool(
                                    (old.eq(new) | (old.isna() & new.isna()))
                                    .all()
                                )
                            record["ConfigurationCheck"] = (
                                f"overlap_keys={len(joined)}; "
                                f"S1_S5_exact_match={equal}; "
                                "formal_42_person_analysis_rerun_separately"
                            )
                            record["ReuseDecision"] = (
                                "compatibility_only"
                            )
                            record["ReuseReason"] = (
                                "Historical 56-person table matches the current "
                                "workbook on overlapping keys/items, but formal "
                                "questionnaire inference was rerun on 42 people."
                            )
            except Exception as exc:  # retained in the audit rather than hidden
                record["ReuseReason"] = f"validation_error: {type(exc).__name__}: {exc}"
        rows.append(record)
    return pd.DataFrame(rows)


def _read_if(path: Path) -> pd.DataFrame:
    return read_table(path) if path.is_file() else pd.DataFrame()


def _standardize_effect_table(
    frame: pd.DataFrame,
    *,
    source: str,
    evidence_role: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    aliases = {
        "Estimate": ("estimate", "estimate.Model1", "Estimate", "beta"),
        "CI95Low": (
            "conf.low", "conf.low.Model1", "ci_low", "CI_low", "lower.CL"
        ),
        "CI95High": (
            "conf.high", "conf.high.Model1", "ci_high", "CI_high", "upper.CL"
        ),
        "RawP": (
            "p.value", "p.value.CR2", "p.value.likelihood",
            "p.value.Model1", "p_value", "p.value.raw", "p"
        ),
        "AdjustedP": (
            "p.value.BH", "p_fdr_bh", "p_adj", "p.value.Holm", "p_holm"
        ),
        "Outcome": ("outcome", "Outcome"),
        "Term": ("term", "contrast", "order_term", "Term"),
        "Model": ("model", "model_type", "Model"),
        "Participants": ("n_subjects", "Participants"),
        "Trials": ("n_trials", "Trials", "n"),
    }
    for target, candidates in aliases.items():
        source_column = next((c for c in candidates if c in result), None)
        if source_column is not None:
            result[target] = result[source_column]
        elif target not in result:
            result[target] = pd.NA
    result.insert(0, "EvidenceSource", source)
    result.insert(1, "EvidenceRole", evidence_role)
    leading = [
        "EvidenceSource", "EvidenceRole", "Outcome", "Model", "Term",
        "Estimate", "CI95Low", "CI95High", "RawP", "AdjustedP",
        "Participants", "Trials",
    ]
    return result[leading + [c for c in result if c not in leading]]


def _bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result


def _parallel_order_carryover_audit(eeg_primary_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trim in (0.0, 5.0, 10.0, 15.0):
        source = (
            eeg_primary_dir / "onset_window_models" / f"trim_{trim:g}s"
            / "09_eeg_order_CR2.csv"
        )
        evidence = _read_if(source)
        if evidence.empty:
            continue
        term = evidence.get("term", pd.Series("", index=evidence.index)).astype(str)
        model = evidence.get("model", pd.Series("", index=evidence.index)).astype(str)
        selected = evidence.loc[
            model.eq("PreviousScene")
            & term.str.startswith(("PreviousWWR", "PreviousComplexity"))
        ].copy()
        selected["onset_trim_s"] = trim
        selected["source_file"] = str(source)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    audit = pd.concat(rows, ignore_index=True, sort=False)
    audit["p_fdr_bh_window"] = np.nan
    for _, index in audit.groupby("onset_trim_s").groups.items():
        audit.loc[index, "p_fdr_bh_window"] = _bh_adjust(
            audit.loc[index, "p.value"]
        )
    audit["p_fdr_bh_parallel"] = _bh_adjust(audit["p.value"])
    audit["detected_raw_0_05"] = pd.to_numeric(
        audit["p.value"], errors="coerce"
    ).lt(0.05)
    audit["detected_window_fdr_0_05"] = audit["p_fdr_bh_window"].lt(0.05)
    audit["detected_parallel_fdr_0_05"] = audit["p_fdr_bh_parallel"].lt(0.05)
    audit["interpretation"] = np.where(
        audit["detected_parallel_fdr_0_05"],
        "detected_after_four_window_joint_fdr",
        "not_detected_after_four_window_joint_fdr",
    )
    return audit


def _build_questionnaire_teacher_audit(
    outputs_root: Path,
    run_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    r_required: bool,
) -> dict[str, Path]:
    out = run_root / "07_reviewer_analysis"
    out.mkdir(parents=True, exist_ok=True)
    questionnaire_source = Path(str(config.get("questionnaire_file", "")))
    registry_path = Path(str(config.get("participant_information", "")))
    trial_mapping_path = Path(str(config.get("trial_order_mapping", "")))
    registry = _read_if(registry_path)
    trials = _read_if(trial_mapping_path)
    if (
        not questionnaire_source.is_file()
        or registry.empty
        or trials.empty
    ):
        raise RuntimeError(
            "The teacher questionnaire audit requires the current questionnaire "
            "workbook, trial mapping and modality-union participant registry."
        )
    questionnaire = load_questionnaire_long_from_wjx(questionnaire_source)
    mapping = trials[[
        "Participant", "GlobalTrialOrder", "WWR", "Complexity", "Block",
        "PositionWithinBlock", "OrderGroup",
    ]].rename(columns={
        "Participant": "participant_id",
        "GlobalTrialOrder": "scene_id",
    })
    questionnaire = questionnaire.merge(
        mapping, on=["participant_id", "scene_id"], how="left", validate="one_to_one"
    )
    questionnaire = questionnaire.merge(
        registry[[
            "Participant", "ExperienceGroup", "Gender",
            "IncludeEEGValid", "IncludeEyeCandidate",
            "IncludeQuestionnaireValid",
        ]].rename(columns={"Participant": "participant_id"}),
        on="participant_id", how="left", validate="many_to_one",
    )
    if len(questionnaire) != 684 or questionnaire["participant_id"].nunique() != 57:
        raise RuntimeError(
            "Current questionnaire workbook did not produce the expected "
            "57 participants × 12 trials."
        )
    valid = set(
        registry.loc[
            registry["IncludeEEGValid"].map(is_truthy), "Participant"
        ].astype(str)
    )
    formal = questionnaire.loc[
        questionnaire["participant_id"].astype(str).isin(valid)
    ].copy()
    counts = pd.DataFrame([{
        "QuestionnaireSource": str(questionnaire_source),
        "QuestionnaireSourceSHA256": (
            file_sha256(questionnaire_source)
        ),
        "PreparedParticipantsAll": questionnaire["participant_id"].nunique(),
        "PreparedTrialsAll": len(questionnaire),
        "FormalRule": "IncludeEEGValid participant cohort",
        "Participants": formal["participant_id"].nunique(),
        "Trials": formal[["participant_id", "scene_id"]].drop_duplicates().shape[0],
        "ExpectedParticipants": 42,
        "ParticipantCountPass": formal["participant_id"].nunique() == 42,
    }])
    if not bool(counts.iloc[0]["ParticipantCountPass"]):
        raise RuntimeError(
            "Formal questionnaire cohort is not the registered 42-person EEG-valid cohort."
        )
    model_input = out / "Questionnaire_42_model_input.csv"
    formal.to_csv(model_input, index=False, encoding="utf-8-sig")
    r_outputs: dict[str, Path] = {}
    if r_required:
        r_script = (
            repo_root / "analysis" / "r"
            / "questionnaire_teacher_analysis.R"
        )
        subprocess.run([
            str(config.get("rscript", "Rscript")),
            str(r_script), str(model_input), str(out),
        ], check=True)
        for csv_path in sorted(out.glob("Questionnaire_42_*.csv")):
            if csv_path == model_input:
                continue
            xlsx_path = csv_path.with_suffix(".xlsx")
            r_outputs[csv_path.stem] = write_table(
                read_table(csv_path), xlsx_path
            )
        from paper_analysis.teacher.reporting import write_markdown_report
        cr2 = _read_if(out / "Questionnaire_42_CR2_BH_results.csv")
        descriptives = _read_if(
            out / "Questionnaire_42_descriptive_statistics.csv"
        )
        diagnostics = _read_if(
            out / "Questionnaire_42_model_diagnostics.csv"
        )
        r_outputs["report"] = write_markdown_report(
            out / "Questionnaire_42_results_report.md",
            title="问卷42人正式分析结果",
            paragraphs=[
                "样本按 IncludeEEGValid 固定为42人、504个场景试次。",
                "S1–S5 使用参与者随机截距 LMM；CR2 与 BH-FDR "
                "用于正式系数审查，WWR outcome内两两比较使用 Holm。",
                "ExperienceGroup 采用仓库Q1.4规则，ExerciseFrequency "
                "不进入正式模型；模型失败不会回退为OLS。",
            ],
            tables=[
                ("S1–S5描述统计", descriptives),
                ("CR2与BH-FDR", cr2),
                ("模型诊断", diagnostics),
            ],
        )
        shutil.copy2(
            r_script, out / "Questionnaire_42_analysis.R"
        )
    return {
        "complete": write_table(
            questionnaire, out / "Questionnaire_57_person_prepared_long.xlsx"
        ),
        "sample": write_table(
            formal, out / "Questionnaire_42_person_formal_sample.xlsx"
        ),
        "counts": write_table(
            counts, out / "Questionnaire_42_person_sample_audit.xlsx"
        ),
        "model_input": model_input,
        **r_outputs,
    }


def _build_reviewer_outputs(
    outputs_root: Path,
    run_root: Path,
) -> dict[str, Path]:
    out = run_root / "07_reviewer_analysis"
    out.mkdir(parents=True, exist_ok=True)
    order_dir = run_root / STAGE_FOLDERS["eeg-order"]
    eeg_primary_dir = run_root / STAGE_FOLDERS["eeg-primary"]
    eye2_dir = run_root / STAGE_FOLDERS["eye-stage2"]
    eye3_dir = run_root / STAGE_FOLDERS["eye-stage3-run"]
    model_comparison = _read_if(
        order_dir / "06_eeg_model0_model1_comparison.csv"
    )
    previous = _read_if(order_dir / "07_eeg_previous_scene_model.csv")
    block1 = _read_if(order_dir / "08_eeg_block1_sensitivity.csv")
    eeg_cr2 = _read_if(order_dir / "09_eeg_order_CR2.csv")
    eye_sensitivity = _read_if(
        eye2_dir / "16_eye_structural_sensitivities.csv"
    )
    eye_cr2 = _read_if(eye2_dir / "12_CR2_robust_results.csv")
    eye_time = _read_if(eye3_dir / "06_time_effect_models.csv")
    eye_carry = _read_if(eye3_dir / "07_carryover_models.csv")
    onset_evidence: list[pd.DataFrame] = []
    for trim in (0.0, 5.0, 10.0, 15.0):
        trim_dir = eeg_primary_dir / "onset_window_models" / f"trim_{trim:g}s"
        for filename, role in (
            ("06_eeg_model0_model1_comparison.csv", "parallel Model0 vs Model1"),
            ("07_eeg_previous_scene_model.csv", "parallel same-block carryover"),
            ("08_eeg_block1_sensitivity.csv", "parallel Block 1"),
            ("09_eeg_order_CR2.csv", "parallel cluster-robust inference"),
        ):
            evidence = _read_if(trim_dir / filename)
            if evidence.empty:
                continue
            evidence["onset_trim_s"] = trim
            onset_evidence.append(_standardize_effect_table(
                evidence,
                source=f"EEG {trim:g} s {filename}",
                evidence_role=role,
            ))
    r1 = pd.concat(
        [
            _standardize_effect_table(
                model_comparison, source="EEG Model0 vs Model1",
                evidence_role="primary time/order adjustment",
            ),
            _standardize_effect_table(
                previous, source="EEG previous-scene",
                evidence_role="same-block carryover",
            ),
            _standardize_effect_table(
                block1, source="EEG Block 1",
                evidence_role="Block 1 sensitivity",
            ),
            _standardize_effect_table(
                eeg_cr2, source="EEG CR2",
                evidence_role="cluster-robust inference",
            ),
            _standardize_effect_table(
                eye_time, source="eye time model",
                evidence_role="linear/quadratic time sensitivity",
            ),
            _standardize_effect_table(
                eye_carry, source="eye previous-scene",
                evidence_role="same-block carryover",
            ),
            _standardize_effect_table(
                eye_sensitivity, source="eye structural sensitivity",
                evidence_role="threshold/Block1/LOO/common sample",
            ),
            _standardize_effect_table(
                eye_cr2, source="eye CR2",
                evidence_role="cluster-robust inference",
            ),
            *onset_evidence,
        ],
        ignore_index=True,
        sort=False,
    )
    legacy_proxy = _read_if(
        outputs_root / "06_robustness" / "order_fatigue_effects.csv"
    )
    proxy_mask = legacy_proxy.get("outcome", pd.Series(dtype=str)).astype(
        str
    ).str.contains(
        "alpha|theta|blink|pupil|scanpath", case=False, regex=True, na=False
    )
    legacy_proxy = legacy_proxy.loc[proxy_mask].copy()
    eeg_proxy = model_comparison.loc[
        model_comparison.get("outcome", pd.Series(dtype=str)).astype(
            str
        ).str.contains("alpha|theta", case=False, regex=True, na=False)
    ].copy()
    r2 = pd.concat([
        _standardize_effect_table(
            eeg_proxy, source="teacher EEG Model0 vs Model1",
            evidence_role="primary alpha/theta temporal proxy",
        ),
        _standardize_effect_table(
            legacy_proxy, source="validated compatibility proxy analysis",
            evidence_role=(
                "supplementary blink/pupil/scanpath and EEG proxy evidence"
            ),
        ),
    ], ignore_index=True, sort=False)
    if not r2.empty:
        r2["InterpretationBoundary"] = (
            "Temporal EEG/eye proxies are not a direct fatigue scale; "
            "proxy changes must not be labelled fatigue."
        )
    carryover_audit = _parallel_order_carryover_audit(eeg_primary_dir)
    carryover_path = write_table(
        carryover_audit,
        out / "Reviewer1_parallel_previous_scene_FDR.csv",
    )
    carryover_summary = (
        f"四窗口共同QC前序项共{len(carryover_audit)}个系数；"
        f"raw p<0.05为{int(carryover_audit.get('detected_raw_0_05', pd.Series(dtype=bool)).sum())}个，"
        f"窗口内FDR后为{int(carryover_audit.get('detected_window_fdr_0_05', pd.Series(dtype=bool)).sum())}个，"
        f"四窗口联合FDR后为{int(carryover_audit.get('detected_parallel_fdr_0_05', pd.Series(dtype=bool)).sum())}个。"
        if not carryover_audit.empty else
        "四窗口共同QC前序项审计未生成，不能作排除性表述。"
    )
    return {
        "reviewer1": write_table(
            r1, out / "Reviewer1_comment11_order_time_carryover.xlsx"
        ),
        "reviewer2": write_table(
            r2, out / "Reviewer2_comment5_fatigue_proxy_evidence.xlsx"
        ),
        "parallel_previous_scene_fdr": carryover_path,
        "summary": write_text(
            "# Reviewer专项分析\n\n"
            "## Reviewer 1 意见11\n\n"
            "并行报告0、5、10、15 s下的 Block、Position、三个OrderGroup、"
            "Model 0/Model 1、同Block前序条件、Block 1与CR2；"
            "不使用“完全排除顺序效应”。" + carryover_summary + "\n\n"
            "## Reviewer 2 意见5\n\n"
            "仅把 alpha/theta、眨眼、瞳孔和扫描路径视为时间变化代理。"
            "实验没有直接疲劳量表，因此不把代理变化直接表述为疲劳。\n",
            out / "Reviewer专项分析结果.md",
        ),
    }


def _build_sync_outputs(
    outputs_root: Path,
    run_root: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    out = run_root / "08_synchronized_crossmodal"
    out.mkdir(parents=True, exist_ok=True)
    configured = str(config.get("synchronized_timebin_file", "")).strip()
    candidates = [
        Path(configured) if configured else None,
        run_root / "08_synchronized_crossmodal"
        / "aligned_synchronized_timebin_table.csv",
        outputs_root / "05_multimodal_fusion" / "aligned_timebin_table.csv",
        outputs_root / "09_data_package" / "aligned_timebin_long.csv",
    ]
    source = next(
        (path for path in candidates if path is not None and path.is_file()),
        None,
    )
    rows = 0
    participants = 0
    trials = 0
    columns = ""
    temporal_status = ""
    temporal_resolution = ""
    if source is not None:
        frame = read_table(source)
        expected_trims = tuple(sorted(float(value) for value in config.get(
            "eeg", {}
        ).get("onset_trim_variants_s", [0, 5, 10, 15])))
        if "onset_trim_s" not in frame:
            raise RuntimeError(
                "Parallel synchronized analysis requires onset_trim_s."
            )
        frame["onset_trim_s"] = pd.to_numeric(
            frame["onset_trim_s"], errors="coerce"
        )
        observed_trims = tuple(sorted(
            frame["onset_trim_s"].dropna().unique().astype(float)
        ))
        if observed_trims != expected_trims:
            raise RuntimeError(
                "Formal synchronized table must contain parallel onset trims "
                f"{list(expected_trims)}; observed {list(observed_trims)}."
            )
        rows = len(frame)
        participant_col = next(
            (c for c in ("Participant", "participant_id") if c in frame), None
        )
        trial_col = next(
            (c for c in ("GlobalTrialOrder", "scene_id") if c in frame), None
        )
        participants = frame[participant_col].nunique() if participant_col else 0
        trials = (
            frame[[participant_col, trial_col]].drop_duplicates().shape[0]
            if participant_col and trial_col else 0
        )
        columns = ",".join(frame.columns)
        temporal_status = ",".join(sorted(
            frame.get("analysis_status", pd.Series(dtype=str))
            .dropna().astype(str).unique()
        ))
        temporal_resolution = ",".join(sorted(
            frame.get("eeg_temporal_resolution", pd.Series(dtype=str))
            .dropna().astype(str).unique()
        ))
    formal_sync = (
        source is not None
        and "window_specific" in temporal_resolution
        and (
            "parallel" in temporal_status
            or "synchronized_timebin" in temporal_status
        )
    )
    if not formal_sync:
        raise RuntimeError(
            "Formal synchronized cross-modal delivery requires a "
            "window_specific synchronized time-bin table. The existing "
            "51,928-row scene_repeated table is compatibility-only."
        )
    eeg_trial_path = Path(
        str(config.get("eeg", {}).get("trial_file", ""))
    )
    eye_trial_path = (
        run_root / "02_eye_stage2"
        / "06_trial_level_eye_tracking_data.xlsx"
    )
    if not eeg_trial_path.is_file() or not eye_trial_path.is_file():
        raise RuntimeError(
            "Formal synchronized analysis requires both the current EEG "
            "scene-QC table and eye Stage 2 trial-QC table."
        )
    eeg_trials = canonicalize_trials(read_table(eeg_trial_path))
    if "IncludeEEGValid" not in eeg_trials:
        registry = read_table(config["participant_information"])
        eeg_trials = eeg_trials.merge(
            registry[["Participant", "IncludeEEGValid"]]
            .drop_duplicates("Participant"),
            on="Participant",
            how="left",
            validate="many_to_one",
        )
    eeg_trials = eeg_trials.loc[
        eeg_trials["IncludeEEGValid"].map(is_truthy)
        & _scene_qc_mask(eeg_trials)
    ].copy()
    eye_trials = read_table(eye_trial_path)
    eye_trials = eye_trials.loc[
        eye_trials["IncludedPrimary"].map(is_truthy)
    ].copy()

    def key_frame(data: pd.DataFrame) -> pd.DataFrame:
        participant_column = next(
            c for c in ("Participant", "participant_id") if c in data
        )
        trial_column = next(
            c for c in ("GlobalTrialOrder", "scene_id") if c in data
        )
        result = data[[participant_column, trial_column]].copy()
        result.columns = ["Participant", "GlobalTrialOrder"]
        result["Participant"] = result["Participant"].astype(str).str.strip()
        result["GlobalTrialOrder"] = pd.to_numeric(
            result["GlobalTrialOrder"], errors="coerce"
        ).astype("Int64")
        return result.drop_duplicates()

    clock_qc_value = str(config.get("clock_scene_qc_file", "")).strip()
    clock_qc_candidates = [
        Path(clock_qc_value) if clock_qc_value else None,
        out / "clock_alignment_scene_qc.csv",
        out / "alignment_scene_qc.csv",
    ]
    clock_qc = next(
        (
            path for path in clock_qc_candidates
            if path is not None and path.is_file()
        ),
        None,
    )
    if clock_qc is None or not clock_qc.is_file():
        raise RuntimeError(
            "Formal synchronized analysis requires clock_scene_qc_file; "
            "a window-specific table without its clock QC is not promotable."
        )
    clock_frame = read_table(clock_qc)
    if "clock_alignment_pass" not in clock_frame:
        raise RuntimeError(
            "Clock scene QC is missing clock_alignment_pass."
        )
    clock_keys = key_frame(
        clock_frame.loc[
            clock_frame["clock_alignment_pass"].map(is_truthy)
        ]
    ).assign(ClockAlignmentPass=True)
    participant_column = next(
        c for c in ("Participant", "participant_id") if c in frame
    )
    trial_column = next(
        c for c in ("GlobalTrialOrder", "scene_id") if c in frame
    )
    trial_window_counts = (
        frame[[participant_column, trial_column, "onset_trim_s"]]
        .drop_duplicates()
        .groupby([participant_column, trial_column])["onset_trim_s"]
        .nunique()
    )
    complete_window_keys = trial_window_counts.loc[
        trial_window_counts.eq(len(expected_trims))
    ].index.to_frame(index=False)
    frame = frame.merge(
        complete_window_keys,
        on=[participant_column, trial_column],
        how="inner",
        validate="many_to_one",
    )
    synchronized_keys = key_frame(frame)
    eeg_keys = key_frame(eeg_trials).assign(EEGSceneQCPass=True)
    eye_keys = key_frame(eye_trials).assign(EyeStage2QCPass=True)
    eligibility = (
        synchronized_keys.merge(
            eeg_keys, on=["Participant", "GlobalTrialOrder"], how="left"
        ).merge(
            eye_keys, on=["Participant", "GlobalTrialOrder"], how="left"
        ).merge(
            clock_keys, on=["Participant", "GlobalTrialOrder"], how="left"
        )
    )
    eligibility["EEGSceneQCPass"] = (
        eligibility["EEGSceneQCPass"].fillna(False).astype(bool)
    )
    eligibility["EyeStage2QCPass"] = (
        eligibility["EyeStage2QCPass"].fillna(False).astype(bool)
    )
    eligibility["ClockAlignmentPass"] = eligibility[
        "ClockAlignmentPass"
    ].fillna(False).astype(bool)
    eligibility["FormalSynchronizedEligible"] = (
        eligibility["EEGSceneQCPass"]
        & eligibility["EyeStage2QCPass"]
        & eligibility["ClockAlignmentPass"]
    )
    eligibility["ExclusionReason"] = np.select(
        [
            ~eligibility["ClockAlignmentPass"],
            ~eligibility["EEGSceneQCPass"]
            & ~eligibility["EyeStage2QCPass"],
            ~eligibility["EEGSceneQCPass"],
            ~eligibility["EyeStage2QCPass"],
        ],
        [
            "clock_alignment_QC_failed",
            "EEG_scene_QC_and_eye_60pct_QC_failed",
            "EEG_scene_QC_failed",
            "eye_60pct_QC_failed",
        ],
        default="",
    )
    write_table(
        eligibility, out / "synchronized_trial_eligibility_audit.xlsx"
    )
    synchronized_participants = set(
        clock_keys["Participant"].astype(str)
    )
    eeg_valid_participants = sorted(
        set(eeg_trials["Participant"].astype(str))
    )
    clock_coverage = pd.DataFrame({
        "Participant": eeg_valid_participants,
        "EEGParticipantAndSceneQCPresent": True,
        "ValidatedClockSynchronizedDataPresent": [
            participant in synchronized_participants
            for participant in eeg_valid_participants
        ],
    })
    clock_coverage["ExclusionReason"] = np.where(
        clock_coverage["ValidatedClockSynchronizedDataPresent"],
        "",
        "no_validated_clock_cache_or_synchronized_sample_export",
    )
    write_table(
        clock_coverage, out / "eeg_clock_coverage_audit.xlsx"
    )
    formal_keys = eligibility.loc[
        eligibility["FormalSynchronizedEligible"],
        ["Participant", "GlobalTrialOrder"],
    ]
    frame_with_keys = frame.copy()
    # Assign keys directly from the source columns to preserve every time-bin row.
    frame_with_keys["_ParticipantKey"] = (
        frame[participant_column].astype(str).str.strip()
    )
    frame_with_keys["_TrialKey"] = pd.to_numeric(
        frame[trial_column], errors="coerce"
    ).astype("Int64")
    formal = frame_with_keys.merge(
        formal_keys.rename(columns={
            "Participant": "_ParticipantKey",
            "GlobalTrialOrder": "_TrialKey",
        }),
        on=["_ParticipantKey", "_TrialKey"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["_ParticipantKey", "_TrialKey"])
    if formal.empty:
        raise RuntimeError(
            "No synchronized time-bin rows survive exact EEG scene QC and "
            "eye 60% trial QC."
        )
    formal_source = out / "aligned_synchronized_timebin_formal_qc.csv"
    formal.to_csv(formal_source, index=False, encoding="utf-8-sig")
    participants = formal[participant_column].nunique()
    trials = formal[[participant_column, trial_column]].drop_duplicates().shape[0]
    rows = len(formal)
    summary = pd.DataFrame([{
        "RawSynchronizedSource": str(source),
        "RawSynchronizedSourceSHA256": file_sha256(source),
        "RawClockParticipants": frame[participant_column].nunique(),
        "RawClockTrials": frame[[
            participant_column, trial_column
        ]].drop_duplicates().shape[0],
        "RawClockTimeBins": len(frame),
        "Source": str(formal_source),
        "SourceSHA256": file_sha256(formal_source),
        "Participants": participants,
        "Trials": trials,
        "TimeBins": rows,
        "ParallelOnsetTrimsS": ",".join(
            f"{value:g}" for value in expected_trims
        ),
        "Columns": columns,
        "EEGTemporalResolution": temporal_resolution,
        "AnalysisStatus": temporal_status,
        "EligibilityRule": (
            "absolute-clock scene QC AND EEG participant+scene QC "
            "AND eye Stage2 60% trial QC AND presence in all parallel trims"
        ),
        "ReuseStatus": "parallel_synchronized_timebin_exact_qc_intersection",
    }])
    model_outputs = run_timebin_models(
        synchronized_timebin_csv=formal_source,
        clock_scene_qc_csv=clock_qc,
        outdir=out,
        scene_model_results_csv=None,
        expected_onset_trims_s=expected_trims,
        time_column="scene_time_norm",
        require_common_trials=True,
    )
    return {
        "summary": write_table(
            summary, out / "synchronized_crossmodal_sample_and_source.xlsx"
        ),
        "readme": write_text(
            "# 时序同步与跨模态结果\n\n"
            f"- 原始绝对时钟 time-bin 来源：`{source}`\n"
            f"- 四窗口并行精确QC交集 time-bin：`{formal_source}`\n"
            f"- 并行窗口：{', '.join(f'{value:g} s' for value in expected_trims)}\n"
            f"- 实际参与者：{participants}\n"
            f"- 实际 Participant × Trial：{trials}\n"
            f"- time-bin 行数：{rows}\n\n"
            "正式跨模态解释只允许使用预设且稳定的 1–2 个 EEG 指标，"
            "并以精确 Participant × Trial 或 time-bin 交集为准。\n",
            out / "README.md",
        ),
        **{f"timebin_{key}": value for key, value in model_outputs.items()},
    }


def _all_files_index(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name != "结果文件总索引.xlsx"
    ):
        rows.append({
            "RelativePath": str(path.relative_to(root)),
            "SizeBytes": path.stat().st_size,
            "ModifiedUTC": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "SHA256": file_sha256(path),
        })
    return pd.DataFrame(rows)


def _completion_matrix(run_root: Path) -> pd.DataFrame:
    requirements = [
        ("眼动 Stage 1", "输入与试次审核", "01_eye_stage1/03_participant_trial_audit.xlsx"),
        ("眼动 Stage 1", "12套AOI映射审核", "01_eye_stage1/06_scene_AOI_mapping_check.xlsx"),
        ("眼动 Stage 1", "AOI质量与重叠审核", "01_eye_stage1/07_AOI_quality_report.xlsx"),
        ("眼动 Stage 1", "像素/球面加权面积", "01_eye_stage1/08_AOI_area_report.xlsx"),
        ("眼动 Stage 1", "AOI叠加图", "01_eye_stage1/FigureS_AOI_overlay_preview.png"),
        ("眼动 Stage 1", "ValidScene预览", "01_eye_stage1/FigureS_ValidScene_preview.png"),
        ("眼动 Stage 1", "运行指纹", "01_eye_stage1/run_manifest.json"),
        ("眼动 Stage 2", "Fixation去重明细", "02_eye_stage2/01_fixation_event_level_data.xlsx"),
        ("眼动 Stage 2", "试次QC", "02_eye_stage2/02_trial_quality_control.xlsx"),
        ("眼动 Stage 2", "排除记录", "02_eye_stage2/03_exclusion_log.xlsx"),
        ("眼动 Stage 2", "参与者保留分布", "02_eye_stage2/04_participant_quality_summary.xlsx"),
        ("眼动 Stage 2", "六个核心结局", "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx"),
        ("眼动 Stage 2", "核心描述统计", "02_eye_stage2/08_core_descriptive_statistics.xlsx"),
        ("眼动 Stage 2", "Family A模型", "02_eye_stage2/09_familyA_primary_models.xlsx"),
        ("眼动 Stage 2", "Family B模型", "02_eye_stage2/10_familyB_primary_models.xlsx"),
        ("眼动 Stage 2", "Holm事后比较", "02_eye_stage2/11_WWR_posthoc_Holm.xlsx"),
        ("眼动 Stage 2", "CR2", "02_eye_stage2/12_CR2_robust_results.xlsx"),
        ("眼动 Stage 2", "50/60/70%敏感性", "02_eye_stage2/13_tracking_threshold_sensitivity_models.xlsx"),
        ("眼动 Stage 2", "Block 1敏感性", "02_eye_stage2/14_block1_sensitivity_models.xlsx"),
        ("眼动 Stage 2", "Leave-one-out", "02_eye_stage2/15_leave_one_participant_out.xlsx"),
        ("眼动 Stage 2", "精确EEG共同样本", "02_eye_stage2/16_EEG_valid_common_sample_sensitivity.xlsx"),
        ("眼动 Stage 2", "共同样本计数", "02_eye_stage2/16b_EEG_valid_common_sample_counts.xlsx"),
        ("眼动 Stage 2", "证据分级", "02_eye_stage2/18_eye_evidence_classification.xlsx"),
        ("眼动 Stage 2", "中文Markdown报告", "02_eye_stage2/19_eye_stage2_results_report_CN.md"),
        ("眼动 Stage 2", "33项summary", "02_eye_stage2/21_stage2_summary.txt"),
        ("眼动 Stage 2", "5张核心图", "02_eye_stage2/Figure5_table_share_by_condition.png"),
        ("眼动图形", "六场景统一AOI图", "09_eye_figures/FigureS_AOI_regions_unified.svg"),
        ("眼动图形", "Fig. 6 fixation事件密度", "09_eye_figures/Figure6_fixation_event_density.svg"),
        ("眼动图形", "fixation时长加权敏感性", "09_eye_figures/FigureS_fixation_duration_density.svg"),
        ("眼动图形", "Block配准QA", "09_eye_figures/Figure6_registration_qc.csv"),
        ("眼动 Stage 3", "计划与触发依据", "03_eye_stage3_plan/stage3_plan.txt"),
        ("眼动 Stage 3", "触发完成表", "04_eye_stage3/18_stage3_trigger_completion.xlsx"),
        ("眼动 Stage 3", "中文Markdown报告", "04_eye_stage3/19_eye_stage3_results_report_CN.md"),
        ("眼动 Stage 3", "summary", "04_eye_stage3/17_stage3_summary.txt"),
        ("EEG 顺序阶段", "42人/504试次结构审核", "05_eeg_order/01_eeg_data_audit.xlsx"),
        ("EEG 顺序阶段", "条件位置平衡", "05_eeg_order/03_condition_position_frequency.xlsx"),
        ("EEG 顺序阶段", "Model0/Model1比较", "05_eeg_order/06_eeg_model0_model1_comparison.xlsx"),
        ("EEG 顺序阶段", "同Block前序模型", "05_eeg_order/07_eeg_previous_scene_model.xlsx"),
        ("EEG 顺序阶段", "Block 1", "05_eeg_order/08_eeg_block1_sensitivity.xlsx"),
        ("EEG 顺序阶段", "CR2", "05_eeg_order/09_eeg_order_CR2.xlsx"),
        ("EEG 顺序阶段", "Markdown报告", "05_eeg_order/11_EEG_order_effect_report.md"),
        ("EEG 顺序阶段", "summary", "05_eeg_order/12_eeg_order_summary.txt"),
        ("EEG 主分析", "relative/absolute数据字典", "06_eeg_primary/06_eeg_variable_dictionary.xlsx"),
        ("EEG 主分析", "42/504与42/471样本流", "06_eeg_primary/03b_eeg_sample_flow.xlsx"),
        ("EEG 主分析", "主模型BH", "06_eeg_primary/04_eeg_primary_models_BH.xlsx"),
        ("EEG 主分析", "次要指标模型", "06_eeg_primary/09c_eeg_secondary_models.xlsx"),
        ("EEG 主分析", "补充指标模型", "06_eeg_primary/09d_eeg_supplemental_models.xlsx"),
        ("EEG 主分析", "WWR事后Holm", "06_eeg_primary/05_eeg_WWR_posthoc_Holm.xlsx"),
        ("EEG 主分析", "CR2", "06_eeg_primary/06_eeg_CR2_results.xlsx"),
        ("EEG 主分析", "5000次Bootstrap", "06_eeg_primary/07_eeg_cluster_bootstrap_5000.xlsx"),
        ("EEG 主分析", "Bootstrap失败记录", "06_eeg_primary/08_eeg_bootstrap_failures.xlsx"),
        ("EEG 主分析", "Block 1", "06_eeg_primary/14_eeg_Block1_sensitivity.xlsx"),
        ("EEG 主分析", "完整12试次", "06_eeg_primary/14b_eeg_complete12_sensitivity.xlsx"),
        ("EEG 主分析", "前序条件", "06_eeg_primary/15_eeg_previous_condition_integration.xlsx"),
        ("EEG 主分析", "Leave-one-out", "06_eeg_primary/16_eeg_leave_one_out.xlsx"),
        ("EEG 主分析", "绝对功率敏感性", "06_eeg_primary/17_eeg_absolute_power_sensitivity.xlsx"),
        ("EEG 主分析", "性别敏感性", "06_eeg_primary/18_eeg_gender_sensitivity.xlsx"),
        ("EEG 主分析", "证据分级", "06_eeg_primary/20_eeg_evidence_classification.xlsx"),
        ("EEG 主分析", "Markdown方法报告", "06_eeg_primary/22_eeg_methods_report.md"),
        ("EEG 主分析", "中文Markdown结果", "06_eeg_primary/24_eeg_results_report_cn.md"),
        ("EEG 主分析", "35项summary", "06_eeg_primary/25_eeg_summary.txt"),
        ("EEG 主分析", "6张图", "06_eeg_primary/FigureS_EEG_leave_one_out.png"),
        ("Reviewer专项", "问卷42人样本审核", "07_reviewer_analysis/Questionnaire_42_person_sample_audit.xlsx"),
        ("Reviewer专项", "当前问卷57人完整准备表", "07_reviewer_analysis/Questionnaire_57_person_prepared_long.xlsx"),
        ("Reviewer专项", "问卷42人LMM", "07_reviewer_analysis/Questionnaire_42_LMM_results.xlsx"),
        ("Reviewer专项", "问卷42人CR2/BH", "07_reviewer_analysis/Questionnaire_42_CR2_BH_results.xlsx"),
        ("Reviewer专项", "问卷42人Holm", "07_reviewer_analysis/Questionnaire_42_WWR_posthoc_Holm.xlsx"),
        ("Reviewer专项", "问卷42人Markdown报告", "07_reviewer_analysis/Questionnaire_42_results_report.md"),
        ("Reviewer专项", "Reviewer 1意见11", "07_reviewer_analysis/Reviewer1_comment11_order_time_carryover.xlsx"),
        ("Reviewer专项", "四窗口前序项联合FDR", "07_reviewer_analysis/Reviewer1_parallel_previous_scene_FDR.csv"),
        ("Reviewer专项", "Reviewer 2意见5", "07_reviewer_analysis/Reviewer2_comment5_fatigue_proxy_evidence.xlsx"),
        ("同步跨模态", "绝对时钟同步样本", "08_synchronized_crossmodal/synchronized_crossmodal_sample_and_source.xlsx"),
        ("同步跨模态", "眼动/EEG精确QC交集", "08_synchronized_crossmodal/synchronized_trial_eligibility_audit.xlsx"),
        ("同步跨模态", "EEG时钟覆盖与排除", "08_synchronized_crossmodal/eeg_clock_coverage_audit.xlsx"),
        ("同步跨模态", "四窗口并行window-specific time-bin表", "08_synchronized_crossmodal/aligned_synchronized_timebin_formal_qc.csv"),
        ("同步跨模态", "四窗口并行time-bin模型", "08_synchronized_crossmodal/timebin_model_results.csv"),
        ("同步跨模态", "time-bin模型诊断", "08_synchronized_crossmodal/timebin_model_diagnostics.csv"),
        ("同步跨模态", "四窗口共同样本流", "08_synchronized_crossmodal/parallel_window_sample_flow.csv"),
        ("同步跨模态", "四窗口稳定性分类", "08_synchronized_crossmodal/parallel_window_stability.csv"),
    ]
    rows = [
        {
            "Category": category,
            "Requirement": requirement,
            "RunRelativePath": relative,
            "Exists": (run_root / relative).is_file(),
            "NonEmpty": (
                (run_root / relative).is_file()
                and (run_root / relative).stat().st_size > 0
            ),
            "Status": (
                "complete"
                if (run_root / relative).is_file()
                and (run_root / relative).stat().st_size > 0
                else "missing_or_empty"
            ),
        }
        for category, requirement, relative in requirements
    ]
    trigger_path = run_root / "04_eye_stage3" / "18_stage3_trigger_completion.xlsx"
    trigger_frame = _read_if(trigger_path)
    for row in trigger_frame.itertuples(index=False):
        if not is_truthy(getattr(row, "Approved", False)):
            continue
        generated = [
            value for value in str(getattr(row, "GeneratedFiles", "")).split(";")
            if value
        ]
        if not generated:
            rows.append({
                "Category": "眼动 Stage 3",
                "Requirement": f"触发项 {row.Trigger} 有可估计输出",
                "RunRelativePath": "",
                "Exists": False,
                "NonEmpty": False,
                "Status": str(getattr(row, "Status", "approved_no_output")),
            })
        for filename in generated:
            path = run_root / "04_eye_stage3" / filename
            rows.append({
                "Category": "眼动 Stage 3",
                "Requirement": f"触发项 {row.Trigger}: {filename}",
                "RunRelativePath": f"04_eye_stage3/{filename}",
                "Exists": path.is_file(),
                "NonEmpty": path.is_file() and path.stat().st_size > 0,
                "Status": (
                    "complete" if path.is_file() and path.stat().st_size > 0
                    else "missing_or_empty"
                ),
            })
    return pd.DataFrame(rows)


def _markdown_preview(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_无可估计结果；详见对应诊断表。_"
    preview = frame[available].head(limit).copy()
    for column in preview:
        if pd.api.types.is_numeric_dtype(preview[column]):
            preview[column] = preview[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        else:
            preview[column] = preview[column].fillna("").astype(str)
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in preview.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _effect_preview(frame: pd.DataFrame, *, limit: int = 10) -> str:
    normalized = _standardize_effect_table(
        frame, source="report preview", evidence_role="formal"
    )
    if normalized.empty:
        return "_无可估计结果；详见对应诊断表。_"
    normalized["Estimate"] = pd.to_numeric(
        normalized["Estimate"], errors="coerce"
    )
    normalized["RawP"] = pd.to_numeric(normalized["RawP"], errors="coerce")
    normalized["AdjustedP"] = pd.to_numeric(
        normalized["AdjustedP"], errors="coerce"
    )
    normalized = normalized.loc[
        normalized["Estimate"].notna()
        & ~normalized["Term"].astype(str).str.contains(
            "Intercept", case=False, na=False
        )
    ].copy()
    normalized["_sort_p"] = normalized["AdjustedP"].fillna(
        normalized["RawP"]
    ).fillna(float("inf"))
    normalized = normalized.sort_values(
        ["_sort_p", "Outcome", "Term"], kind="stable"
    )
    return _markdown_preview(
        normalized,
        [
            "Outcome", "Model", "Term", "Estimate", "CI95Low",
            "CI95High", "RawP", "AdjustedP",
        ],
        limit,
    )


def _build_top_report(outputs_root: Path, run_root: Path) -> Path:
    eye_counts = read_table(
        run_root / "02_eye_stage2" / "16b_EEG_valid_common_sample_counts.xlsx"
    )
    eye_qc = read_table(
        run_root / "02_eye_stage2" / "04_participant_quality_summary.xlsx"
    )
    eeg_flow = read_table(
        run_root / "06_eeg_primary" / "03b_eeg_sample_flow.xlsx"
    )
    sync = read_table(
        run_root / "08_synchronized_crossmodal"
        / "synchronized_crossmodal_sample_and_source.xlsx"
    )
    eye_valid_participants = int((eye_qc["PrimaryValidTrials"] > 0).sum())
    eye_valid_trials = int(eye_qc["PrimaryValidTrials"].sum())
    eye_figure_summary_path = (
        run_root / "09_eye_figures" / "eye_scene_figure_summary.json"
    )
    if eye_figure_summary_path.is_file():
        eye_figure_summary = json.loads(
            eye_figure_summary_path.read_text(encoding="utf-8")
        )
        eye_figure_text = (
            "Fig. 6 将 Block 2 配准到 Block 1 后，按六个 "
            "Complexity × WWR 条件汇总通过60% tracking QC且命中 "
            f"ValidScene 的 fixation；共纳入 {int(eye_figure_summary['participants'])} "
            f"人、{int(eye_figure_summary['trials'])} 个试次、"
            f"{int(eye_figure_summary['valid_scene_fixations_before_registration'])} "
            "个配准前有效 fixation，最终绘制 "
            f"{int(eye_figure_summary['fixations_plotted'])} 个。六组配准均通过预设 "
            "RANSAC、重投影误差、ValidScene/AOI IoU及fixation保留率门槛。"
        )
    else:
        eye_figure_text = (
            "六场景AOI图与Fig. 6由独立的眼动场景绘图流程生成；"
            "正式解释前必须同时核对配准QA与图形source data。"
        )
    common = eye_counts.iloc[0]
    structural = eeg_flow.iloc[0]
    model = eeg_flow.iloc[1]
    sync_row = sync.iloc[0]
    clock_coverage = read_table(
        run_root / "08_synchronized_crossmodal"
        / "eeg_clock_coverage_audit.xlsx"
    )
    no_clock_count = int(
        (~clock_coverage[
            "ValidatedClockSynchronizedDataPresent"
        ].map(is_truthy)).sum()
    )
    questionnaire_audit = read_table(
        run_root / "07_reviewer_analysis"
        / "Questionnaire_42_person_sample_audit.xlsx"
    ).iloc[0]
    eye_models = pd.concat([
        _read_if(run_root / "02_eye_stage2" / "09_familyA_primary_models.csv"),
        _read_if(run_root / "02_eye_stage2" / "10_familyB_primary_models.csv"),
    ], ignore_index=True, sort=False)
    questionnaire_models = _read_if(
        run_root / "07_reviewer_analysis"
        / "Questionnaire_42_CR2_BH_results.csv"
    )
    eeg_models = _read_if(
        run_root / "06_eeg_primary" / "04_eeg_primary_models_BH.csv"
    )
    order_comparison = _read_if(
        run_root / "05_eeg_order" / "06_eeg_model0_model1_comparison.csv"
    )
    crossmodal = _read_if(
        run_root / "06_eeg_primary" / "19_eeg_crossmodal_results.csv"
    )
    timebin = _read_if(
        run_root / "08_synchronized_crossmodal" / "timebin_model_results.csv"
    )
    parallel_flow = _read_if(
        run_root / "08_synchronized_crossmodal"
        / "parallel_window_sample_flow.csv"
    )
    parallel_stability = _read_if(
        run_root / "08_synchronized_crossmodal"
        / "parallel_window_stability.csv"
    )
    eye_evidence = _read_if(
        run_root / "02_eye_stage2" / "18_eye_evidence_classification.xlsx"
    )
    eeg_evidence = _read_if(
        run_root / "06_eeg_primary" / "20_eeg_evidence_classification.csv"
    )
    eye_effect_text = _effect_preview(eye_models, limit=12)
    questionnaire_effect_text = _effect_preview(
        questionnaire_models, limit=10
    )
    eeg_effect_text = _effect_preview(eeg_models, limit=12)
    order_effect_text = _effect_preview(order_comparison, limit=12)
    crossmodal_text = _effect_preview(crossmodal, limit=8)
    timebin_report = timebin.loc[
            timebin.get("hypothesis_family", pd.Series("", index=timebin.index))
            .astype(str).str.startswith("H_time_")
        ].copy()
    timebin_sort = [
        column for column in ("outcome", "term", "onset_trim_s")
        if column in timebin_report
    ]
    if timebin_sort:
        timebin_report = timebin_report.sort_values(timebin_sort, kind="stable")
    timebin_text = _markdown_preview(
        timebin_report,
        [
            "onset_trim_s", "outcome", "term", "estimate", "ci_low",
            "ci_high", "p_value", "p_fdr_bh_family", "p_fdr_bh_parallel",
            "n_subjects", "n_trials", "status",
        ],
        48,
    )
    parallel_flow_text = _markdown_preview(
        parallel_flow,
        [
            "onset_trim_s", "participants_before_common_filter",
            "trials_before_common_filter", "timebin_rows_before_common_filter",
            "participants_common", "trials_common", "timebin_rows_common",
        ],
        8,
    )
    stability_sort = [
        column for column in ("outcome", "term")
        if column in parallel_stability
    ]
    if stability_sort:
        parallel_stability = parallel_stability.sort_values(
            stability_sort, kind="stable"
        )
    parallel_stability_text = _markdown_preview(
        parallel_stability,
        [
            "outcome", "term", "hypothesis_family", "windows_estimated",
            "direction_consistent", "window_significant_count",
            "parallel_significant_count", "estimate_min", "estimate_max",
            "classification",
        ],
        40,
    )
    otheta_wwr75 = timebin.loc[
        timebin.get("outcome", pd.Series("", index=timebin.index)).astype(str)
        .str.fullmatch("eeg_O_theta", case=False, na=False)
        & timebin.get("term", pd.Series("", index=timebin.index)).astype(str)
        .str.contains("WWR.*75.*scene_time_norm", case=False, regex=True, na=False)
    ].copy()
    otheta_wwr75_text = _markdown_preview(
        otheta_wwr75,
        [
            "onset_trim_s", "estimate", "ci_low", "ci_high", "p_value",
            "p_fdr_bh_family", "p_fdr_bh_parallel", "n_subjects",
            "n_trials", "status",
        ],
        8,
    )
    evidence_text = _markdown_preview(
        pd.concat([
            eye_evidence.assign(Modality="Eye"),
            eeg_evidence.assign(Modality="EEG"),
        ], ignore_index=True, sort=False),
        [
            "Modality", "Outcome", "outcome", "EvidenceGrade",
            "evidence_grade", "Rule",
        ],
        20,
    )
    text = f"""# 论文数据分析结果报告

> 当前正式老师流程运行：`{run_root.name}`  
> 本报告是结果文件夹的阅读入口；所有判断以所列机器可读表为准。

## 一、完整数据分析结果

### 1. 数据来源与样本流

- 眼动为独立模态样本。Stage 1 审核 57 名候选、12 个真实场景/AOI；Stage 2 按 60% 主阈值实际保留 {eye_valid_participants} 人、{eye_valid_trials} 个试次。来源：`12_teacher_analysis/02_eye_stage2/04_participant_quality_summary.xlsx`。
- EEG 的 42 人、{int(structural['Trials'])} 个试次用于结构完整性审核；场景级 QC 后，正式模型使用 {int(model['Participants'])} 人、{int(model['Trials'])} 个试次。来源：`12_teacher_analysis/06_eeg_primary/03b_eeg_sample_flow.xlsx`。
- 问卷正式口径使用 EEG 有效的 42 人，共 {int(questionnaire_audit['Trials'])} 个参与者×场景试次；入口文件为 `E:\\26\\补\\VR+EEG实验问卷-总-原始数据-文字.xlsx`。来源：`12_teacher_analysis/07_reviewer_analysis/Questionnaire_42_person_sample_audit.xlsx`。
- 眼动主分析没有因缺少 EEG 而丢弃合格眼动试次。统一样本稳健性使用精确 Participant × GlobalTrialOrder 交集，实际为 {int(common['Participants'])} 人、{int(common['Trials'])} 个试次。来源：`12_teacher_analysis/02_eye_stage2/16b_EEG_valid_common_sample_counts.xlsx`。
- 时序同步表经本次复核包含 {int(sync_row['Participants'])} 人、{int(sync_row['Trials'])} 个 Participant × Trial、{int(sync_row['TimeBins'])} 行 time-bin。来源：`12_teacher_analysis/08_synchronized_crossmodal/synchronized_crossmodal_sample_and_source.xlsx`。
- 42名 EEG 有效参与者中有 {no_clock_count} 人缺少通过验证的时钟缓存/样本导出，因此同步分析不会强行补值；逐人原因见 `12_teacher_analysis/08_synchronized_crossmodal/eeg_clock_coverage_audit.xlsx`。

### 2. 问卷结果

问卷从最新入口重新准备了 57 人×12场景的完整表，并按 IncludeEEGValid 固定出 42 人/504试次正式样本。S1–S5 以参与者随机截距 LMM、CR2、BH-FDR 与 outcome内 Holm 重新分析；没有 OLS 回退。42 人明细见 `12_teacher_analysis/07_reviewer_analysis/Questionnaire_42_person_formal_sample.xlsx`，正式结果见同目录 `Questionnaire_42_results_report.md`。

{questionnaire_effect_text}

历史问卷产物只在与当前源数据重合键和值一致时作为兼容证据登记，复用核验见顶层 `artifact_reuse_manifest.xlsx`。问卷不参与眼动主样本筛选，也不会把眼动 A 人样本强行缩减为 42 人。

### 3. 眼动 QC 与六个核心结局

60% 是唯一主阈值，50% 与 70% 只作敏感性。每个 fixation 唯一归入 Table、Window、Equipment、Background 或 OffStimulus；C0 的 Equipment 为 structural NA；share 分母只使用 ValidScene TFD。核心模型按三水平分类 WWR、Complexity、ExperienceGroup、Gender、Block、Position 和三个 OrderGroup 运行，且没有 OLS 回退。

下表列出按校正后或原始 p 值排序的核心固定效应预览；完整结果及所有非显著项见 `12_teacher_analysis/02_eye_stage2/09_familyA_primary_models.xlsx` 与 `10_familyB_primary_models.xlsx`。

{eye_effect_text}

### 3.1 六场景AOI与Fig. 6眼动热图

{eye_figure_text}

主图 `12_teacher_analysis/09_eye_figures/Figure6_fixation_event_density.svg` 使用fixation事件等权密度；`FigureS_fixation_duration_density.svg` 为FixationDuration加权敏感性图。六个面板使用相同核宽与共享的蓝—绿—黄—红Tobii风格色标，红色表示最高密度。统一AOI图固定使用Table `#DC2D2D`、Window `#EEC428`、Equipment `#2E6ADC`、ValidScene `#28AA5A`；正式展示使用带白色衬边的加粗语义色轮廓和10%同色透明填充，在增强可辨识度的同时保留场景细节，场景外像素设为白色。完整配准矩阵、阈值QA及纳入明细均保存在同目录。

### 4. EEG 正式主分析

EEG relative power 是预先固定的主分析；log10 absolute power 仅作为敏感性，不按显著性切换。主模型只含老师指定的二阶交互，并控制 Gender、Block、Position 与三水平 OrderGroup。CR2、5000 次参与者聚类 Bootstrap、完整 12 试次、Block 1、Previous condition、LOO、性别和极端值结果均独立保存。

{eeg_effect_text}

来源：`12_teacher_analysis/06_eeg_primary/04_eeg_primary_models_BH.xlsx`；95% CI、raw p、BH-FDR 和 Holm 事后比较分别保留在正式表中。

### 5. 顺序、时间与 carryover

Model 0 与时间/顺序调整后的 Model 1 并列报告；变化百分比、方向、95% CI 与 CR2 共同用于判断，不把显著性是否跨过 0.05 当成唯一标准。

{order_effect_text}

来源：`12_teacher_analysis/05_eeg_order/06_eeg_model0_model1_comparison.xlsx`。同 Block 前序条件和 Block 1 结果分别见同目录 `07`、`08`、`09` 号表。

### 6. 精确跨模态与同步 time-bin

眼动—EEG 场景级跨模态模型只使用精确 Participant × GlobalTrialOrder 交集，并限制为预设 O_theta、O_alpha 两个 EEG 指标。

{crossmodal_text}

真正时序同步分析只接受 `eeg_temporal_resolution=window_specific` 且通过绝对时钟 QC 的表。0、5、10、15 s是同等地位的并行窗口，四组使用共同 Participant × Trial 集合，并按原场景 `scene_time_norm` 时间轴摆放；不从四组中事后选择一个窗口作为主分析。共同样本流如下：

{parallel_flow_text}

四窗口模型同时报告窗口内 BH-FDR (`p_fdr_bh_family`) 与四窗口联合 BH-FDR (`p_fdr_bh_parallel`)：

{timebin_text}

### 7. 证据分级与解释边界

{evidence_text}

WWR 始终按三水平分类变量处理；本结果不支持倒 U、连续非线性或“最优 WWR”结论。ExperienceGroup 严格沿用代码仓库 Q1.4 的高低经验分类，不使用老师材料中误写的 ExerciseFrequency。

## 二、Reviewer专项结果

### Reviewer 1 意见11

顺序与时间证据在0、5、10、15 s四个并行窗口中同时报告未调整 Model 0、加入 Block/Position/三个 OrderGroup 的 Model 1、同 Block PreviousWWR/PreviousComplexity、Block 1 与 CR2。具体 Estimate、95% CI、raw p 和 adjusted p 见 `12_teacher_analysis/07_reviewer_analysis/Reviewer1_comment11_order_time_carryover.xlsx`。结论只描述调整前后方向和幅度稳定性，不声称“完全排除顺序效应”。

可直接用于回复信的边界表述：在控制 Block、试次位置与随机顺序组，并检查同 Block 前序场景、Block 1 和 CR2 后，报告效应方向与幅度的稳定程度；这些分析降低了顺序/累积暴露混杂的可能性，但不能证明顺序效应绝对不存在。

### Reviewer 2 意见5

EEG alpha/theta 以及已有眨眼、瞳孔、扫描路径指标只作为时间变化代理。实验没有直接疲劳量表，因此代理变化不能直接等同于疲劳。证据表见 `12_teacher_analysis/07_reviewer_analysis/Reviewer2_comment5_fatigue_proxy_evidence.xlsx`。

可直接用于回复信的边界表述：补充分析检查了 alpha/theta、眨眼、瞳孔和扫描路径代理随 Block/Position 的变化，并比较环境效应在时间调整前后的稳定性；由于研究未采集直接疲劳量表，这些结果只能说明时间相关生理或行为变化，不能被解释为对疲劳的直接测量。

## 三、针对老师四窗口新方案的分析结果

### 1. 四组数据如何对齐

四组数据具有同一个场景起始点。0 s保留全部开头数据，5、10、15 s分别删除场景开始后的前5、10、15 s；删除后的数据不向前平移，仍保留在原始绝对时钟和原场景时间位置。眼动和EEG在每个2 s time-bin中使用完全相同的 `bin_start_epoch_ms` 与 `bin_end_epoch_ms`。

### 2. 四窗口共同样本

{parallel_flow_text}

只有同时存在于四个窗口、通过绝对时钟QC、EEG场景QC和眼动60% Stage 2 QC的Participant × Trial进入并行比较。这样可避免样本变化被误写为窗口效应。

### 3. 四窗口稳定性分类

{parallel_stability_text}

`parallel_stable_evidence`表示四窗口方向一致且均通过四窗口联合校正；`window_specific_evidence`表示只有部分窗口通过窗口内校正；`direction_consistent_insufficient_evidence`表示方向相同但统计证据不足；`window_unstable`表示方向随剔除长度改变；`not_estimable_all_windows`不能被解释为“无效应”。

### 4. O_theta的WWR75时间轨迹

{otheta_wwr75_text}

这四行必须并列解释。“方向一致”只表示Estimate符号相同，不等于显著性一致；只有相应四窗口联合校正结果支持时，才写成跨窗口稳定证据。

旧结果是否复用、复用理由及核验方式见顶层 `artifact_reuse_manifest.xlsx`。整个文件夹的逐文件哈希索引见 `结果文件总索引.xlsx`。老师任务的逐项状态见 `老师任务完成矩阵.xlsx`。

### 解释边界

这是一项观察/实验条件比较分析。统计关联不能自动升级为认知机制或疲劳因果结论；跨模态结果必须以精确交集、同步质量和预设稳定 EEG 指标为前提。证据分级为 Robust、Partially robust、Exploratory 或 Unsupported，不能只由 p<0.05 决定。
"""
    return write_text(text, outputs_root / "论文数据分析结果报告.md")


def _promote(run_root: Path, outputs_root: Path) -> Path:
    target = outputs_root / "12_teacher_analysis"
    folders = [
        *STAGE_FOLDERS.values(), "07_reviewer_analysis",
        "08_synchronized_crossmodal", "09_eye_figures", "99_completion",
    ]
    backup_root = (
        outputs_root / "teacher_runs" / "_promoted_backups"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    for folder in folders:
        source = run_root / folder
        if source.is_dir():
            destination = target / folder
            if destination.is_dir():
                backup_root.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(backup_root / folder))
            shutil.copytree(source, destination)
    return target


def run_all_results(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outputs_root: str | Path,
    run_id: str,
    repo_root: str | Path,
    self_review: bool,
    reuse_valid: bool,
    resume: bool,
    promote: bool,
    r_required: bool = True,
) -> dict[str, Path]:
    root = Path(outputs_root)
    run_root = root / "teacher_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root)
    config_path = Path(config_path)
    config.setdefault("eye", {})
    config.setdefault("eeg", {})
    config.setdefault("stage3", {})
    config["eye"]["provisional_user_authorization"] = bool(self_review)
    for command, folder in STAGE_FOLDERS.items():
        if command == "eye-stage1":
            config["eye"]["stage1_dir"] = str(run_root / folder)
        elif command == "eye-stage2":
            config["eye"]["stage2_dir"] = str(run_root / folder)
        elif command == "eye-stage3-plan":
            config["eye"]["stage3_plan_dir"] = str(run_root / folder)
        elif command == "eeg-order":
            config["eeg"]["order_stage_dir"] = str(run_root / folder)
    stage_reuse = (
        _import_reusable_stages(root, run_root, repo)
        if reuse_valid else pd.DataFrame()
    )
    questionnaire_outputs = _build_questionnaire_teacher_audit(
        root, run_root, config, repo, r_required
    )
    config["stage3"]["s3_trial_file"] = str(
        questionnaire_outputs["complete"]
    )
    (run_root / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outputs: dict[str, Path] = {}
    _run_or_resume(
        name="eye-stage1", function=run_eye_stage1, config=config,
        config_path=config_path, outdir=run_root / "01_eye_stage1",
        repo_root=repo, resume=resume,
    )
    _run_or_resume(
        name="eye-stage2", function=run_eye_stage2, config=config,
        config_path=config_path, outdir=run_root / "02_eye_stage2",
        repo_root=repo, resume=resume, r_required=r_required,
    )
    figure_outputs = build_eye_scene_figures(
        run_root=run_root,
        mapping_file=Path(config["scene_aoi_mapping"]),
        output_dir=run_root / "09_eye_figures",
        manual_registration_file=Path(
            config.get(
                "eye_scene_registration_file",
                repo / "configs" / "eye_scene_registration.json",
            )
        ),
    )
    outputs.update({f"eye_figure_{name}": path for name, path in figure_outputs.items()})
    _run_or_resume(
        name="eye-stage3-plan", function=run_eye_stage3_plan, config=config,
        config_path=config_path, outdir=run_root / "03_eye_stage3_plan",
        repo_root=repo, resume=resume,
    )
    if self_review:
        _approve_stage3(run_root / "03_eye_stage3_plan")
    _run_or_resume(
        name="eye-stage3-run", function=run_eye_stage3, config=config,
        config_path=config_path, outdir=run_root / "04_eye_stage3",
        repo_root=repo, resume=resume, r_required=r_required,
    )
    _run_or_resume(
        name="eeg-order", function=run_eeg_order, config=config,
        config_path=config_path, outdir=run_root / "05_eeg_order",
        repo_root=repo, resume=resume, r_required=r_required,
    )
    _run_or_resume(
        name="eeg-primary", function=run_eeg_primary, config=config,
        config_path=config_path, outdir=run_root / "06_eeg_primary",
        repo_root=repo, resume=resume, r_required=r_required,
    )
    _build_reviewer_outputs(root, run_root)
    _build_sync_outputs(root, run_root, config)
    completion_dir = run_root / "99_completion"
    completion_dir.mkdir(parents=True, exist_ok=True)
    matrix = _completion_matrix(run_root)
    outputs["run_completion"] = write_table(
        matrix, completion_dir / "teacher_task_completion_matrix.xlsx"
    )
    if not bool(matrix["Exists"].all() and matrix["NonEmpty"].all()):
        missing = matrix.loc[
            ~(matrix["Exists"] & matrix["NonEmpty"]), "RunRelativePath"
        ].tolist()
        raise RuntimeError(f"Teacher result package is incomplete: {missing}")
    if reuse_valid:
        reuse = _reuse_manifest(root, config)
        if not stage_reuse.empty:
            stage_reuse = stage_reuse.rename(columns={
                "Stage": "Artifact",
                "ReuseReason": "ReuseReason",
            })
            stage_reuse["Purpose"] = "teacher-stage validated reuse"
            stage_reuse["Exists"] = stage_reuse["Destination"].map(
                lambda value: Path(str(value)).is_dir()
            )
            reuse = pd.concat([
                reuse,
                stage_reuse.reindex(columns=reuse.columns),
            ], ignore_index=True, sort=False)
    else:
        reuse = pd.DataFrame(columns=[
            "Artifact", "Purpose", "Exists", "ReuseDecision", "ReuseReason"
        ])
    outputs["reuse_manifest"] = write_table(
        reuse, root / "artifact_reuse_manifest.xlsx"
    )
    if promote:
        outputs["promoted"] = _promote(run_root, root)
    outputs["completion_matrix"] = write_table(
        matrix, root / "老师任务完成矩阵.xlsx"
    )
    outputs["report"] = _build_top_report(root, run_root)
    questionnaire_summary = read_table(
        run_root / "07_reviewer_analysis"
        / "Questionnaire_42_person_sample_audit.xlsx"
    ).iloc[0]
    eye_summary = read_table(
        run_root / "02_eye_stage2"
        / "04_participant_quality_summary.xlsx"
    )
    common_summary = read_table(
        run_root / "02_eye_stage2"
        / "16b_EEG_valid_common_sample_counts.xlsx"
    ).iloc[0]
    eeg_summary = read_table(
        run_root / "06_eeg_primary" / "03b_eeg_sample_flow.xlsx"
    )
    sync_summary = read_table(
        run_root / "08_synchronized_crossmodal"
        / "synchronized_crossmodal_sample_and_source.xlsx"
    ).iloc[0]
    clock_coverage_summary = read_table(
        run_root / "08_synchronized_crossmodal"
        / "eeg_clock_coverage_audit.xlsx"
    )
    eye_grade = _read_if(
        run_root / "02_eye_stage2"
        / "18_eye_evidence_classification.xlsx"
    )
    eeg_grade = _read_if(
        run_root / "06_eeg_primary"
        / "20_eeg_evidence_classification.csv"
    )
    bootstrap_failures = _read_if(
        run_root / "06_eeg_primary"
        / "08_eeg_bootstrap_failures.csv"
    )
    summary = {
        "status": "complete",
        "run_id": run_id,
        "run_root": str(run_root),
        "promoted": bool(promote),
        "formal_teacher_root": str(root / "12_teacher_analysis"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "self_review": self_review,
        "reuse_valid": reuse_valid,
        "resume": resume,
        "method_contract_hash": method_contract_hash(repo),
        "completion_requirements": int(len(matrix)),
        "completion_requirements_passed": int(
            (matrix["Exists"] & matrix["NonEmpty"]).sum()
        ),
        "questionnaire": {
            "source": str(questionnaire_summary["QuestionnaireSource"]),
            "source_sha256": str(
                questionnaire_summary["QuestionnaireSourceSHA256"]
            ),
            "prepared_participants": int(
                questionnaire_summary["PreparedParticipantsAll"]
            ),
            "prepared_trials": int(
                questionnaire_summary["PreparedTrialsAll"]
            ),
            "formal_participants": int(
                questionnaire_summary["Participants"]
            ),
            "formal_trials": int(questionnaire_summary["Trials"]),
        },
        "eye": {
            "candidates": 57,
            "scenes_and_aoi": 12,
            "primary_threshold": 0.60,
            "primary_participants": int(
                (eye_summary["PrimaryValidTrials"] > 0).sum()
            ),
            "primary_trials": int(eye_summary["PrimaryValidTrials"].sum()),
            "exact_eeg_common_participants": int(
                common_summary["Participants"]
            ),
            "exact_eeg_common_trials": int(common_summary["Trials"]),
            "evidence_grade_counts": (
                eye_grade["EvidenceGrade"].value_counts().to_dict()
                if "EvidenceGrade" in eye_grade else {}
            ),
        },
        "eeg": {
            "structural_participants": int(
                eeg_summary.iloc[0]["Participants"]
            ),
            "structural_trials": int(eeg_summary.iloc[0]["Trials"]),
            "scene_qc_participants": int(
                eeg_summary.iloc[1]["Participants"]
            ),
            "scene_qc_trials": int(eeg_summary.iloc[1]["Trials"]),
            "evidence_grade_counts": (
                eeg_grade["grade"].value_counts().to_dict()
                if "grade" in eeg_grade else {}
            ),
            "bootstrap_failed_replicates": int(
                pd.to_numeric(
                    bootstrap_failures.get("failed_replicates"),
                    errors="coerce",
                ).fillna(0).sum()
            ) if not bootstrap_failures.empty else 0,
        },
        "synchronized_timebin": {
            "eeg_valid_participants_without_validated_clock": int(
                (~clock_coverage_summary[
                    "ValidatedClockSynchronizedDataPresent"
                ].map(is_truthy)).sum()
            ),
            "participants": int(sync_summary["Participants"]),
            "trials": int(sync_summary["Trials"]),
            "timebins": int(sync_summary["TimeBins"]),
            "temporal_resolution": str(
                sync_summary["EEGTemporalResolution"]
            ),
            "analysis_status": str(sync_summary["AnalysisStatus"]),
        },
    }
    outputs["summary"] = write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        root / "realdata_run_summary.json",
    )
    outputs["readme"] = write_text(
        "# 当前有效数据分析结果\n\n"
        f"当前正式运行：`teacher_runs/{run_id}`。\n\n"
        "主阅读入口为 `论文数据分析结果报告.md`；老师规范阶段结果位于 "
        "`12_teacher_analysis`；逐文件哈希见 `结果文件总索引.xlsx`。\n",
        root / "README_当前有效结果.md",
    )
    write_run_manifest(
        completion_dir,
        stage="all-results",
        fingerprint=file_sha256(run_root / "resolved_config.json"),
        config_path=config_path,
        arguments={
            "outputs_root": str(root), "run_id": run_id,
            "self_review": self_review, "reuse_valid": reuse_valid,
            "resume": resume, "promote": promote,
        },
        repo_root=repo,
        extra={"status": "complete"},
    )
    outputs["index"] = write_table(
        _all_files_index(root), root / "结果文件总索引.xlsx"
    )
    return outputs
