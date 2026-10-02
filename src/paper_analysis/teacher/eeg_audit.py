"""Independent, non-promoting EEG validation with explicit hypothesis families."""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .contracts import build_modality_registry, canonicalize_trials
from .eeg import _scene_qc_mask
from .state import StageBlockedError, file_sha256, input_hash_records, write_run_manifest
from paper_analysis.utils.io import is_truthy, read_table

TRIMS = (0, 5, 10, 15)
METRICS = tuple(f"{roi}_{band}" for roi in ("F", "P", "O") for band in ("theta", "alpha", "beta"))
CORE = ("O_theta", "F_theta", "O_alpha", "O_beta")
TEMPORAL = ("Block", "PositionWithinBlockCentered")
CONDITION = (
    "WWRWWR45", "WWRWWR75", "ComplexityC1", "ExperienceGroupLow",
    "WWRWWR45:ComplexityC1", "WWRWWR75:ComplexityC1",
    "WWRWWR45:ExperienceGroupLow", "WWRWWR75:ExperienceGroupLow",
    "ComplexityC1:ExperienceGroupLow",
)
PREVIOUS = ("PreviousWWRWWR45", "PreviousWWRWWR75", "PreviousComplexityC1")
FACTOR_LEVELS = {
    "WWR": ["WWR15", "WWR45", "WWR75"], "Complexity": ["C0", "C1"],
    "ExperienceGroup": ["High", "Low"], "Gender": ["Female", "Male"],
    "OrderGroup": ["new order2", "order1", "order2"],
    "PreviousWWR": ["WWR15", "WWR45", "WWR75"], "PreviousComplexity": ["C0", "C1"],
}
DESIGN = ("WWR", "Complexity", "Gender", "ExperienceGroup", "OrderGroup",
          "Block", "PositionWithinBlockCentered")
KEYS = ["Participant", "GlobalTrialOrder"]
METHOD_FILES = (
    "src/paper_analysis/teacher/eeg_audit.py", "src/paper_analysis/teacher/eeg.py",
    "src/paper_analysis/teacher/contracts.py", "src/paper_analysis/teacher/state.py",
    "src/paper_analysis/utils/io.py", "src/paper_analysis/utils/coding.py",
    "analysis/r/eeg_audit.R", "analysis/r/eeg_factor_tests.R", "analysis/r/common.R", "analysis/r/renv.lock",
    "scripts/run_teacher_analysis.py", "scripts/portable_rscript.cmd",
    "scripts/validate_eeg_audit.py",
)


def outcome_spec() -> pd.DataFrame:
    return pd.DataFrame([
        {"outcome": f"{m}_relative" if scale == "relative_power" else f"log10_{m}_absolute",
         "metric": m, "scale": scale, "core": scale == "relative_power" and m in CORE}
        for scale in ("relative_power", "log10_absolute_power") for m in METRICS
    ])


def sample_hash(frame: pd.DataFrame) -> str:
    keys = sorted((str(a), int(b)) for a, b in frame[KEYS].itertuples(index=False, name=None))
    return hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode()).hexdigest()


def prepare_inputs(config: dict[str, Any]) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Form order covariates before QC, then intersect all windows/outcomes."""
    eeg = config["eeg"]
    if sorted(eeg.get("onset_trim_variants_s", [])) != list(TRIMS):
        raise StageBlockedError("eeg-audit requires four windows: 0, 5, 10, 15")
    if eeg.get("onset_trim_strategy") != "parallel":
        raise StageBlockedError("eeg-audit requires parallel window status")
    if tuple(eeg.get("core_metrics", CORE)) != CORE:
        raise StageBlockedError("eeg-audit requires the documented four core metrics in order")
    raw = read_table(eeg["onset_sensitivity_trial_file"])
    if "onset_trim_s" not in raw:
        raise StageBlockedError("Missing onset_trim_s")
    observed = pd.to_numeric(raw.onset_trim_s, errors="coerce")
    if observed.isna().any() or set(observed) != set(TRIMS):
        raise StageBlockedError("Input windows do not match the four-window contract")
    registry = build_modality_registry(config["participant_information"])
    prepared = {}; exclusions = []; valid_sets = []
    for trim in TRIMS:
        frame = canonicalize_trials(raw.loc[observed.eq(trim)].copy())
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise StageBlockedError(f"Invalid/duplicate trial key in window {trim}")
        if not frame.Block.isin([1, 2]).all() or not frame.PositionWithinBlock.isin(range(1, 7)).all():
            raise StageBlockedError("Invalid block or position")
        if not frame.GlobalTrialOrder.eq((frame.Block - 1) * 6 + frame.PositionWithinBlock).all():
            raise StageBlockedError("Trial key disagrees with block/position")
        covariates = [c for c in ("Gender", "ExperienceGroup", "OrderGroup", "IncludeEEGValid") if c not in frame]
        frame = frame.merge(registry[["Participant", *covariates]], on="Participant", how="left", validate="many_to_one")
        missing = set(DESIGN) - set(frame)
        required_power = [f"{m}_{scale}" for m in METRICS for scale in ("relative", "absolute")]
        missing |= set(required_power) - set(frame)
        if missing:
            raise StageBlockedError(f"Missing required fields: {sorted(missing)}")
        reasons: dict[str, pd.Series] = {
            "participant_excluded": ~frame.IncludeEEGValid.map(is_truthy),
            "scene_qc_excluded": ~_scene_qc_mask(frame),
        }
        for field in DESIGN:
            if field in FACTOR_LEVELS:
                reasons[f"invalid_design:{field}"] = ~frame[field].isin(FACTOR_LEVELS[field])
            else:
                reasons[f"invalid_design:{field}"] = ~np.isfinite(pd.to_numeric(frame[field], errors="coerce"))
        for m in METRICS:
            relative = pd.to_numeric(frame[f"{m}_relative"], errors="coerce")
            absolute = pd.to_numeric(frame[f"{m}_absolute"], errors="coerce")
            reasons[f"invalid_relative:{m}"] = ~np.isfinite(relative)
            reasons[f"nonpositive_or_missing_absolute:{m}"] = ~np.isfinite(absolute) | absolute.le(0)
            frame[f"{m}_relative"] = relative
            frame[f"log10_{m}_absolute"] = np.log10(absolute.where(absolute.gt(0)))
        invalid = pd.DataFrame(reasons).any(axis=1)
        for label, mask in reasons.items():
            selected = frame.loc[mask, KEYS].copy()
            selected["onset_trim_s"] = trim; selected["reason"] = label
            exclusions.append(selected)
        prepared[trim] = frame
        valid_sets.append(set(frame.loc[~invalid, KEYS].itertuples(index=False, name=None)))
    common = set.intersection(*valid_sets)
    if not common:
        raise StageBlockedError("No complete trial keys shared by all 18 outcomes and four windows")
    for trim, frame in prepared.items():
        keep = pd.Series([key in common for key in frame[KEYS].itertuples(index=False, name=None)], index=frame.index)
        removed = frame.loc[~keep, KEYS].copy()
        removed["onset_trim_s"] = trim; removed["reason"] = "not_in_four_window_complete_sample"
        exclusions.append(removed)
        prepared[trim] = frame.loc[keep].reset_index(drop=True)
    # Fixed covariates and predecessor identities must agree across windows.
    reference = prepared[0].set_index(KEYS)[[*DESIGN, "PreviousWWR", "PreviousComplexity"]].sort_index()
    for trim in TRIMS[1:]:
        candidate = prepared[trim].set_index(KEYS)[reference.columns].sort_index()
        if not candidate.equals(reference):
            raise StageBlockedError(f"Design/predecessors disagree across windows: {trim}")
    return prepared, pd.concat(exclusions, ignore_index=True)


def family_tables(coefficients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize every expected hypothesis; never reduce BH n after failures."""
    families = [
        ("temporal_relative", "relative_power", [f"{m}_relative" for m in METRICS], "Model1", TEMPORAL),
        ("temporal_absolute", "log10_absolute_power", [f"log10_{m}_absolute" for m in METRICS], "Model1", TEMPORAL),
        ("condition_core_relative", "relative_power", [f"{m}_relative" for m in CORE], "Model1", CONDITION),
        ("condition_expanded_relative", "relative_power", [f"{m}_relative" for m in METRICS], "Model1", CONDITION),
        ("condition_expanded_absolute", "log10_absolute_power", [f"log10_{m}_absolute" for m in METRICS], "Model1", CONDITION),
        ("previous_core_relative", "relative_power", [f"{m}_relative" for m in CORE], "PreviousScene", PREVIOUS),
    ]
    tables = []; status = []
    keys = ["onset_trim_s", "outcome", "model", "term"]
    if coefficients.duplicated(keys).any():
        raise StageBlockedError("Duplicate coefficient identity")
    for name, scale, outcomes, model, terms in families:
        grid = pd.DataFrame(itertools.product(TRIMS, outcomes, [model], terms), columns=keys)
        table = grid.merge(coefficients.drop(columns="scale", errors="ignore"), on=keys, how="left", validate="one_to_one")
        table["family_id"] = name; table["scale"] = scale
        p = pd.to_numeric(table["p.value"], errors="coerce")
        eligible = p.between(0, 1) & np.isfinite(p)
        if "inference_valid" in table:
            eligible &= table.inference_valid.eq(True)
        table["within_q"] = np.nan; table["joint_q"] = np.nan
        for scope, members in [("joint", table.index), *[(f"window_{t}", table.index[table.onset_trim_s.eq(t)]) for t in TRIMS]]:
            complete = bool(eligible.loc[members].all())
            status.append({"family_id": name, "scope": scope, "expected_tests": len(members),
                           "valid_tests": int(eligible.loc[members].sum()),
                           "status": "complete" if complete else "incomplete_no_final_q"})
            if complete:
                column = "joint_q" if scope == "joint" else "within_q"
                table.loc[members, column] = multipletests(p.loc[members].to_numpy(), method="fdr_bh")[1]
        table["corrected_positive"] = (table.joint_q.lt(0.05) & table.estimate.gt(0)).astype("boolean").mask(table.joint_q.isna())
        table["corrected_detected"] = table.joint_q.lt(0.05).astype("boolean").mask(table.joint_q.isna())
        tables.append(table)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(status)


def validate_provenance(config: dict[str, Any]) -> list[str]:
    eeg = config["eeg"]
    paths = [eeg["onset_sensitivity_trial_file"], config["participant_information"], eeg["preprocessing_audit_file"]]
    for path in paths:
        if not Path(path).is_file():
            raise StageBlockedError(f"Missing input: {path}")
    audit = read_table(paths[-1])
    if not {"Parameter", "Value"}.issubset(audit.columns):
        raise StageBlockedError("Preprocessing audit must contain Parameter and Value")
    values = audit.set_index("Parameter")["Value"].astype(str)
    declared = str(eeg.get("preprocessing_parameters", {}).get("relative_power_denominator_hz", ""))
    if declared != "1-45" or values.get("relative_power_denominator_hz") != declared:
        raise StageBlockedError("Relative denominator metadata must match audited 1-45 Hz source; no power recomputation is performed")
    if not eeg.get("preprocessing_confirmed", False):
        raise StageBlockedError("Preprocessing provenance is not confirmed")
    return paths


def run_eeg_audit(config: dict[str, Any], *, config_path: str | Path, outdir: str | Path,
                  repo_root: str | Path, r_required: bool = True) -> dict[str, Path]:
    if not r_required:
        raise StageBlockedError("eeg-audit requires R; use --dry-run for a read-only preflight")
    root = Path(repo_root); out = Path(outdir)
    if out.exists():
        raise StageBlockedError("Use a new validation output directory; overwrite/cache reuse is prohibited")
    paths = validate_provenance(config)
    prepared, exclusions = prepare_inputs(config)
    source_records = input_hash_records([*paths, str(config_path)])
    code_records = [{"path": p, "sha256": file_sha256(root/p)} for p in METHOD_FILES if (root/p).is_file()]
    contract = {"version": 1, "purpose": "code_validation_not_formal_results", "historically_prespecified": False,
                "windows": TRIMS, "core_metrics": CORE, "factor_levels": FACTOR_LEVELS,
                "temporal_terms": TEMPORAL, "condition_terms": CONDITION, "previous_terms": PREVIOUS,
                "sample_policy": "four_window_all_18_outcome_complete_cases", "final_q": "joint_q", "alpha": 0.05,
                "ci": "CR2 Satterthwaite 95%, unadjusted", "seed": 20260906,
                "input_records": source_records, "code_records": code_records, "config": config}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    (out/"analysis_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    outcome_spec().to_csv(out/"outcomes.csv", index=False)
    exclusions.to_csv(out/"exclusions.csv", index=False, encoding="utf-8-sig")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    (out/"git_status.txt").write_text(dirty, encoding="utf-8")
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True).stdout
    (out/"tracked_changes.patch").write_bytes(diff)
    # Include untracked implementation files as well as tracked code.
    for record in code_records:
        target = out/"code_snapshot"/record["path"]
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((root/record["path"]).read_bytes())
    manifest_args = dict(stage="eeg-audit", fingerprint=fingerprint, config_path=config_path,
                         arguments={"outdir": str(out)}, repo_root=root)
    write_run_manifest(out, **manifest_args, extra={"status": "running", "purpose": contract["purpose"], "git_dirty": bool(dirty)})
    try:
        coefficients = []; diagnostics = []; samples = []
        for trim, frame in prepared.items():
            folder = out/f"trim_{trim}s"; folder.mkdir()
            frame.to_csv(folder/"input.csv", index=False, encoding="utf-8-sig")
            command = [str(config.get("rscript", "Rscript")), str(root/"analysis/r/eeg_audit.R"),
                       str(folder/"input.csv"), str(folder), str(out/"outcomes.csv"), str(out/"analysis_contract.json")]
            with (folder/"R_execution.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
            for filename, destination in [("coefficients", coefficients), ("diagnostics", diagnostics), ("model_samples", samples)]:
                try:
                    table = pd.read_csv(folder/f"{filename}.csv")
                except pd.errors.EmptyDataError:
                    table = pd.DataFrame(columns=["outcome", "model", "term", "p.value", "estimate"])
                table["onset_trim_s"] = trim; destination.append(table)
        coef = pd.concat(coefficients, ignore_index=True)
        diag = pd.concat(diagnostics, ignore_index=True)
        sample = pd.concat(samples, ignore_index=True)
        hashes = pd.DataFrame([
            {"onset_trim_s": trim, "outcome": outcome, "model": model, "sample_hash": sample_hash(group)}
            for (trim, outcome, model), group in sample.groupby(["onset_trim_s", "outcome", "model"])
        ], columns=["onset_trim_s", "outcome", "model", "sample_hash"])
        diag = diag.merge(hashes, how="left", on=["onset_trim_s", "outcome", "model"], validate="one_to_one")
        diag["inference_valid"] = diag.fit_status.eq("fit") & diag.converged.eq(True) & diag.singular.eq(False) & diag.rank_deficient.eq(False) & diag.cr2_status.eq("computed")
        coef = coef.merge(diag[["onset_trim_s", "outcome", "model", "inference_valid", "sample_hash"]], on=["onset_trim_s", "outcome", "model"], validate="many_to_one")
        audit, families = family_tables(coef)
        for name, table in [("coefficients", coef), ("diagnostics", diag), ("model_samples", sample), ("family_coefficients", audit), ("family_status", families)]:
            table.to_csv(out/f"{name}.csv", index=False, encoding="utf-8-sig")
        main = diag.loc[diag.model.eq("Model1")]
        alpha = audit.loc[audit.family_id.eq("temporal_relative") & audit.outcome.str.contains("_alpha_")]
        theta = audit.loc[audit.family_id.eq("temporal_relative") & audit.outcome.str.contains("_theta_") & audit.term.eq("Block")]
        alpha.to_csv(out/"alpha_temporal_check.csv", index=False, encoding="utf-8-sig")
        theta.to_csv(out/"theta_round_check.csv", index=False, encoding="utf-8-sig")
        audit.loc[audit.family_id.eq("previous_core_relative")].to_csv(out/"previous_scene_check.csv", index=False, encoding="utf-8-sig")
        final_theta = "UNRESOLVED" if theta.joint_q.isna().any() else f"{theta.joint_q.min():.17g}–{theta.joint_q.max():.17g}"
        summary = {"purpose": contract["purpose"], "model1_expected": 72, "model1_recorded": len(main),
                   "model1_converged": int(main.converged.eq(True).sum()), "model1_inference_valid": int(main.inference_valid.sum()),
                   "supplementary_models": int(diag.model.ne("Model1").sum()),
                   "common_participants": prepared[0].Participant.nunique(), "common_trials": len(prepared[0]),
                   "theta_relative_round_joint_q_range_all_three_rois": final_theta,
                   "family_scopes_complete": int(families.status.eq("complete").sum()), "family_scopes_total": len(families)}
        (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        text = "# EEG代码验证报告\n\n代码验证用途，非正式替代结果。未更新既有分析或正式结果指针。\n\n"
        text += "\n".join(f"- {key}: {value}" for key, value in summary.items())
        text += "\n\n联合q仅在预定检验族完整且诊断通过时发布。奇异拟合与收敛分别记录，未为获得目标结论改变模型。\n"
        text += "\n[逐模型诊断](diagnostics.csv) · [完整系数与校正](family_coefficients.csv) · [检验族状态](family_status.csv) · [分析约定](analysis_contract.json)\n"
        (out/"README.md").write_text(text, encoding="utf-8")
        for record in source_records:
            if file_sha256(record["path"]) != record["sha256"]:
                raise StageBlockedError("Input changed during validation")
        for record in code_records:
            if file_sha256(root/record["path"]) != record["sha256"]:
                raise StageBlockedError("Implementation changed during validation; use a fresh run")
        status = "validated" if len(main) == 72 and diag.inference_valid.all() and families.status.eq("complete").all() else "validation_issues"
        write_run_manifest(out, **manifest_args, extra={"status": status, "purpose": contract["purpose"], "git_dirty": bool(dirty), "summary": summary})
        records = [{"path": p.relative_to(out).as_posix(), "sha256": file_sha256(p)} for p in sorted(out.rglob("*")) if p.is_file()]
        (out/"output_hashes.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        return {"manifest": out/"run_manifest.json"}
    except Exception as exc:
        write_run_manifest(out, **manifest_args, extra={"status": "failed", "purpose": contract["purpose"], "git_dirty": bool(dirty), "error": str(exc)})
        raise
