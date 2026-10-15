"""Paired HF-removal audit with an evidence-gated formal QC revision.

Original and candidate QC always start from the same unfiltered features.
Participant exclusion never feeds back into the metric threshold population.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.eeg.pipeline import apply_eeg_quality_qc
from paper_analysis.utils.io import is_truthy
from .eeg_audit import TRIMS, KEYS, CORE, FACTOR_LEVELS, CONDITION, TEMPORAL, PREVIOUS
from .fresh import dump, git_sha, stage, unique, verify_inputs
from .state import StageBlockedError, file_sha256

TRIAL_KEYS = ["participant_id", "scene_id"]
WINDOW_KEYS = [*TRIAL_KEYS, "onset_trim_s"]
HF = "hf_ratio_20_40Hz"
PRE_FLAG = "trial_qc_flag_before_subject_exclusion"
PRE_REASONS = "trial_qc_reasons_before_subject_exclusion"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def qc_profiles(original):
    if original.get("policy") != "robust" or HF not in original.get("robust_metrics", []):
        raise StageBlockedError("Original QC must explicitly include robust HF")
    revised = copy.deepcopy(original)
    revised["robust_metrics"] = [m for m in original["robust_metrics"] if m != HF]
    return {"Original": copy.deepcopy(original), "NoHF": revised}


def common_membership(frame):
    unique(frame, WINDOW_KEYS)
    if set(frame.onset_trim_s) != set(TRIMS):
        raise StageBlockedError("Exactly four onset-trim variants are required")
    counts = frame.groupby(TRIAL_KEYS).onset_trim_s.nunique()
    if not counts.eq(4).all():
        raise StageBlockedError("Incomplete trial/window grid")
    passed = ~frame.bad_eeg_quality.map(is_truthy)
    passed &= ~frame.eeg_subject_quality_exclusion.map(is_truthy)
    keep = frame.assign(Pass=passed).groupby(TRIAL_KEYS).Pass.all()
    return pd.Series([bool(keep.loc[tuple(k)]) for k in frame[TRIAL_KEYS].to_numpy()], index=frame.index)


def reconstruct_pair(features, original):
    """Freeze thresholds on the entire pre-exclusion population per window."""
    profiles = qc_profiles(original)
    unique(features, WINDOW_KEYS)
    tables, thresholds = {}, []
    for version, profile in profiles.items():
        windows = []
        for trim in TRIMS:
            source = features.loc[features.onset_trim_s.eq(trim)].reset_index(drop=True)
            if source.empty:
                raise StageBlockedError(f"Missing window {trim}")
            result, _, _, limits = apply_eeg_quality_qc(source, profile)
            limits["onset_trim_s"] = trim
            limits["QCVersion"] = version
            thresholds.append(limits)
            windows.append(result)
        data = pd.concat(windows, ignore_index=True)
        data["CommonQCIncluded"] = common_membership(data)
        data["QCVersion"] = version
        tables[version] = data
    thresholds = pd.concat(thresholds, ignore_index=True)
    for trim in TRIMS:
        original_limits = thresholds.loc[thresholds.QCVersion.eq("Original") & thresholds.onset_trim_s.eq(trim)].set_index("metric")
        revised_limits = thresholds.loc[thresholds.QCVersion.eq("NoHF") & thresholds.onset_trim_s.eq(trim)].set_index("metric")
        pd.testing.assert_frame_equal(original_limits.loc[revised_limits.index].drop(columns="QCVersion"),
                                      revised_limits.drop(columns="QCVersion"), check_exact=True)
    old = tables["Original"].set_index(WINDOW_KEYS)
    new = tables["NoHF"].set_index(WINDOW_KEYS).reindex(old.index)
    if ((~old.bad_eeg_quality) & new.bad_eeg_quality).any():
        raise StageBlockedError("Removing HF lost an Original retained trial")
    return tables, thresholds


def impact_tables(tables):
    old, new = tables["Original"], tables["NoHF"]
    overlap, people, flow, differences = [], [], [], []
    for trim in TRIMS:
        a = old.loc[old.onset_trim_s.eq(trim)].copy()
        b = new.loc[new.onset_trim_s.eq(trim)].copy()
        reasons = a[PRE_REASONS].fillna("").str.split(";")
        hf = reasons.map(lambda r: "robust_" + HF in r)
        other = reasons.map(lambda r: any(v and v != "robust_" + HF for v in r))
        row = {"onset_trim": trim, "n_total_trials": len(a), "n_any_flag": int(a[PRE_FLAG].sum()),
               "n_hf_flag": int(hf.sum()), "n_hf_only": int((hf & ~other).sum()), "n_hf_overlap": int((hf & other).sum())}
        for key, value in [("pct_hf_flag_of_all", hf.mean()), ("pct_hf_only_of_all", (hf & ~other).mean()),
                           ("pct_hf_only_of_hf", (hf & ~other).sum() / hf.sum() if hf.any() else np.nan)]:
            row[key] = 100 * value
        for column, labels in {"n_rms_flag": ["robust_rms_mean_uV"], "n_ptp_flag": ["robust_peak_to_peak_uV"],
                "n_duration_flag": ["segment_duration", "segment_valid_duration"], "n_nan_flag": ["nan_fraction"],
                "n_flat_flag": ["flat_fraction"]}.items():
            row[column] = int(reasons.map(lambda r: any(v in r for v in labels)).sum())
        row.update(n_boundary_flag=0, boundary_is_active_criterion=False,
                   n_boundary_status_recorded=int(a.get("near_boundary", pd.Series(0, index=a.index)).map(is_truthy).sum()))
        overlap.append(row)
        a["HFOnly"] = hf & ~other
        b_by = b.set_index("participant_id")
        for participant, g in a.groupby("participant_id"):
            h = b_by.loc[[participant]]
            frac_a, frac_b = float(g[PRE_FLAG].mean()), float(h[PRE_FLAG].mean())
            ea, eb = bool(g.eeg_subject_quality_exclusion.any()), bool(h.eeg_subject_quality_exclusion.any())
            category = "A_original_excluded_nohf_retained" if ea and not eb else "B_both_excluded" if ea and eb else "C_both_retained"
            if eb and not ea:
                raise StageBlockedError("No-HF unexpectedly excludes a new participant")
            people.append({"participant_id": participant, "onset_trim": trim, "total_trials": len(g),
                "n_flagged_original": int(g[PRE_FLAG].sum()), "proportion_flagged_original": frac_a,
                "excluded_original": ea, "n_flagged_no_hf": int(h[PRE_FLAG].sum()),
                "proportion_flagged_no_hf": frac_b, "excluded_no_hf": eb, "n_hf_only": int(g.HFOnly.sum()), "category": category})
    for version, frame in tables.items():
        for trim in (*TRIMS, "common"):
            sub = frame.loc[frame.onset_trim_s.eq(0 if trim == "common" else trim)]
            keep = sub.CommonQCIncluded if trim == "common" else ~sub.bad_eeg_quality & ~sub.eeg_subject_quality_exclusion
            kept = sub.loc[keep]
            flow.append({"QCVersion": version, "onset_trim": trim, "participants": kept.participant_id.nunique(), "trials": len(kept)})
    for trim in (*TRIMS, "common"):
        selected = {}
        for version, frame in tables.items():
            f = frame.loc[frame.onset_trim_s.eq(0 if trim == "common" else trim)]
            mask = f.CommonQCIncluded if trim == "common" else ~f.bad_eeg_quality & ~f.eeg_subject_quality_exclusion
            selected[version] = set(f.loc[mask, TRIAL_KEYS].itertuples(index=False, name=None))
        for kind, keys in [("added", selected["NoHF"] - selected["Original"]), ("removed", selected["Original"] - selected["NoHF"])]:
            for participant, trial in sorted(keys):
                differences.append({"onset_trim": trim, "participant_id": participant, "scene_id": trial, "change": kind,
                    "participant_membership_changed": participant not in {k[0] for k in selected["Original" if kind == "added" else "NoHF"]}})
    diff = pd.DataFrame(differences, columns=["onset_trim", *TRIAL_KEYS, "change", "participant_membership_changed"])
    person_diff = diff.loc[diff.participant_membership_changed].drop(columns="scene_id").drop_duplicates()
    return {"HF_QC_flag_overlap_by_variant": pd.DataFrame(overlap), "HF_QC_participant_impact": pd.DataFrame(people),
            "HF_QC_sample_comparison": pd.DataFrame(flow), "HF_QC_trial_difference": diff,
            "HF_QC_participant_difference": person_diff}


def preflight(config, repo):
    from .runtime_lock import validate_analysis
    cfg = config["hf_qc_revision"]
    source = Path(cfg["source_run"])
    evidence_run = Path(cfg.get("source_evidence_run", source))
    required = [source / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv",
                source / "00_eeg_trial_qc/QC_policy.json", source / "resolved_config.json",
                source / "00_teacher_inputs/participant_information.xlsx",
                source / "00_teacher_inputs/trial_order_mapping.xlsx",
                source / "00_teacher_inputs/scene_AOI_mapping.xlsx",
                source / "00_eeg_trial_qc/preprocessing_source_inventory.csv",
                source / "10_eeg_denominator_sensitivity/main_1_45/family_coefficients.csv",
                source / "10_eeg_denominator_sensitivity/main_1_45/family_factors.csv",
                Path(cfg["hf_formula_proof"]) / "audit_completion.json",
                Path(cfg["hf_formula_proof"]) / "mean_waveform_scale_check.json"]
    required.extend([source / "00_synchronized_source/aligned_synchronized_timebin_table.csv",
                     source / "00_synchronized_source/clock_alignment_scene_qc.csv",
                     source / "02_eye_stage2/eye_stage2_model_input.csv",
                     source / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx",
                     Path(cfg["source_psd_cache"]) / "cache_manifest.json",
                     Path(cfg["manuscript"])])
    for p in required:
        if not p.is_file():
            raise StageBlockedError(f"Missing revision source: {p}")
    # Verify the exact historical outputs used, against their original stage seals.
    for name in ["00_teacher_inputs", "00_eeg_trial_qc", "10_eeg_denominator_sensitivity"]:
        seal_path = evidence_run / "stage_seals" / f"{name}.json"
        seal = read_json(seal_path)
        registered = {r["path"]: r["sha256"] for r in seal["files"]}
        for p in required:
            if p.is_relative_to(source / name):
                if registered.get(p.relative_to(source).as_posix()) != file_sha256(p):
                    raise StageBlockedError(f"Unregistered/changed Original artifact: {p}")
        required.append(seal_path)
    source_records = read_json(source / "source_hashes_before.json")
    waveform_records = [r for r in source_records if Path(r["path"]).suffix.lower() in {".set", ".fdt"}]
    if not waveform_records:
        raise StageBlockedError("No registered waveform source hashes")
    verify_inputs(waveform_records)
    completion = read_json(Path(cfg["hf_formula_proof"]) / "audit_completion.json")
    if completion.get("status") != "passed" or not completion.get("source_hashes_unchanged"):
        raise StageBlockedError("HF formula proof is incomplete")
    for name, expected in completion["outputs"].items():
        p = Path(cfg["hf_formula_proof"]) / name
        if not p.is_file() or file_sha256(p) != expected:
            raise StageBlockedError(f"HF formula proof output changed: {name}")
    # Pairing and upstream reference facts are taken from the validated inventory.
    inventory = pd.read_csv(source / "00_eeg_trial_qc/preprocessing_source_inventory.csv")
    if not inventory.nbchan.eq(8).all() or not inventory.history.str.contains(r"pop_reref\s*\(\s*EEG\s*,\s*\[\s*\]\s*\)", regex=True).all():
        raise StageBlockedError("Not all upstream inputs document eight-channel rereferencing")
    from .fresh import set_record
    for record in inventory.to_dict("records"):
        actual = set_record(record["set_path"], config.get("fresh", {}).get("fdt_aliases"))
        if Path(actual["fdt_path"]).resolve() != Path(record["fdt_path"]).resolve():
            raise StageBlockedError("SET/FDT pairing differs from the registered baseline")
    required.extend([source / "source_hashes_before.json", source / "00_eeg_trial_qc/preprocessing_source_inventory.csv"])
    qc_profiles(read_json(required[1]))
    return {"code_sha": git_sha(repo), "runtime_lock": validate_analysis(config, repo, require_r=True),
            "inputs": [*[{"path": str(p), "sha256": file_sha256(p)} for p in required], *waveform_records],
            "source_run": str(source), "mode": "audit_then_conditional_formal"}


def bound_config(config, source, run, version):
    c = read_json(source / "resolved_config.json")
    c["rscript"] = config["rscript"]
    c["fresh"].update(config.get("fresh", {}))
    for field, name in [("participant_information", "participant_information.xlsx"),
                        ("trial_order_mapping", "trial_order_mapping.xlsx"), ("scene_aoi_mapping", "scene_AOI_mapping.xlsx")]:
        c[field] = str(source / "00_teacher_inputs" / name)
    c["eeg"].update(trial_file=str(run / f"qc/{version}/eeg_trial_common_qc.csv"),
        onset_sensitivity_trial_file=str(run / f"qc/{version}/eeg_onset_sensitivity_trial_long.csv"),
        qc_config=str(run / f"qc/{version}/QC_policy.json"), order_stage_dir=str(run / "05_eeg_order"))
    for field, folder in [("stage1_dir", "01_eye_stage1"), ("stage2_dir", "02_eye_stage2"), ("stage3_plan_dir", "03_eye_stage3_plan")]:
        c["eye"][field] = str(source / folder)
    c["synchronized_timebin_file"] = str(source / "00_synchronized_source/aligned_synchronized_timebin_table.csv")
    c["clock_scene_qc_file"] = str(source / "00_synchronized_source/clock_alignment_scene_qc.csv")
    return c


def build_audit_qc(config, source, run):
    folder = run / "qc"
    baseline = pd.read_csv(source / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv", encoding="utf-8-sig")
    policy = read_json(source / "00_eeg_trial_qc/QC_policy.json")
    tables, thresholds = reconstruct_pair(baseline, policy)
    # Verify Original against the actual current baseline, including reason strings.
    a, b = tables["Original"].set_index(WINDOW_KEYS).sort_index(), baseline.set_index(WINDOW_KEYS).sort_index()
    for field in ("bad_eeg_quality", "eeg_subject_quality_exclusion", "CommonQCIncluded"):
        if not a[field].map(is_truthy).equals(b[field].map(is_truthy)):
            raise StageBlockedError(f"Original baseline membership mismatch: {field}")
    if not a.eeg_qc_reasons.fillna("").equals(b.eeg_qc_reasons.fillna("")):
        raise StageBlockedError("Original baseline QC reasons mismatch")
    for version, frame in tables.items():
        dest = folder / version
        write_csv(frame, dest / "eeg_onset_sensitivity_trial_long.csv")
        reference = frame.loc[frame.onset_trim_s.eq(10)].copy()
        reference["bad_eeg_quality"] |= ~reference.CommonQCIncluded
        write_csv(reference, dest / "eeg_trial_common_qc.csv")
        dump(dest / "QC_policy.json", qc_profiles(policy)[version])
        from .independent_eeg import reconstruct_qc
        ind, _ = reconstruct_qc(baseline.rename(columns={"participant_id": "Participant", "scene_id": "GlobalTrialOrder"}), qc_profiles(policy)[version])
        merged = frame.merge(ind.rename(columns={"Participant": "participant_id", "GlobalTrialOrder": "scene_id"})[
            WINDOW_KEYS + ["QCIncluded", "CommonQCIncluded"]], on=WINDOW_KEYS, validate="one_to_one", suffixes=("", "_independent"))
        if not merged.QCIncluded.eq(~merged.bad_eeg_quality).all() or not merged.CommonQCIncluded.eq(merged.CommonQCIncluded_independent).all():
            raise StageBlockedError("Independent QC reconstruction mismatch")
        write_csv(ind, dest / "independent_QC_reconstruction.csv")
    write_csv(thresholds, folder / "frozen_thresholds.csv")
    for name, table in impact_tables(tables).items():
        write_csv(table, folder / f"{name}.csv")
    dump(folder / "verification.json", {"Original_membership_and_reasons_match": True, "independent_QC_match": True,
         "remaining_thresholds_exactly_equal": True, "boundary_active": False, "NoHF_is_monotonic_superset": True})


def model_comparison(original, revised, pair, kind):
    from .eeg_denominator import compare_tables
    result = compare_tables(original, revised, pair, kind)
    x, y = result.estimate_from, result.estimate_to
    result["estimate_absolute_change"] = y - x
    result["estimate_percent_change"] = (100 * (y-x)/x.abs()).where(x.abs().ge(1e-8))
    return result


def audit_models(config, source, run, repo):
    from .eeg_audit import prepare_inputs
    from .eeg_denominator import run_models
    folder = run / "audit_models"
    folder.mkdir()
    c = bound_config(config, source, run, "NoHF")
    frames, excluded = prepare_inputs(c)
    qc = pd.read_csv(run / "qc/NoHF/eeg_onset_sensitivity_trial_long.csv", encoding="utf-8-sig")
    expected = set(qc.loc[qc.onset_trim_s.eq(0) & qc.CommonQCIncluded, TRIAL_KEYS].itertuples(index=False, name=None))
    if set(frames[0][KEYS].itertuples(index=False, name=None)) != expected:
        raise StageBlockedError("Invalid powers/design/registry would silently reduce the revised QC sample")
    write_csv(excluded, folder / "excluded_trials.csv")
    contract = {"seed": 20260906, "windows": TRIMS, "core_metrics": CORE, "factor_levels": FACTOR_LEVELS,
        "condition_terms": CONDITION, "temporal_terms": TEMPORAL, "previous_terms": PREVIOUS,
        "factor_method": "equal-weight marginal CR2/HTZ reimplementation; historical script unavailable",
        "previous_formula": "PreviousWWR + PreviousComplexity"}
    dump(folder / "analysis_contract.json", contract)
    # The complete Model1 set also supplies the two temporal families. Supplementary
    # fits are sealed here and can be reused after the formal decision gate.
    result = run_models(frames, c, repo, folder / "NoHF_1_45", folder / "analysis_contract.json")
    old_folder = source / "10_eeg_denominator_sensitivity/main_1_45"
    comparison = model_comparison(pd.read_csv(old_folder / "family_coefficients.csv"), result["family_coefficients"], "Original→NoHF", "coefficient")
    factors = model_comparison(pd.read_csv(old_folder / "family_factors.csv"), result["family_factors"], "Original→NoHF", "factor")
    write_csv(comparison, folder / "HF_QC_all_coefficient_comparison.csv")
    write_csv(factors, folder / "HF_QC_factor_comparison.csv")
    core = comparison.loc[comparison.family_id.eq("condition_core_relative")]
    write_csv(core, folder / "HF_QC_core_EEG_comparison.csv")
    write_csv(core.loc[core.outcome.eq("F_theta_relative") & core.term.eq("WWRWWR45:ComplexityC1")], folder / "HF_QC_key_effects_comparison.csv")
    write_csv(factors.loc[factors.family_id.eq("factor_core_relative") & factors.outcome.eq("O_theta_relative") & factors.term.eq("Complexity")], folder / "HF_QC_occipital_theta_complexity.csv")
    write_csv(comparison.loc[comparison.family_id.str.startswith("temporal_")], folder / "HF_QC_temporal_comparison.csv")


def methodological_gate(config, run, repo):
    proof = Path(config["hf_qc_revision"]["hf_formula_proof"])
    completion = read_json(proof / "audit_completion.json")
    scale = read_json(proof / "mean_waveform_scale_check.json")
    verify = read_json(run / "qc/verification.json")
    # Explicit numerical proof fields are configured, bound to their hashed files;
    # a prose assertion or statistical outcome cannot satisfy this gate.
    evidence = config["hf_qc_revision"]["cancellation_evidence"]
    def lookup(obj, path):
        for key in path.split("."):
            obj = obj[key]
        return obj
    mean_rms = float(lookup(scale, evidence["mean_waveform_rms_field"]))
    channel_rms = float(lookup(scale, evidence["mean_channel_rms_field"]))
    confirmed = lookup(completion, evidence["formula_verified_field"]) is True
    checks = [
        {"condition": 1, "pass": confirmed and channel_rms > 0 and 0 <= mean_rms/channel_rms < 1e-6,
         "evidence": {"mean_waveform_rms": mean_rms, "mean_channel_rms": channel_rms,
                      "proof_sha256": file_sha256(proof / "audit_completion.json"), "scale_sha256": file_sha256(proof / "mean_waveform_scale_check.json")}},
        {"condition": 2, "pass": verify["independent_QC_match"], "evidence": "Independent NoHF QC reconstruction exactly matches"},
        {"condition": 3, "pass": verify["remaining_thresholds_exactly_equal"], "evidence": "Same full-population thresholds; boundary recorded, not an active criterion"},
        {"condition": 4, "pass": verify["independent_QC_match"], "evidence": "Unchanged >30% rule applied to pre-participant criterion flags"},
        {"condition": 5, "pass": qc_profiles(read_json(run / "qc/Original/QC_policy.json"))["NoHF"] == read_json(run / "qc/NoHF/QC_policy.json"),
         "evidence": "Only robust_metrics HF entry removed; no replacement metric or retuning"}]
    decision = {"adopt_NoHF": all(r["pass"] for r in checks), "checks": checks,
                "selection_uses_significance": False, "code_sha": git_sha(repo)}
    dump(run / "methodological_gate.json", decision)
    return decision


def run(config, config_path, repo, outdir, *, resume=False, phase="all"):
    repo, run_root = Path(repo), Path(outdir)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise StageBlockedError("Real-data revision requires clean committed code")
    plan = preflight(config, repo)
    identity = {"config": config, "preflight": plan}
    fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if run_root.exists() and not resume:
        raise StageBlockedError("Use a new output directory or verified --resume")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / "run_manifest.json"
    if manifest.exists() and read_json(manifest)["fingerprint"] != fingerprint:
        raise StageBlockedError("Revision code/config/input/environment identity changed")
    dump(manifest, {"status": "running", "fingerprint": fingerprint, **plan})
    source = Path(plan["source_run"])
    stage(run_root, "qc", fingerprint, lambda: build_audit_qc(config, source, run_root), resume)
    if phase == "qc":
        return run_root
    stage(run_root, "audit_models", fingerprint, lambda: audit_models(config, source, run_root, repo), resume)
    decision = methodological_gate(config, run_root, repo)
    from .hf_qc_reports import audit_report
    audit_report(config, source, run_root, decision)
    if phase == "audit" or not decision["adopt_NoHF"]:
        verify_inputs(plan["inputs"])
        dump(manifest, {"status": "phase_A_complete", "fingerprint": fingerprint, **plan, "decision": decision})
        return run_root
    from .hf_qc_formal import formal_run
    formal_run(config, source, run_root, repo, fingerprint, resume)
    verify_inputs(plan["inputs"])
    dump(manifest, {"status": "analysis_complete_pending_delivery", "fingerprint": fingerprint, **plan, "decision": decision})
    return run_root
