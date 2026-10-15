"""Formal downstream reanalysis after the HF methodology gate passes."""
from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .eeg_audit import TRIMS, METRICS, KEYS, prepare_inputs
from .fresh import dump, stage, set_record, git_sha
from .hf_qc_revision import read_json, write_csv, bound_config, model_comparison
from .state import StageBlockedError, file_sha256


def verify_cache_implementation(manifest, epochs, method_hashes, current_runtime, source_runtime):
    """Check the historical key and the actual numerical environment separately.

    A lock JSON's byte hash is provenance, not the environment itself. Retain
    its registered historical hash when verifying the old identity, and require
    exact equality of the complete versions/paths/function hashes/runtime settings.
    """
    from .eeg_denominator import cache_identity
    if current_runtime["snapshot"] != source_runtime["snapshot"]:
        raise StageBlockedError("Prior and current MATLAB runtime snapshots differ")
    historical_methods = dict(method_hashes, native_runtime=source_runtime["lock_sha256"])
    if cache_identity(manifest["sources"], epochs, historical_methods) != manifest["cache_key"]:
        raise StageBlockedError("Prior PSD implementation/epoch identity no longer matches")
    return {"snapshots_exactly_equal": True, "original_lock_sha256": source_runtime["lock_sha256"],
            "current_lock_sha256": current_runtime["lock_sha256"], "original_cache_identity_verified": True}


def verify_integrals(cache, required_keys):
    """Independently integrate every reused/new spectrum that enters this run."""
    powers = pd.read_csv(cache / "paired_powers.csv")
    manifest = read_json(cache / "cache_manifest.json")
    rows = []
    for index, source in enumerate(manifest["sources"], 1):
        mat = loadmat(cache / f"spectra_{index:03d}.mat", squeeze_me=True, struct_as_record=False)
        if mat["metadata"].source.participant != source["participant"]:
            raise StageBlockedError("Spectrum source order mismatch")
        spectra = np.asarray(mat["spectra"], dtype=object)
        for column, roi in enumerate("FPO"):
            records = powers.loc[powers.Participant.eq(source["participant"]) & powers.roi.eq(roi)]
            if len(records) != spectra.shape[0]:
                raise StageBlockedError("Spectrum/power row mismatch")
            for j, record in enumerate(records.itertuples(index=False)):
                key = (record.Participant, record.GlobalTrialOrder, record.onset_trim_s)
                if key not in required_keys:
                    continue
                spectrum = spectra[j, column]
                if spectrum.first_sample != record.first_sample or spectrum.last_sample != record.last_sample:
                    raise StageBlockedError("Spectrum endpoints disagree")
                f, p = np.asarray(spectrum.f).ravel(), np.asarray(spectrum.pxx).ravel()
                for field, lo, hi in [("total_1_45", 1, 45), ("total_1_40", 1, 40), ("power_40_45", 40, 45),
                                      ("theta", 4, 7), ("alpha", 8, 12), ("beta", 13, 30)]:
                    mask = (f >= lo) & (f <= hi)
                    value = float(np.trapezoid(p[mask], f[mask]))
                    expected = getattr(record, field)
                    passed = bool(np.isclose(value, expected, rtol=1e-10, atol=1e-12))
                    rows.append({"Participant": record.Participant, "GlobalTrialOrder": record.GlobalTrialOrder,
                        "onset_trim_s": record.onset_trim_s, "roi": roi, "field": field,
                        "independent_value": value, "source_value": expected, "passed": passed,
                        "cache": str(cache)})
    result = pd.DataFrame(rows)
    if result.empty or not result.passed.all():
        raise StageBlockedError("Independent PSD integration mismatch")
    return result


def revised_denominator(config, source, run, repo):
    from .eeg_denominator import compute_psd, paired_inputs, run_models, verify_cache, PSD_PARAMETERS
    folder = run / "10_eeg_denominator_sensitivity"
    folder.mkdir()
    c = bound_config(config, source, run, "NoHF")
    frames, _ = prepare_inputs(c)
    cache = Path(config["hf_qc_revision"]["source_psd_cache"])
    old = read_json(cache / "cache_manifest.json")
    verify_cache(cache, old["cache_key"])
    if old["parameters"] != PSD_PARAMETERS:
        raise StageBlockedError("Prior PSD parameters changed")
    old_config = read_json(cache / "matlab_config.json")
    old_source = {s["participant"]: s for s in old["sources"]}
    # Epoch and method identities must agree; no output-only reuse assertion.
    epochs = pd.read_csv(cache / "frozen_epochs.csv", float_precision="round_trip")
    keys = [*KEYS, "onset_trim_s"]
    required = pd.concat([f[keys + ["view_start_s", "view_end_s"]] for f in frames.values()], ignore_index=True)
    intersection = required.merge(epochs, on=keys, suffixes=("", "_old"), how="inner", validate="one_to_one")
    for field in ["view_start_s", "view_end_s"]:
        if not np.allclose(intersection[field], intersection[field+"_old"], rtol=0, atol=1e-12):
            raise StageBlockedError("Cached epoch boundaries changed")
    from .eeg_denominator import METHOD_FILES, cache_identity
    from .native_runtime import validate_matlab
    code = {p: file_sha256(repo/p) for p in METHOD_FILES if p.endswith(".m")}
    native = validate_matlab(c["fresh"]["matlab"], c["fresh"]["eeglab_root"], repo)
    code["native_runtime"] = native["lock_sha256"]
    historical_runtime = read_json(source / "run_manifest.json")["runtime_lock"]["matlab"]
    runtime_reuse = verify_cache_implementation(old, epochs, code, native, historical_runtime)
    for person in intersection.Participant.unique():
        record = old_source[person]
        for kind in ["set", "fdt"]:
            if file_sha256(record[kind+"_path"]) != record[kind+"_sha256"]:
                raise StageBlockedError("Prior PSD waveform hash changed")
    old_powers = pd.read_csv(cache / "paired_powers.csv")
    selected = old_powers.merge(required[keys], on=keys, how="inner", validate="many_to_one")
    covered = set(intersection[keys].itertuples(index=False, name=None))
    all_keys = set(required[keys].itertuples(index=False, name=None))
    missing = all_keys - covered
    sources, gap_frames = [], {}
    for trim, frame in frames.items():
        mask = [(p, t, trim) in missing for p, t in frame[KEYS].itertuples(index=False, name=None)]
        gap_frames[trim] = frame.loc[mask].copy()
    component_caches = [str(cache)]
    checks = [verify_integrals(cache, all_keys)]
    if missing:
        for participant in sorted({k[0] for k in missing}):
            rec = set_record(Path(c["fresh"]["preprocessed_root"]) / f"{participant}.set", c["fresh"].get("fdt_aliases"))
            sources.append({"participant": participant, "set_path": rec["set_path"], "fdt_path": rec["fdt_path"]})
        c["denominator_sensitivity"] = {"matlab": c["fresh"]["matlab"], "eeglab_root": c["fresh"]["eeglab_root"], "psd_cache_dir": str(folder / "psd_cache")}
        powers, proof = compute_psd(c, gap_frames, sources, repo, folder)
        component_caches.append(proof["cache_dir"])
        checks.append(verify_integrals(Path(proof["cache_dir"]), missing))
        selected = pd.concat([selected, powers], ignore_index=True)
    write_csv(pd.concat(checks, ignore_index=True), folder / "independent_integral_verification.csv")
    versions, paired = paired_inputs(frames, selected)
    regression = []
    for trim in TRIMS:
        for metric in METRICS:
            for suffix in ["relative", "absolute"]:
                name = f"{metric}_{suffix}"
                passed = bool(np.allclose(frames[trim][name], versions["B"][trim][name], rtol=1e-10, atol=1e-12))
                regression.append({"onset_trim_s": trim, "metric": name, "passed": passed})
    write_csv(pd.DataFrame(regression), folder / "exporter_PSD_regression.csv")
    if not all(r["passed"] for r in regression):
        raise StageBlockedError("Revised PSD does not reproduce source-export powers")
    write_csv(paired, folder / "paired_psd_integrals.csv")
    # Phase A has already fitted the revised 1–45 data; preserve that identity.
    main = {}
    for p in (run / "audit_models/NoHF_1_45").glob("*.csv"):
        if p.stem != "outcomes":
            main[p.stem] = pd.read_csv(p)
    shutil.copytree(run / "audit_models/NoHF_1_45", folder / "main_1_45")
    sensitivity = run_models(versions["C"], c, repo, folder / "sensitivity_1_40", run / "audit_models/analysis_contract.json", reuse_absolute=main)
    comparisons = pd.concat([model_comparison(main[name], sensitivity[name], "Revised1–45→Revised1–40", kind)
        for name, kind in [("family_coefficients", "coefficient"), ("family_factors", "factor"), ("all_model_coefficients", "supplementary_coefficient")]], ignore_index=True)
    write_csv(comparisons, folder / "model_comparisons.csv")
    dump(folder / "PSD_provenance.json", {"status": "verified", "component_caches": component_caches,
        "required_trial_windows": len(all_keys), "reused_trial_windows": len(covered), "new_trial_windows": len(missing),
        "parameters": PSD_PARAMETERS, "code_sha": git_sha(repo), "original_cache_unchanged": True})
    dump(folder / "PSD_runtime_reuse_verification.json", runtime_reuse)


def formal_tables(run):
    out = run / "formal_tables"
    coeff = pd.read_csv(run / "audit_models/NoHF_1_45/family_coefficients.csv")
    factors = pd.read_csv(run / "audit_models/NoHF_1_45/family_factors.csv")
    for name, frame in [
        ("REVISED_core_EEG_144_coefficients", coeff.loc[coeff.family_id.eq("condition_core_relative")]),
        ("REVISED_factor_level_EEG", factors),
        ("REVISED_temporal_EEG", coeff.loc[coeff.family_id.str.startswith("temporal_")]),
        ("REVISED_previous_scene_EEG", coeff.loc[coeff.family_id.eq("previous_core_relative")]),
        ("REVISED_denominator_sensitivity", pd.read_csv(run / "10_eeg_denominator_sensitivity/model_comparisons.csv")),
        ("REVISED_QC_sample_flow", pd.read_csv(run / "qc/HF_QC_sample_comparison.csv").loc[lambda d: d.QCVersion.eq("NoHF")]),
        ("REVISED_QC_common_sample", pd.read_csv(run / "qc/NoHF/eeg_onset_sensitivity_trial_long.csv").loc[lambda d: d.onset_trim_s.eq(0) & d.CommonQCIncluded])]:
        write_csv(frame, out / f"{name}.csv")
    thresholds = pd.read_csv(run / "qc/frozen_thresholds.csv")
    table = thresholds.loc[thresholds.QCVersion.eq("NoHF")].pivot(index="onset_trim_s", columns="metric", values="threshold").reset_index()
    write_csv(table, out / "REVISED_Table_B3_EEG_QC_thresholds.csv")


def repeat_models(config, source, run, repo):
    from .eeg_denominator import run_models
    from .reproducibility import compare_tables
    c = bound_config(config, source, run, "NoHF")
    frames, _ = prepare_inputs(c)
    result = run_models(frames, c, repo, run / "model_repeat", run / "audit_models/analysis_contract.json")
    checks = []
    for name in ["coefficients", "factor_tests", "family_coefficients", "family_factors"]:
        left = pd.read_csv(run / f"audit_models/NoHF_1_45/{name}.csv")
        keys = ["onset_trim_s", "outcome", "model", "term"]
        if "family_id" in left:
            keys.append("family_id")
        comp = compare_tables(left, result[name], keys, model=True, exact_numeric=True)
        write_csv(comp, run / f"numerical_verification/{name}.csv")
        checks.append({"table": name, "passed": bool(comp.passed.all()), "rows": len(left)})
    dump(run / "numerical_verification/summary.json", {"checks": checks, "passed": all(r["passed"] for r in checks)})
    if not all(r["passed"] for r in checks):
        raise StageBlockedError("Revised independent-process model repeat failed")


def formal_run(config, source, run, repo, fingerprint, resume):
    if not read_json(run / "methodological_gate.json")["adopt_NoHF"]:
        raise StageBlockedError("No-HF formal adoption gate not passed")
    c = bound_config(config, source, run, "NoHF")
    # Preserve full original covariate and source tables, not just the old QC set.
    def formal_inputs():
        out = run / "00_teacher_inputs"
        shutil.copytree(source / "00_teacher_inputs", out)
        qc = run / "00_eeg_trial_qc"
        shutil.copytree(run / "qc/NoHF", qc)
        for name in ["preprocessing_parameters.csv", "preprocessing_source_inventory.csv"]:
            shutil.copyfile(source / "00_eeg_trial_qc" / name, qc / name)
        flow = pd.read_csv(run / "qc/HF_QC_sample_comparison.csv")
        flow = flow.loc[flow.QCVersion.eq("NoHF")].copy()
        flow["Stage"] = ["four_window_common_QC" if t == "common" else f"{t}s_individual_QC" for t in flow.onset_trim]
        write_csv(flow.rename(columns={"participants": "Participants", "trials": "Trials"}), qc / "sample_flow.csv")
        dump(out / "reuse_provenance.json", {"source": str(source), "scope": "unchanged trial design and covariates; no historical QC membership reused"})
        # The existing synchronized entrypoint reads this local trial table.
        # Copy its immutable eye-only metrics/eligibility; EEG membership is
        # rebuilt from the explicitly configured revised QC table downstream.
        eye_source = source / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx"
        eye_target = run / "02_eye_stage2" / eye_source.name
        eye_target.parent.mkdir(exist_ok=True)
        shutil.copyfile(eye_source, eye_target)
        dump(eye_target.parent / "verified_reuse.json", {"status": "verified_source_copy", "new_calculation": False,
            "source": str(eye_source), "source_sha256": file_sha256(eye_source), "copy_sha256": file_sha256(eye_target)})
    stage(run, "00_teacher_inputs", fingerprint, formal_inputs, resume)
    c["eeg"]["preprocessing_audit_file"] = str(run / "00_eeg_trial_qc/preprocessing_parameters.csv")
    dump(run / "resolved_config.json", c)
    stage(run, "10_eeg_denominator_sensitivity", fingerprint, lambda: revised_denominator(config, source, run, repo), resume)
    from .eeg import run_eeg_order, run_eeg_primary
    stage(run, "05_eeg_order", fingerprint, lambda: run_eeg_order(c, config_path=run / "resolved_config.json", outdir=run / "05_eeg_order", repo_root=repo), resume)
    # Production primary includes its 5000-draw bootstrap and scene-level joint
    # analyses. Other window jobs use exactly the same production R script.
    def remaining_windows():
        from .r_runner import invoke_r
        def window(trim):
            dest = run / f"07_eeg_window_robustness/trim_{trim}s"
            dest.mkdir(parents=True)
            shutil.copyfile(run / f"10_eeg_denominator_sensitivity/main_1_45/trim_{trim}s/input.csv", dest / "input.csv")
            e = c["eeg"]
            invoke_r(c["rscript"], repo / "analysis/r/eeg_primary_analysis.R", [str(dest/"input.csv"), str(dest),
                ",".join(f"{v}_relative" for v in e["core_metrics"]), ",".join(f"log10_{v}_absolute" for v in e["core_metrics"]),
                "5000", ",".join(f"{v}_relative" for v in e["secondary_metrics"]), ",".join(f"{v}_relative" for v in e["supplemental_metrics"])], required=True)
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(window, [0, 5, 15]))
    # Four independent R processes retain their own locked RNG state and exact
    # production scripts. Parallel scheduling changes no fitting parameter.
    with ThreadPoolExecutor(max_workers=2) as executor:
        primary = executor.submit(stage, run, "06_eeg_primary", fingerprint,
            lambda: run_eeg_primary(c, config_path=run / "resolved_config.json", outdir=run / "06_eeg_primary", repo_root=repo), resume)
        windows = executor.submit(stage, run, "07_eeg_window_robustness", fingerprint, remaining_windows, resume)
        primary.result()
        windows.result()
    from .complete import _build_sync_outputs
    stage(run, "08_synchronized_crossmodal", fingerprint, lambda: _build_sync_outputs(Path(config["outputs_root"]), run, c), resume)
    # The eye-only main analysis remains reusable; rerun only its joint-sample
    # sensitivity on the revised exact intersection in the dedicated helper.
    from .hf_qc_joint import eye_common_sensitivity, bootstrap_repeat
    stage(run, "eye_common_sensitivity", fingerprint, lambda: eye_common_sensitivity(c, source, run, repo), resume)
    if not (run / "model_repeat").exists():
        repeat_models(config, source, run, repo)
    stage(run, "bootstrap_repeat", fingerprint, lambda: bootstrap_repeat(c, run, repo), resume)
    formal_tables(run)
    from .hf_qc_reports import formal_report
    formal_report(run)
