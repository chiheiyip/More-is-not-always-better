"""Current-request incremental audit, with independent calculations sealed first."""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from statsmodels.stats.multitest import multipletests

from . import independent_eye as eye
from . import independent_eeg as eeg

CORE = eye.CORE


def git_sha(repo):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    base = Path(path).resolve().parent
    for k in ["outputs_root", "participant_information", "trial_order_mapping", "scene_aoi_mapping", "questionnaire_file",
              "aoi_approval", "rscript", "r_library", "matlab", "preprocessed_root", "eeg_qc_config",
              "historical_eeg_trial_file", "prior_verification", "sensitivity_run", "formal_eye_run", "delivery_root", "archive_root", "acquisition_source_map"]:
        p = Path(config[k]); config[k] = str(p if p.is_absolute() else (base / p).resolve())
    config["request_files"] = [str(Path(p).resolve()) for p in config.get("request_files", [])]
    return config


def preflight(config, repo):
    required = ["outputs_root", "participant_information", "trial_order_mapping", "scene_aoi_mapping", "questionnaire_file", "aoi_approval",
                "rscript", "r_library", "matlab", "preprocessed_root", "eeg_qc_config", "historical_eeg_trial_file", "prior_verification", "sensitivity_run", "formal_eye_run", "acquisition_source_map"]
    missing = [config[k] for k in required if not Path(config[k]).exists()]
    missing += [p for p in config.get("request_files", []) if not Path(p).is_file()]
    if missing:
        raise ValueError(f"Missing audit input/runtime: {missing}")
    a, b = Path(config["delivery_root"]).resolve(), Path(config["archive_root"]).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("Delivery and archive must be separate directories")
    return {"purpose": "incremental independent eye and EEG audit; ICA专项核查已取消",
            "eye": eye.preflight(config), "eeg_sets": len(list(Path(config["preprocessed_root"]).glob("*.set"))),
            "prior_verification": config["prior_verification"], "code_sha": git_sha(repo),
            "outdir": str(Path(config["outputs_root"]) / "teacher_request_runs" / config["run_id"])}


def r_env(config):
    env = os.environ.copy()
    for k in ["R_HOME", "LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_MONETARY", "LC_TIME", "LANG"]:
        env.pop(k, None)
    env["R_LIBS_USER"] = config["r_library"] + ";" + str(Path.home() / "AppData/Local/R/win-library/4.5")
    return env


def run_r(config, repo, script, job, output, label):
    work = Path(repo) / ".codex_tmp" / "incremental_jobs" / config["run_id"] / label
    work.mkdir(parents=True, exist_ok=True)
    # A new independent trial table is an input to the independent R model,
    # not a substitute for the separate source readers in the prior EEG audit.
    payload = dict(job)
    if "input" in payload:
        copy = work / "input.csv"; shutil.copyfile(payload["input"], copy); payload["input"] = str(copy)
    payload["outdir"] = str(work / "results")
    (work / "results").mkdir(exist_ok=True)
    job_path = work / "job.json"; eye.dump(job_path, payload)
    execution_sha = git_sha(repo)
    script_hash = eye.sha(Path(repo) / script)
    snapshot = work / "committed_analysis.R"
    shutil.copyfile(Path(repo) / script, snapshot)
    output.mkdir(parents=True, exist_ok=True)
    eye.dump(output / "job.json", {**job, "execution_job": str(job_path)})
    with (output / "execution.log").open("w", encoding="utf-8") as log:
        subprocess.run([config["rscript"], "--vanilla", str(snapshot), str(job_path)], env=r_env(config), stdout=log, stderr=subprocess.STDOUT, check=True)
    for p in (work / "results").iterdir():
        if p.is_file(): shutil.copyfile(p, output / p.name)
    eye.dump(output / "execution_provenance.json", {"code_sha": execution_sha, "script": script,
        "script_sha256": script_hash, "input_sha256": eye.sha(job["input"]) if "input" in job else None,
        "computed_at": datetime.now(timezone.utc).isoformat()})


def eye_models(config, repo, output):
    source = output / "eye" / "independent_eye_trials.csv"
    jobs = [("main", {"input": str(source), "mode": "main"}),
            ("boundary", {"input": str(output / "eye" / "independent_eye_sensitivity_trials.csv"), "mode": "boundary"})]
    people = pd.read_csv(source, encoding="utf-8-sig").query("QC60").Participant.unique()
    for n, names in enumerate(np.array_split(people, min(4, len(people)))):
        jobs.append((f"lopo_{n}", {"input": str(source), "mode": "lopo", "participants": list(names)}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(config.get("r_workers", 3))) as pool:
        tasks = {pool.submit(run_r, config, repo, "analysis/r/eye_independent_audit_analysis.R", job, output / "eye_models" / label, label): label for label, job in jobs}
        for task in concurrent.futures.as_completed(tasks):
            task.result(); print("Independent R finished:", tasks[task], flush=True)
    merge_eye_models(repo, output)


def merge_eye_models(repo, output):
    jobs = sorted(p for p in (output / "eye_models").iterdir() if p.is_dir())
    if {p.name for p in jobs} != {"main", "boundary", "lopo_0", "lopo_1", "lopo_2", "lopo_3"}:
        raise ValueError("Incomplete independent eye job roster")
    for folder in jobs:
        if not (folder / "execution_provenance.json").is_file(): raise ValueError("Independent eye job unfinished: "+folder.name)
    for filename in ["independent_coefficients.csv", "independent_factor_tests.csv", "independent_contrasts.csv", "08_model_specification_audit.csv", "independent_contrast_matrices.csv"]:
        frames = []
        for folder in jobs:
            p = folder / filename
            try: frames.append(pd.read_csv(p, encoding="utf-8-sig"))
            except pd.errors.EmptyDataError: pass
        eye.save(pd.concat(frames, ignore_index=True), output / "eye" / filename)
    eye.save(pd.read_csv(output / "eye" / "independent_factor_tests.csv"), output / "eye" / "09_multiplicity_audit.csv")
    eye.save(pd.read_csv(output / "eye" / "independent_contrasts.csv"), output / "eye" / "10_contrast_direction_audit.csv")
    seal(output / "eye", "independent_calculation_seal.json", repo)


def seal(directory, filename, repo):
    files = [{"path": str(p), "sha256": eye.sha(p)} for p in sorted(directory.glob("*.csv"))]
    eye.dump(directory / filename, {"code_sha": git_sha(repo), "completed_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "comparison_access": "formal eye results may be opened only after this independent calculation seal"})


def eeg_sources(config, repo, output):
    target = output / "eeg"; target.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path(config["preprocessed_root"]).glob("*.set")) + sorted(Path(config["preprocessed_root"]).glob("*.fdt"))
    before = [{"path": str(p), "sha256": eye.sha(p), "bytes": p.stat().st_size} for p in sources]
    eye.dump(target / "source_hashes_before.json", before)
    work = Path(repo) / ".codex_tmp" / "incremental_jobs" / config["run_id"] / "eeg_source"
    work.mkdir(parents=True, exist_ok=True)
    job = work / "job.json"; eye.dump(job, {"preprocessed_root": config["preprocessed_root"], "outdir": str(target)})
    matlab_dir = (Path(repo) / "matlab/eeg_bandpower_pipeline").as_posix().replace("'", "''")
    code = f"addpath('{matlab_dir}');run_eeg_independent_source_audit('{job.as_posix().replace(chr(39), chr(39)*2)}')"
    with (target / "matlab_execution.log").open("w", encoding="utf-8") as log:
        subprocess.run([config["matlab"], "-batch", code], stdout=log, stderr=subprocess.STDOUT, check=True)
    for r in before:
        if eye.sha(r["path"]) != r["sha256"]: raise ValueError("EEG source changed during read-only audit")
    eye.dump(target / "source_hashes_after.json", before)
    eeg.prepare(config, target)
    eeg.acquisition_audit(config, target)
    eye.dump(target / "execution_provenance.json", {"code_sha": git_sha(repo), "source_state": "preprocessed SET/FDT",
        "ICA_special_audit": "cancelled by user", "operator_information": config["ica_operator_statement"]})


def eeg_current_models(config, repo, output):
    target = output / "eeg"
    raw = pd.read_csv(target / "independent_raw_trigger_events.csv", encoding="utf-8-sig", dtype={"Marker":str})
    sets = pd.read_csv(target / "independent_eeg_set_events.csv", encoding="utf-8-sig", dtype={"Marker":str})
    sets = sets.loc[sets.Participant.isin(raw.Participant.unique())]
    eye.save(eeg.match_scene_epochs(raw, sets), target / "raw_to_SET_epoch_endpoint_matching.csv")
    inputs = eeg.current_model_inputs(config, target)
    files = [{"path":str(p),"sha256":eye.sha(p)} for p in sorted(inputs.rglob("input.csv"))]
    eye.dump(target / "current_QC_input_manifest.json", {"code_sha":git_sha(repo),"files":files,
        "interpretation":"Additional current-source QC sensitivity; historical A/B/C frozen sample unchanged"})
    run_r(config, repo, "analysis/r/eeg_independent_current_qc_models.R", {"inputs":str(inputs),
          "contract":str(Path(config["sensitivity_run"])/"analysis_contract.json")}, target / "current_QC_models", "eeg_current_QC")
    for record in files:
        if eye.sha(record["path"])!=record["sha256"]: raise ValueError("Current-QC input changed during model fitting")


def compare_values(audit, formal, fields, keys=eye.KEY, rtol=1e-10, atol=1e-12):
    eye.unique(audit, keys); eye.unique(formal, keys)
    a = audit[keys + fields].merge(formal[keys + fields], on=keys, how="outer", suffixes=("_audit", "_formal"), indicator=True)
    summaries = []
    for c in fields:
        x, y = a[c + "_audit"], a[c + "_formal"]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            match = np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
            difference = (x - y).abs(); maximum = float(difference.max()) if difference.notna().any() else 0
        else:
            match = x.fillna("<missing>").astype(str).eq(y.fillna("<missing>").astype(str)); maximum = None
        a[c + "_match"] = match & a._merge.eq("both")
        summaries.append({"Field": c, "ComparedRows": len(a), "MismatchedRows": int((~a[c + "_match"]).sum()), "MaxAbsDifference": maximum})
    return a, pd.DataFrame(summaries)


def verify_eye_bh(target):
    factors = pd.read_csv(target / "independent_factor_tests.csv", encoding="utf-8-sig")
    rows = []
    for (variant, effect, family, omitted), g in factors.groupby(["variant", "effect", "family", "omitted_participant"], dropna=False):
        if family not in ["A", "B"]: continue
        expected = CORE[:3] if family == "A" else CORE[3:]
        complete = set(g.outcome) == set(expected) and len(g) == 3 and np.isfinite(g.raw_p).all() and g.converged.all()
        q = multipletests(g.raw_p, method="fdr_bh")[1] if complete else np.full(len(g), np.nan)
        if not np.allclose(q, g.q, atol=1e-12, rtol=1e-10, equal_nan=True): raise ValueError("Independent Python/R eye BH differs")
        rows.append({"variant": variant, "effect": effect, "family": family, "omitted_participant": omitted,
                     "expected_size": 3, "observed_size": len(g), "complete": bool(complete), "PythonRMatch": True})
    eye.save(pd.DataFrame(rows), target / "Python_R_BH_checks.csv")


def compare_eye(config, output):
    target = output / "eye"
    if not (target / "independent_calculation_seal.json").is_file(): raise ValueError("Independent eye calculations not sealed")
    saved = json.loads((target / "independent_calculation_seal.json").read_text(encoding="utf-8"))
    for r in saved["files"]:
        if eye.sha(r["path"]) != r["sha256"]: raise ValueError("Independent eye outputs changed before comparison")
    independent = pd.read_csv(target / "independent_eye_trials.csv", encoding="utf-8-sig")
    formal = pd.read_csv(Path(config["formal_eye_run"]) / "02_eye_stage2/eye_stage2_model_input.csv", encoding="utf-8-sig")
    rename = {"ExperienceGroup": "ExerciseFrequency"}
    formal = formal.rename(columns={k: v for k, v in rename.items() if v not in formal})
    selected = independent.loc[independent.QC60].copy()
    fields = [c for c in [*CORE, "Gender", "ExerciseFrequency", "WWR", "Complexity", "Block", "PositionWithinBlock", "PositionWithinBlockCentered", "OrderGroup", "PreviousWWR", "PreviousComplexity", "ValidTrackingRatio", "ValidTrackingSeconds", "TableAreaShare", "WindowAreaShare", "EquipmentAreaShare"] if c in formal]
    # Explicitly decode categorical display labels; never infer Q1.4 from these.
    for c in ["WWR", "Complexity", "PreviousWWR", "PreviousComplexity"]:
        if c in formal:
            formal[c] = pd.to_numeric(formal[c].astype(str).str.replace("WWR", "", regex=False).str.replace("C", "", regex=False), errors="coerce")
    values, summary = compare_values(selected, formal, fields)
    eye.save(values, target / "trial_input_comparison.csv"); eye.save(summary, target / "trial_input_comparison_summary.csv")
    q = independent.drop_duplicates("Participant")[["Participant", "Q1.4Original", "ExerciseFrequency"]]
    used = formal.groupby("Participant").ExerciseFrequency.agg(lambda v: ";".join(sorted(set(v.dropna().astype(str))))).reset_index().rename(columns={"ExerciseFrequency": "FormalExerciseFrequency"})
    q = q.merge(used, on="Participant", how="left")
    q["Match"] = q.ExerciseFrequency.eq(q.FormalExerciseFrequency)
    q["Notes"] = np.where(q.FormalExerciseFrequency.isna(), "No retained formal eye trial; candidate still audited", "Q1.4-derived group independently compared")
    q["Q1.5ActuallyUsedDetected"] = np.where(q.FormalExerciseFrequency.isna(), "not_in_model", np.where(q.Match, "no_group_discrepancy; provenance_check_required", "investigate"))
    eye.save(q, target / "00_exercise_frequency_grouping_audit.csv")
    verify_eye_bh(target)
    return {"independent_participants": int(selected.Participant.nunique()), "independent_trials": len(selected),
            "formal_participants": int(formal.Participant.nunique()), "formal_trials": len(formal),
            "input_fields": fields, "mismatched_fields": summary.loc[summary.MismatchedRows > 0, "Field"].tolist(),
            "group_mismatches_in_formal_sample": int((q.FormalExerciseFrequency.notna() & ~q.Match).sum())}


def compare_eeg(config, repo, output):
    target = output / "eeg"
    summary, hashes = eeg.verify_prior(config, repo)
    eye.save(pd.DataFrame(hashes), target / "prior_result_reuse_hash_checks.csv")
    d = pd.read_csv(target / "independent_eeg_trial_QC.csv", encoding="utf-8-sig")
    comparison = eeg.compare_qc(d, config["historical_eeg_trial_file"])
    eye.save(comparison, target / "current_source_vs_historical_QC.csv")
    # Historical input-state QC can also be independently checked from its
    # archived waveform-derived features; explicitly different from raw rebuild.
    old = pd.read_csv(config["historical_eeg_trial_file"], encoding="utf-8-sig").rename(columns={"participant_id": "Participant", "scene_id": "GlobalTrialOrder"})
    cols = eye.KEY + ["onset_trim_s", "analysis_dur_s", "segment_valid_duration", "nan_fraction", "flat_fraction", *eeg.METRICS]
    historical_qc, thresholds = eeg.reconstruct_qc(old[cols], json.loads(Path(config["eeg_qc_config"]).read_text(encoding="utf-8")))
    identity = historical_qc[eye.KEY + ["onset_trim_s", "QCIncluded", "CommonQCIncluded"]].merge(old[eye.KEY + ["onset_trim_s", "bad_eeg_quality"]], on=eye.KEY + ["onset_trim_s"], validate="one_to_one")
    identity["HistoricalQCMatch"] = identity.QCIncluded.eq(~identity.bad_eeg_quality.astype(str).str.lower().isin(["true", "1"]))
    eye.save(identity, target / "historical_feature_QC_independent_check.csv")
    eye.save(thresholds, target / "historical_feature_QC_thresholds.csv")
    common = historical_qc.loc[historical_qc.CommonQCIncluded & historical_qc.onset_trim_s.eq(0), eye.KEY]
    frozen = pd.read_csv(Path(config["sensitivity_run"]) / "B/trim_0s/input.csv", encoding="utf-8-sig")
    common_check = common.merge(frozen[eye.KEY], on=eye.KEY, how="outer", indicator=True)
    eye.save(common_check, target / "historical_common_QC_identity.csv")
    required = d.loc[d.onset_trim_s.eq(0)].merge(frozen[eye.KEY], on=eye.KEY)
    power_rows = []
    for trim in [0, 5, 10, 15]:
        source = d.loc[d.onset_trim_s.eq(trim)].copy()
        b = pd.read_csv(Path(config["sensitivity_run"]) / f"B/trim_{trim}s/input.csv", encoding="utf-8-sig")
        c = pd.read_csv(Path(config["sensitivity_run"]) / f"C/trim_{trim}s/input.csv", encoding="utf-8-sig")
        merged = source.merge(b, on=eye.KEY, suffixes=("_audit", "_B"), validate="one_to_one")
        for roi in ["F", "P", "O"]:
            for band in ["theta", "alpha", "beta"]:
                col = f"{roi}_{band}_relative"
                c_values = c.set_index(eye.KEY).loc[pd.MultiIndex.from_frame(merged[eye.KEY]), col].to_numpy()
                independent_c = merged[col + "_1_40"].to_numpy()
                power_rows.append({"onset_trim_s": trim, "ROI": roi, "band": band, "trials": len(merged),
                    "B_max_abs_error": float((merged[col + "_audit"] - merged[col + "_B"]).abs().max()),
                    "B_integral_match": bool(np.allclose(merged[col + "_audit"], merged[col + "_B"], rtol=1e-10, atol=1e-12)),
                    "C_max_abs_error": float(np.max(np.abs(independent_c - c_values))),
                    "C_integral_match": bool(np.allclose(independent_c, c_values, rtol=1e-10, atol=1e-12))})
        if trim == 0:
            spot = merged.sample(n=min(12, len(merged)), random_state=20261006)
            spot_cols = eye.KEY + ["onset_trim_s_audit", "AnalysisStartSample", "EndSample"] + [col for col in merged if ("_absolute_audit" in col or "_relative_audit" in col or "relative_1_40" in col or "total_1_" in col or "power_40_45" in col)]
            eye.save(spot[spot_cols], target / "waveform_power_spot_check.csv")
    eye.save(pd.DataFrame(power_rows), target / "waveform_to_B_C_power_checks.csv")
    run_r(config, repo, "analysis/r/eeg_independent_model_details.R", {"source_run": config["sensitivity_run"],
          "contract": str(Path(config["sensitivity_run"]) / "analysis_contract.json")}, target, "eeg_details")
    prior = pd.read_csv(Path(config["prior_verification"]) / "r_refit_coefficients.csv", encoding="utf-8-sig")
    details = pd.read_csv(target / "independent_eeg_core_details.csv", encoding="utf-8-sig")
    paired = details.merge(prior, on=["version", "onset_trim_s", "outcome", "model", "term"], suffixes=("_new", "_prior"), validate="one_to_one")
    paired["EstimateMatch"] = (paired.estimate_new - paired.estimate_prior).abs() <= 1e-8
    paired["PMatch"] = (paired.raw_p - paired["p.value"]).abs() <= 1e-6
    eye.save(paired, target / "core_details_regression.csv")
    counts = [{"onset_trim_s": t, "participants": int(g.loc[g.QCIncluded, "Participant"].nunique()), "trials": int(g.QCIncluded.sum())} for t, g in d.groupby("onset_trim_s")]
    return {"prior_models_reused": summary["R_refit_models"], "prior_spectra_verified": summary["spectra_checked"],
            "prior_source_files_verified": len(hashes), "current_variant_counts": counts,
            "current_common_participants": int(d.loc[d.CommonQCIncluded, "Participant"].nunique()),
            "current_common_trials": int(d.loc[d.CommonQCIncluded & d.onset_trim_s.eq(0)].shape[0]),
            "current_vs_historical_QC_differences": int((~comparison.InclusionMatch).sum()),
            "historical_QC_exact_match": bool(identity.HistoricalQCMatch.all()),
            "historical_common_identity_match": bool(common_check._merge.eq("both").all()),
            "waveform_power_match": bool(pd.DataFrame(power_rows)[["B_integral_match", "C_integral_match"]].all().all()),
            "core_details_regression_passed": bool(paired.EstimateMatch.all() and paired.PMatch.all()),
            "source_state": "现有预处理 SET/FDT；原始采集文件的可访问性另行登记"}


def run(config, config_path, repo, output, phase):
    repo, output = Path(repo), Path(output)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("Commit and synchronize code before a real audit")
    output.mkdir(parents=True, exist_ok=True)
    eye.dump(output / "request_config.json", config)
    eye.dump(output / "run_provenance.json", {"code_sha": git_sha(repo), "config_sha256": eye.sha(config_path),
         "python": sys.version, "ICA_special_audit": "cancelled by user", "request_files": [{"path": p, "sha256": eye.sha(p)} for p in config.get("request_files", [])]})
    eye.dump(output / ("phase_"+phase+"_provenance.json"), {"code_sha":git_sha(repo),"config_sha256":eye.sha(config_path),"started_at":datetime.now(timezone.utc).isoformat()})
    if phase in ["all", "eye-process"]:
        eye.process(config, output / "eye")
    if phase in ["all", "eye-models"]:
        eye_models(config, repo, output)
    if phase == "eye-boundary-models":
        run_r(config,repo,"analysis/r/eye_independent_audit_analysis.R",{"input":str(output/"eye/independent_eye_sensitivity_trials.csv"),"mode":"boundary"},output/"eye_models/boundary","boundary")
    if phase == "eye-seal-models":
        merge_eye_models(repo,output)
    if phase in ["all", "eeg-source"]:
        eeg_sources(config, repo, output)
    if phase in ["all", "eeg-current-models"]:
        eeg_current_models(config, repo, output)
    if phase == "eeg-compare":
        eye.dump(output / "eeg_comparison_summary.json", compare_eeg(config, repo, output))
    if phase == "eeg-evidence":
        from .incremental_evidence import eeg_evidence
        eeg_evidence(config,output)
    if phase == "eye-compare":
        from .incremental_evidence import eye_arithmetic
        eye.dump(output / "eye_comparison_summary.json", {**compare_eye(config,output),**eye_arithmetic(output)})
    if phase in ["all", "compare"]:
        eye_result = compare_eye(config, output)
        eeg_result = compare_eeg(config, repo, output)
        eye.dump(output / "audit_summary.json", {"eye": eye_result, "eeg": eeg_result, "code_sha": git_sha(repo)})
    return output
