"""Fresh eye/EEG execution; original measurement and statistical methods retained.

This coordinator deliberately has no historical-result input. Completed stages
can resume only inside this run, with exact code/input/output fingerprints.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

from .state import StageBlockedError, file_sha256

TRIMS = (0, 5, 10, 15)
KEY = ["Participant", "GlobalTrialOrder"]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def git_sha(repo):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def unique(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise StageBlockedError(f"Missing/duplicate keys: {keys}")


def set_record(path, fdt_aliases=None):
    """Read the FDT named by SET, never infer its basename silently."""
    path = Path(path).resolve()
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as source:
            def text(dataset):
                a=np.asarray(dataset[()])
                if a.dtype==h5py.ref_dtype:
                    return text(source[a.ravel()[0]])
                if dataset.attrs.get("MATLAB_class",b"")==b"char" or a.dtype==np.uint16:
                    return "".join(chr(int(v)) for v in a.ravel())
                return str(a.item())
            fields={k:float(source[k][()].item()) for k in ["nbchan","pnts","trials","srate"]}
            data=text(source["data"])
            labels=[text(source[v]) for v in source["chanlocs/labels"][()].ravel()]
            types=[text(source[v]).removesuffix(".0") for v in source["event/type"][()].ravel()]
            history=text(source["history"])
    else:
        fields = loadmat(path, squeeze_me=True, struct_as_record=False,
                         variable_names=["data", "nbchan", "pnts", "trials", "srate", "event", "chanlocs", "history"])
        data=fields.get("data")
        labels=[str(c.labels) for c in np.atleast_1d(fields["chanlocs"])]
        types=[str(e.type).removesuffix(".0") for e in np.atleast_1d(fields["event"])]
        history=str(fields.get("history",""))
    if not isinstance(data, str) or Path(data).name != data:
        raise StageBlockedError(f"Expected a local external FDT reference: {path}")
    declared_fdt=data
    if data != path.with_suffix(".fdt").name:
        alias=(fdt_aliases or {}).get(path.name)
        if not alias or alias.get("declared")!=data or alias.get("actual")!=path.with_suffix(".fdt").name:
            raise StageBlockedError(f"Explicit audited SET/FDT filename alias required: {path}: {data}")
        data=alias["actual"]
    fdt = path.parent / data
    expected = int(fields["nbchan"]) * int(fields["pnts"]) * int(fields["trials"]) * 4
    if not fdt.is_file() or fdt.stat().st_size != expected:
        raise StageBlockedError(f"SET/FDT dimensions or referenced file mismatch: {path}")
    if len(labels) != int(fields["nbchan"]) or not {"F3", "F4", "P3", "PZ", "P4", "O1", "OZ", "O2"}.issubset({v.upper() for v in labels}):
        raise StageBlockedError(f"EEG ROI channels unavailable: {path}")
    starts=[i for i,v in enumerate(types) if v=="7"]
    if len(starts)!=12:
        raise StageBlockedError(f"Expected twelve scene start/end events: {path}")
    # Additional 8 markers delimit block/questionnaire transitions; only 7→8
    # adjacent events are scene epochs in the original exporter.
    if any(i+1>=len(types) or types[i+1]!="8" for i in starts):
        raise StageBlockedError(f"Unpaired scene events: {path}")
    return {"Participant": path.stem, "set_path": str(path), "fdt_path": str(fdt),
            "nbchan": int(fields["nbchan"]), "pnts": int(fields["pnts"]),
            "srate": float(fields["srate"]), "channels": labels,
            "history": history,"declared_fdt":declared_fdt,"fdt_alias_registered":declared_fdt!=data}


def input_inventory(config, repo):
    fresh = config["fresh"]
    sets = [set_record(p,fresh.get("fdt_aliases")) for p in sorted(Path(fresh["preprocessed_root"]).glob("*.set"))]
    if not sets:
        raise StageBlockedError("No preprocessed SET files")
    participants = pd.read_excel(config["participant_information"], sheet_name=0)
    mapping = pd.read_excel(config["trial_order_mapping"], sheet_name=0)
    scenes = pd.read_excel(config["scene_aoi_mapping"], sheet_name=0)
    unique(participants, ["Participant"]); unique(mapping, KEY)
    if not set(r["Participant"] for r in sets).issubset(set(participants.Participant)):
        raise StageBlockedError("EEG source not present in participant/design mapping")
    paths = {Path(config[k]).resolve() for k in ["participant_information", "trial_order_mapping", "scene_aoi_mapping", "questionnaire_file"]}
    if config.get("eye", {}).get("aoi_pixel_lock"):
        aoi_lock = Path(config["eye"]["aoi_pixel_lock"]).resolve()
        paths.add(aoi_lock)
        paths.add(aoi_lock.parent / json.loads(aoi_lock.read_text(encoding="utf-8"))["mask_archive"])
    for record in sets:
        paths.update(Path(record[k]) for k in ["set_path", "fdt_path"])
    for table, columns in [(mapping, ["CSVFile"]), (scenes, ["AOIFile", "BaseImageFile", "ValidSceneFile"])]:
        for col in columns:
            paths.update(Path(str(v)).resolve() for v in table[col].dropna() if str(v).strip())
    paths.update([Path(repo) / "configs/eeg_qc.json", Path(repo) / "configs/eeg_analysis.json"])
    # EASY/INFO are source measurements, not a previously computed clock table.
    acquisition = Path(fresh["acquisition_root"])
    paths.update(acquisition.rglob("*.easy")); paths.update(acquisition.rglob("*.info"))
    inventory = []
    for p in sorted(paths, key=str):
        if not p.is_file():
            raise StageBlockedError(f"Source input missing: {p}")
        inventory.append({"path": str(p), "bytes": p.stat().st_size, "sha256": file_sha256(p)})
    by_path = {r["path"]: r["sha256"] for r in inventory}
    for name, expected in fresh.get("locked_source_hashes", {}).items():
        p = str((Path(fresh["preprocessed_root"]) / name).resolve())
        if by_path.get(p) != expected.lower():
            raise StageBlockedError(f"Locked corrected waveform changed: {name}")
    return inventory, sets


def preflight(config, repo):
    from .runtime_lock import validate_analysis
    from .aoi_lock import verify_aoi
    environment = validate_analysis(config, repo, require_matlab=True)
    aoi = verify_aoi(config, required=True)
    fresh = config.get("fresh", {})
    for key in ["preprocessed_root", "acquisition_root", "matlab", "eeglab_root"]:
        if not fresh.get(key) or not Path(fresh[key]).exists():
            raise StageBlockedError(f"Missing fresh runtime/source: {key}")
    if not Path(config["rscript"]).is_file():
        raise StageBlockedError("R launcher missing")
    if config["eeg"].get("onset_trim_strategy") != "parallel" or tuple(config["eeg"].get("onset_trim_variants_s", [])) != TRIMS:
        raise StageBlockedError("Fresh mode retains four parallel windows")
    if config["eye"]["primary_tracking_threshold"] != 0.6:
        raise StageBlockedError("Fresh mode retains the eye 60% main threshold")
    inventory, sets = input_inventory(config, repo)
    return {"scope": "eye-eeg", "source_files": len(inventory), "eeg_sources": len(sets),
            "code_sha": git_sha(repo), "questionnaire_models": "out_of_scope",
            "historical_derived_results": "prohibited", "input_inventory": inventory,
            "set_inventory": sets, "runtime_lock": environment, "aoi_pixel_lock": aoi}


def verify_inputs(records):
    for r in records:
        p = Path(r["path"])
        if not p.is_file() or file_sha256(p) != r["sha256"]:
            raise StageBlockedError(f"Source changed during run: {p}")


def stage(run, name, fingerprint, action, resume):
    """A stage seal covers every artifact except the seal itself."""
    seal = run / "stage_seals" / f"{name}.json"
    if resume and seal.is_file():
        previous = json.loads(seal.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise StageBlockedError(f"Resume source/code/config changed: {name}")
        for r in previous["files"]:
            p = run / r["path"]
            if not p.is_file() or file_sha256(p) != r["sha256"]:
                raise StageBlockedError(f"Resume artifact changed: {name}: {p}")
        print(f"Verified same-run resume: {name}", flush=True)
        return
    folder = run / name
    if folder.exists() and any(folder.iterdir()):
        raise StageBlockedError(f"Unsealed stage exists; retain it and use a new run: {name}")
    print(f"START {name}", flush=True)
    action()
    records = [{"path": p.relative_to(run).as_posix(), "sha256": file_sha256(p)}
               for p in sorted(folder.rglob("*")) if p.is_file()]
    if not records:
        raise StageBlockedError(f"No stage outputs: {name}")
    dump(seal, {"fingerprint": fingerprint, "files": records, "completed_at": datetime.now(timezone.utc).isoformat()})
    print(f"COMPLETE {name}: {len(records)} files", flush=True)


def command(args, repo, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("w", encoding="utf-8") as stream:
        subprocess.run([str(v) for v in args], cwd=repo, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build_design(config, run, sets):
    from .independent_eye import read_questionnaire
    out = run / "00_teacher_inputs"; out.mkdir(parents=True, exist_ok=True)
    info = pd.read_excel(config["participant_information"], sheet_name=0)
    original = read_questionnaire(config["questionnaire_file"])
    q = original[["Participant", "ExerciseFrequency", "Q1.4Original", "Q1.5Original"]]
    info = info.merge(q, on="Participant", how="left", validate="one_to_one")
    if info.ExerciseFrequency.isna().any() or not info.ExperienceGroup.eq(info.ExerciseFrequency).all():
        raise StageBlockedError("Original Q1.4 disagrees with frozen participant grouping")
    info["ExperienceGroup"] = info.ExerciseFrequency
    info["IncludeEEGValid"] = info.Participant.isin([s["Participant"] for s in sets])
    info.to_excel(out / "participant_information.xlsx", index=False)
    original.to_csv(out / "Q1_4_group_check.csv", index=False, encoding="utf-8-sig")
    mapping = pd.read_excel(config["trial_order_mapping"], sheet_name=0).sort_values(KEY)
    for _, g in mapping.groupby("Participant"):
        if len(g) != 12 or set(g.GlobalTrialOrder) != set(range(1, 13)):
            raise StageBlockedError("Trial design must contain twelve unique trials per candidate")
        if not g.Block.eq(np.ceil(g.GlobalTrialOrder / 6)).all() or not g.PositionWithinBlock.eq((g.GlobalTrialOrder - 1) % 6 + 1).all():
            raise StageBlockedError("Block/position disagrees with original trial order")
    mapping["PreviousWWR"] = mapping.groupby(["Participant", "Block"]).WWR.shift()
    mapping["PreviousComplexity"] = mapping.groupby(["Participant", "Block"]).Complexity.shift()
    mapping.to_excel(out / "trial_order_mapping.xlsx", index=False)
    pd.read_excel(config["scene_aoi_mapping"]).to_excel(out / "scene_AOI_mapping.xlsx", index=False)
    standard = info.rename(columns={"Participant": "participant_id"})
    standard["exclude"] = False
    standard.to_csv(out / "participants_standardized.csv", index=False, encoding="utf-8-sig")
    scene = mapping.rename(columns={"Participant": "participant_id", "GlobalTrialOrder": "scene_id", "Block": "block", "PositionWithinBlock": "position", "CSVFile": "eye_csv_path"})
    # SceneID is a stimulus identity; scene_id is the participant's trial key.
    aoi = pd.read_excel(config["scene_aoi_mapping"])
    unique(aoi, ["SceneID", "OrderGroup", "Block"])
    scene = scene.merge(aoi[["SceneID", "OrderGroup", "Block", "AOIFile", "BaseImageFile"]].rename(columns={"Block": "block"}),
                        on=["SceneID", "OrderGroup", "block"], validate="many_to_one")
    scene["aoi_json_path"] = scene.AOIFile; scene["eye_filename"] = scene.eye_csv_path.map(lambda v: Path(v).name)
    import re
    def recording_id(value):
        match=re.search(r"_(\d{12}_\d+)\.csv$",Path(value).name)
        if not match:raise StageBlockedError(f"Eye acquisition date unavailable in source filename: {value}")
        return match.group(1)
    scene["eye_record_id"]=scene.eye_csv_path.map(recording_id)
    scene=scene.merge(standard[["participant_id","Gender","ExperienceGroup"]],on="participant_id",validate="many_to_one")
    scene["WWR_numeric"]=scene.WWR
    scene["Cond"]="C"+scene.Complexity.astype(int).astype(str)
    scene["condition_id"]=scene.Cond+"_W"+scene.WWR.astype(int).astype(str)
    scene.to_csv(out / "scene_manifest_standardized.csv", index=False, encoding="utf-8-sig")


def resolved_config(config, run):
    c = copy.deepcopy(config)
    inputs = run / "00_teacher_inputs"
    c["participant_information"] = str(inputs / "participant_information.xlsx")
    c["trial_order_mapping"] = str(inputs / "trial_order_mapping.xlsx")
    c["scene_aoi_mapping"] = str(inputs / "scene_AOI_mapping.xlsx")
    c["eeg"]["trial_file"] = str(run / "00_eeg_trial_qc/eeg_trial_common_qc.csv")
    c["eeg"]["onset_sensitivity_trial_file"] = str(run / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv")
    c["eeg"]["preprocessing_audit_file"] = str(run / "00_eeg_trial_qc/preprocessing_parameters.csv")
    c["eeg"]["onset_analysis_dir"] = ""
    c["eeg"]["order_stage_dir"] = str(run / "05_eeg_order")
    for key, folder in [("stage1_dir", "01_eye_stage1"), ("stage2_dir", "02_eye_stage2"), ("stage3_plan_dir", "03_eye_stage3_plan")]:
        c["eye"][key] = str(run / folder)
    c["eye"]["provisional_user_authorization"] = True
    c.setdefault("stage3", {})["s3_trial_file"] = ""
    c["synchronized_timebin_file"] = str(run / "00_synchronized_source/aligned_synchronized_timebin_table.csv")
    c["clock_scene_qc_file"] = str(run / "00_synchronized_source/clock_alignment_scene_qc.csv")
    c.pop("denominator_sensitivity", None)
    return c


def build_qc(config, run, repo, sets):
    from paper_analysis.eeg.pipeline import run_eeg_pipeline
    inputs = run / "00_teacher_inputs"
    exported = run / "00_eeg_raw_export/summary/all_subjects_scene_level.csv"
    out = run / "00_eeg_trial_qc"
    result = run_eeg_pipeline(inputs / "participants_standardized.csv", inputs / "scene_manifest_standardized.csv",
                              exported, out, repo / "configs/eeg_qc.json", repo / "configs/eeg_analysis.json",
                              require_onset_metadata=True)
    data = pd.read_csv(result["eeg_onset_sensitivity_trial_long"], encoding="utf-8-sig")
    unique(data, ["participant_id", "scene_id", "onset_trim_s"])
    if len(data) != len(sets) * 12 * 4:
        raise StageBlockedError("Exporter lost/duplicated source trials")
    from .independent_eeg import reconstruct_qc
    independent, independent_thresholds=reconstruct_qc(data.rename(columns={"participant_id":"Participant","scene_id":"GlobalTrialOrder"}),
        json.loads((repo/"configs/eeg_qc.json").read_text(encoding="utf-8")))
    independent.to_csv(out/"independent_QC_reconstruction.csv",index=False,encoding="utf-8-sig")
    independent_thresholds.to_csv(out/"independent_QC_thresholds.csv",index=False,encoding="utf-8-sig")
    check=data.merge(independent[[*KEY,"onset_trim_s","QCIncluded","eeg_subject_quality_exclusion"]].rename(
        columns={"Participant":"participant_id","GlobalTrialOrder":"scene_id"}),
        on=["participant_id","scene_id","onset_trim_s"],validate="one_to_one",suffixes=("","_independent"))
    if not check.QCIncluded.eq(~check.bad_eeg_quality & ~check.eeg_subject_quality_exclusion).all() or not check.eeg_subject_quality_exclusion.eq(check.eeg_subject_quality_exclusion_independent).all():
        raise StageBlockedError("Independent QC membership reconstruction differs")
    passed = ~data.bad_eeg_quality & ~data.eeg_subject_quality_exclusion
    common = data.assign(Pass=passed).groupby(["participant_id", "scene_id"]).agg(Windows=("onset_trim_s", "nunique"), Pass=("Pass", "all"))
    keep = set(common.loc[common.Windows.eq(4) & common.Pass].index)
    data["CommonQCIncluded"] = [tuple(v) in keep for v in data[["participant_id", "scene_id"]].to_numpy()]
    data.to_csv(out / "eeg_onset_sensitivity_trial_long.csv", index=False, encoding="utf-8-sig")
    reference = data.loc[data.onset_trim_s.eq(10)].copy()
    reference["bad_eeg_quality"] |= ~reference.CommonQCIncluded
    reference.loc[~reference.CommonQCIncluded, "eeg_qc_reasons"] = reference.loc[~reference.CommonQCIncluded, "eeg_qc_reasons"].fillna("") + ";not_in_four_window_common_QC"
    reference.to_csv(out / "eeg_trial_common_qc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sets).to_csv(out / "preprocessing_source_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Parameter": k, "Value": v} for k, v in config["eeg"].get("preprocessing_parameters", {}).items()]).to_csv(
        out / "preprocessing_parameters.csv", index=False, encoding="utf-8-sig")
    flow = []
    for trim, g in data.groupby("onset_trim_s"):
        p = g.loc[~g.bad_eeg_quality & ~g.eeg_subject_quality_exclusion]
        flow.append({"Stage": f"{int(trim)}s_individual_QC", "Participants": p.participant_id.nunique(), "Trials": len(p)})
    selected = data.loc[data.onset_trim_s.eq(0) & data.CommonQCIncluded]
    flow.append({"Stage": "four_window_common_QC", "Participants": selected.participant_id.nunique(), "Trials": len(selected)})
    pd.DataFrame(flow).to_csv(out / "sample_flow.csv", index=False, encoding="utf-8-sig")
    dump(out / "QC_policy.json", json.loads((repo / "configs/eeg_qc.json").read_text(encoding="utf-8")))


def denominator_models(config, run, repo):
    from .eeg_audit import prepare_inputs, CORE, FACTOR_LEVELS, CONDITION, TEMPORAL, PREVIOUS
    from .eeg_denominator import compute_psd, paired_inputs, run_models, compare_tables, manuscript_focus, PSD_PARAMETERS
    out = run / "10_eeg_denominator_sensitivity"; out.mkdir()
    frames, excluded = prepare_inputs(config)
    excluded.to_csv(out / "excluded_trials.csv", index=False, encoding="utf-8-sig")
    c = copy.deepcopy(config)
    c["denominator_sensitivity"] = {"matlab": c["fresh"]["matlab"], "psd_cache_dir": str(out / "psd_cache")}
    sources = [{"participant": v, "set_path": str(Path(c["fresh"]["preprocessed_root"]) / f"{v}.set"),
                "fdt_path": set_record(Path(c["fresh"]["preprocessed_root"]) / f"{v}.set",c["fresh"].get("fdt_aliases"))["fdt_path"]}
               for v in sorted(frames[0].Participant.unique())]
    powers, provenance = compute_psd(c, frames, sources, repo, out)
    versions, paired = paired_inputs(frames, powers)
    # The fresh 1–45 input must reproduce the production exporter, including endpoints.
    checks = []
    for trim in TRIMS:
        for metric in [f"{r}_{b}" for r in "FPO" for b in ["theta", "alpha", "beta"]]:
            for suffix in ["_absolute", "_relative"]:
                a = frames[trim][metric + suffix].to_numpy(); b = versions["B"][trim][metric + suffix].to_numpy()
                ok = np.allclose(a, b, rtol=1e-10, atol=1e-12)
                checks.append({"onset_trim_s": trim, "metric": metric + suffix, "max_abs_diff": float(np.max(abs(a-b))), "Pass": bool(ok)})
    pd.DataFrame(checks).to_csv(out / "exporter_PSD_regression.csv", index=False, encoding="utf-8-sig")
    if not all(r["Pass"] for r in checks):
        raise StageBlockedError("Fresh PSD does not reproduce original exporter methods")
    paired.to_csv(out / "paired_psd_integrals.csv", index=False, encoding="utf-8-sig")
    contract = {"version": 1, "purpose": "fresh_formal_eye_eeg", "seed": 20260906,
                "windows": TRIMS, "core_metrics": CORE, "factor_levels": FACTOR_LEVELS,
                "condition_terms": CONDITION, "temporal_terms": TEMPORAL, "previous_terms": PREVIOUS,
                "factor_method": "equal-weight marginal CR2/HTZ reimplementation; historical script unavailable",
                "previous_formula": "PreviousWWR + PreviousComplexity", "PSD": PSD_PARAMETERS}
    dump(out / "analysis_contract.json", contract); dump(out / "PSD_provenance.json", provenance)
    main = run_models(versions["B"], config, repo, out / "main_1_45", out / "analysis_contract.json")
    sensitivity = run_models(versions["C"], config, repo, out / "sensitivity_1_40", out / "analysis_contract.json", reuse_absolute=main)
    comparisons = pd.concat([compare_tables(main[name], sensitivity[name], "fresh_1_45→fresh_1_40", kind)
                             for name, kind in [("family_coefficients", "coefficient"), ("family_factors", "factor"), ("all_model_coefficients", "supplementary_coefficient")]], ignore_index=True)
    comparisons.to_csv(out / "model_comparisons.csv", index=False, encoding="utf-8-sig")
    manuscript_focus(comparisons).to_csv(out / "manuscript_comparisons.csv", index=False, encoding="utf-8-sig")
    fields = ["percent_40_45", "denominator_reduction_percent", "relative_increase_percent"]
    paired.groupby(["onset_trim_s", "roi"])[fields].agg(["mean", "min", "max"]).to_csv(out / "power_fraction_summary.csv", encoding="utf-8-sig")
    dump(out / "summary.json", {"status": "complete", "participants": int(frames[0].Participant.nunique()), "trials": len(frames[0]),
        "absolute_model_reuse": "same-new-run identical numerator/input only", "historical_reuse": False,
        "joint_q_flips": int(comparisons.flip_joint_q.eq(True).sum()), "direction_changes": int(comparisons.direction_changed.eq(True).sum())})


def independent_power_check(config, run, repo):
    from .request_handoff import python_spectra
    out = run / "10_eeg_denominator_sensitivity"
    provenance = json.loads((out / "PSD_provenance.json").read_text(encoding="utf-8"))
    actual = python_spectra(provenance)
    actual.to_csv(out / "python_independent_psd_integrals.csv", index=False, encoding="utf-8-sig")
    expected = pd.read_csv(out / "paired_psd_integrals.csv")
    keys = KEY + ["onset_trim_s", "roi"]
    unique(actual, keys); unique(expected, keys)
    paired = actual.merge(expected, on=keys, suffixes=("_independent", "_export"), how="outer", indicator=True, validate="one_to_one")
    if not paired._merge.eq("both").all():
        raise StageBlockedError("Independent PSD keys differ")
    for field in ["theta", "alpha", "beta", "total_1_45", "total_1_40", "power_40_45"]:
        if not np.allclose(paired[field + "_independent"], paired[field + "_export"], rtol=1e-10, atol=1e-12):
            raise StageBlockedError(f"Independent PSD integration differs: {field}")
    dump(out / "independent_power_check.json", {"spectra": len(actual), "status": "passed", "rtol": 1e-10, "atol": 1e-12})
    verification = out / "independent_verification"; verification.mkdir()
    q = pd.read_excel(config["questionnaire_file"], dtype=object)
    import re
    columns = [next(c for c in q if re.match(re.escape(code) + r"(?:_|\s|$)", str(c)) and not str(c).endswith("_word")) for code in ["Q1.0", "Q1.4", "Q1.5"]]
    reads = [{"id": f"input_{t}", "path": str(out / "main_1_45" / f"trim_{t}s/input.csv")} for t in TRIMS]
    job = {"output": str(verification), "reads": reads, "questionnaire_file": config["questionnaire_file"],
           "questionnaire_columns": columns, "cache": provenance, "source_run": str(out),
           "versions": ["main_1_45", "sensitivity_1_40"]}
    work = repo / ".codex_tmp/fresh_verification" / run.name; work.mkdir(parents=True, exist_ok=True)
    dump(work / "job.json", job)
    env = os.environ.copy(); env["R_LIBS_USER"] = config["fresh"]["verification_r_library"]
    with (verification / "execution.log").open("w", encoding="utf-8") as log:
        subprocess.run([config["fresh"]["verification_rscript"], str(repo / "analysis/r/eeg_request_read_verify.R"),
                        str(work / "job.json")], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rpower = pd.read_csv(verification / "r_psd_integrals.csv")
    joined = actual.merge(rpower, on=keys, how="outer", indicator=True, suffixes=("_Python", "_R"), validate="one_to_one")
    if not joined._merge.eq("both").all(): raise StageBlockedError("R PSD identities differ")
    for field in ["theta", "alpha", "beta", "total_1_45", "total_1_40", "power_40_45"]:
        if not np.allclose(joined[field + "_Python"], joined[field + "_R"], rtol=1e-10, atol=1e-12):
            raise StageBlockedError(f"Python/R PSD integral difference: {field}")
    for record in reads:
        left = pd.read_csv(record["path"], dtype=str).fillna("")
        right = pd.read_csv(verification / "r_reads" / f"{record['id']}.csv", dtype=str).fillna("")
        if not left.equals(right): raise StageBlockedError("Python/R input cell or missing-position difference")
    groups = pd.read_csv(verification / "r_Q1_4_groups.csv")
    original = pd.read_csv(run / "00_teacher_inputs/Q1_4_group_check.csv")
    paired_groups = original.merge(groups, on="Participant", validate="one_to_one", how="outer", indicator=True)
    if not paired_groups._merge.eq("both").all() or not paired_groups.ExerciseFrequency.eq(paired_groups.ExperienceGroup).all():
        raise StageBlockedError("Independent R original Q1.4 grouping differs")
    bh_checks = []
    for version in job["versions"]:
        for kind in ["coefficients", "factors"]:
            left = pd.read_csv(out / version / f"family_{kind}.csv")
            right = pd.read_csv(verification / "r_bh" / f"{version}_{kind}.csv")
            for field in ["within_q", "joint_q"]:
                ok = np.allclose(left[field], right[field], rtol=0, atol=1e-6, equal_nan=True)
                bh_checks.append({"version": version, "kind": kind, "field": field, "Pass": bool(ok), "tests": len(left)})
                if not ok: raise StageBlockedError("Independent R full-family BH differs")
    pd.DataFrame(bh_checks).to_csv(verification / "Python_R_BH_checks.csv", index=False)
    from scipy.stats import t as student_t, f as f_distribution
    statistical_checks=[]
    for version in job["versions"]:
        co=pd.read_csv(out/version/"coefficients.csv")
        valid=co.loc[co.inference_valid.eq(True)]
        p=2*student_t.sf(np.abs(valid.estimate/valid["std.error"]),valid.df)
        statistical_checks.append({"version":version,"check":"CR2 coefficient t/df/p","Pass":bool(np.allclose(p,valid["p.value"],rtol=0,atol=1e-6))})
        ft=pd.read_csv(out/version/"factor_tests.csv");ft=ft.loc[ft.inference_valid.eq(True)]
        p=f_distribution.sf(ft.Fstat,ft.df_num,ft.df_denom)
        statistical_checks.append({"version":version,"check":"HTZ F/df/p","Pass":bool(np.allclose(p,ft["p.value"],rtol=0,atol=1e-6))})
    pd.DataFrame(statistical_checks).to_csv(verification/"statistic_relation_checks.csv",index=False)
    if not all(r["Pass"] for r in statistical_checks):raise StageBlockedError("Independent statistic/df/p relation differs")
    dump(verification / "summary.json", {"status": "passed", "spectra": len(actual), "source_csvs": len(reads),
        "Q1_4_participants": len(groups), "BH_scopes": len(bh_checks), "R_reads_original_sources": True})


def window_robustness(config, run, repo):
    from .r_runner import invoke_r
    core = config["eeg"]["core_metrics"]
    for trim in [0, 5, 15]:
        out = run / "07_eeg_window_robustness" / f"trim_{trim}s"; out.mkdir(parents=True)
        source = run / "10_eeg_denominator_sensitivity/main_1_45" / f"trim_{trim}s/input.csv"
        target = out / "input.csv"; shutil.copyfile(source, target)
        invoke_r(config["rscript"], repo / "analysis/r/eeg_primary_analysis.R",
                 [str(target), str(out), ",".join(f"{v}_relative" for v in core),
                  ",".join(f"log10_{v}_absolute" for v in core), str(config["eeg"]["bootstrap_iterations"]),
                  ",".join(f"{v}_relative" for v in config["eeg"]["secondary_metrics"]),
                  ",".join(f"{v}_relative" for v in config["eeg"]["supplemental_metrics"])], required=True)


def run_fresh(config, config_path, outputs_root, run_id, repo_root, *, resume=False, promote=False):
    repo = Path(repo_root); run = Path(outputs_root) / "teacher_runs" / run_id
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise StageBlockedError("Fresh real-data execution requires clean committed code")
    plan = preflight(config, repo)
    identity = {"config": config, "sha": plan["code_sha"], "inputs": plan["input_inventory"],
                "runtime_lock": plan["runtime_lock"], "aoi_pixel_lock": plan["aoi_pixel_lock"]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if run.exists() and not resume:
        raise StageBlockedError("Use a new run ID or verified same-run --resume")
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise StageBlockedError("Run identity changed; choose a new run ID")
    dump(manifest, {"status": "running", "fingerprint": fingerprint, "git_commit": plan["code_sha"], "scope": "eye-eeg",
                    "config_sha256": file_sha256(config_path), "Python": sys.version, "platform": platform.platform(),
                    "Python_packages":{"numpy":np.__version__,"pandas":pd.__version__,"h5py":h5py.__version__},
                    "runtime_lock": plan["runtime_lock"], "aoi_pixel_lock": plan["aoi_pixel_lock"],
                    "random_seeds":{"EEG_models":20260906,"EEG_bootstrap":20260726,"onset":20260802}})
    dump(run / "source_hashes_before.json", plan["input_inventory"])
    os.environ.pop("PAPER_ANALYSIS_R_REUSE_ROOT", None)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    stage(run, "00_teacher_inputs", fingerprint, lambda: build_design(config, run, plan["set_inventory"]), resume)
    c = resolved_config(config, run); dump(run / "resolved_config.json", c)
    fresh = c["fresh"]
    stage(run, "00_clock_cache", fingerprint, lambda: command([sys.executable, "scripts/build_eeg_clock_cache.py",
        "--acquisition-root", fresh["acquisition_root"], "--eeg-root", fresh["preprocessed_root"],
        "--cache-root", run / "00_clock_cache", "--matlab-command", fresh["matlab"], "--allow-partial"], repo, run / "00_clock_cache/execution.log"), resume)
    stage(run, "00_eeg_raw_export", fingerprint, lambda: command([sys.executable, "scripts/run_eeg_from_raw.py",
        "--eeg_root", fresh["preprocessed_root"], "--outdir", run / "00_eeg_raw_export", "--eeglab_root", fresh["eeglab_root"],
        "--matlab_command", fresh["matlab"], "--eeg-clock-cache-root", run / "00_clock_cache", "--export-eeg-samples"], repo, run / "00_eeg_raw_export/execution.log"), resume)
    stage(run, "00_eeg_trial_qc", fingerprint, lambda: build_qc(c, run, repo, plan["set_inventory"]), resume)
    from .eye import run_eye_stage1, run_eye_stage2, run_eye_stage3_plan, run_eye_stage3
    from .eeg import run_eeg_order, run_eeg_primary
    from .complete import _approve_stage3, _build_sync_outputs
    for name, function in [("01_eye_stage1", run_eye_stage1), ("02_eye_stage2", run_eye_stage2),
                           ("03_eye_stage3_plan", run_eye_stage3_plan)]:
        stage(run, name, fingerprint, lambda n=name, f=function: f(c, config_path=config_path, outdir=run/n, repo_root=repo), resume)
    # User explicitly authorizes the existing Stage 3 workflow. Approval is a
    # separate artifact so that the completed plan's sealed bytes remain intact.
    approval = run / "03_eye_stage3_plan/stage3_plan_approved.txt"
    if not approval.exists():
        _approve_stage3(run / "03_eye_stage3_plan")
        # Extend the plan seal to register the new authorization artifact.
        seal = run / "stage_seals/03_eye_stage3_plan.json"
        record = json.loads(seal.read_text(encoding="utf-8"))
        record["files"].append({"path": approval.relative_to(run).as_posix(), "sha256": file_sha256(approval)})
        dump(seal, record)
    for name, function in [("04_eye_stage3", run_eye_stage3), ("05_eeg_order", run_eeg_order), ("06_eeg_primary", run_eeg_primary)]:
        stage(run, name, fingerprint, lambda n=name, f=function: f(c, config_path=config_path, outdir=run/n, repo_root=repo), resume)
    stage(run, "10_eeg_denominator_sensitivity", fingerprint, lambda: (denominator_models(c, run, repo), independent_power_check(c, run, repo)), resume)
    stage(run, "07_eeg_window_robustness", fingerprint, lambda: window_robustness(c, run, repo), resume)
    from paper_analysis.fusion.clock_sync import run_clock_synchronized_fusion
    stage(run, "00_synchronized_source", fingerprint, lambda: run_clock_synchronized_fusion(
        run / "00_teacher_inputs/scene_manifest_standardized.csv", run / "00_eeg_raw_export/summary/eeg_sample_file_manifest.csv",
        run / "00_synchronized_source", export_pointwise=False, onset_trim_s=10, onset_trim_variants_s=TRIMS), resume)
    stage(run, "08_synchronized_crossmodal", fingerprint, lambda: _build_sync_outputs(Path(outputs_root), run, c), resume)
    from .eye_figures import build_eye_scene_figures
    stage(run, "09_eye_figures", fingerprint, lambda: build_eye_scene_figures(run_root=run,
        mapping_file=Path(c["scene_aoi_mapping"]), output_dir=run / "09_eye_figures",
        manual_registration_file=repo / "configs/eye_scene_registration.json"), resume)
    verify_inputs(plan["input_inventory"]); dump(run / "source_hashes_after.json", plan["input_inventory"])
    from .fresh_handoff import build_report, publish
    build_report(c, run, repo)
    value = json.loads(manifest.read_text(encoding="utf-8")); value["status"] = "complete"
    value["completed_at"] = datetime.now(timezone.utc).isoformat(); dump(manifest, value)
    if promote:
        publish(c, run, Path(outputs_root), repo)
    print(json.dumps({"status": "complete", "run_root": str(run), "git_commit": plan["code_sha"]}, ensure_ascii=False), flush=True)
