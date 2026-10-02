"""Frozen-sample EEG denominator sensitivity, with source drift kept separate."""
from __future__ import annotations

import hashlib
import base64
import itertools
import json
import subprocess
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .eeg_audit import (CORE, DESIGN, FACTOR_LEVELS, KEYS, METRICS, TRIMS,
                        family_tables, outcome_spec, sample_hash)
from .state import StageBlockedError, file_sha256

EFFECTS = ("WWR", "Complexity", "ExperienceGroup", "WWR:Complexity",
           "WWR:ExperienceGroup", "Complexity:ExperienceGroup")
PSD_PARAMETERS = {
    "roi": {"F": ["F3", "F4"], "P": ["P3", "PZ", "P4"], "O": ["O1", "OZ", "O2"]},
    "roi_operation": "time_domain_mean_omitnan_then_remove_nonfinite",
    "endpoints": "inclusive_round_seconds_times_fs_then_add_round_trim_times_fs",
    "minimum_finite_samples": "max(8,round(fs))",
    "welch_window": "min(n,max(round(2*fs),8)); MATLAB numeric-window default Hamming",
    "overlap": "floor(window/2)", "nfft": "max(2^nextpow2(window),window)",
    "integration": "inclusive_frequency_mask_trapz_no_interpolation",
    "bands": {"theta": [4, 7], "alpha": [8, 12], "beta": [13, 30]},
}
METHOD_FILES = ("src/paper_analysis/teacher/eeg_denominator.py",
    "src/paper_analysis/teacher/eeg_audit.py", "analysis/r/eeg_audit.R",
    "analysis/r/eeg_factor_tests.R", "analysis/r/common.R", "scripts/run_teacher_analysis.py",
    "scripts/portable_rscript.cmd", "matlab/eeg_bandpower_pipeline/eeg_denominator_spectrum.m",
    "matlab/eeg_bandpower_pipeline/run_eeg_denominator_psd.m")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def archive_table(archive, trim, filename):
    candidates = [n for n in archive.namelist() if n.startswith("filesource_flat/")
                  and f"onset_window_models_trim_{trim}s__" in n and n.endswith(filename)]
    if len(candidates) != 1:
        raise StageBlockedError(f"Expected one historic {filename} for trim {trim}, found {len(candidates)}")
    with archive.open(candidates[0]) as stream:
        return pd.read_csv(stream), candidates[0]


def freeze_history(config):
    cfg = config["denominator_sensitivity"]
    frames = {}; members = []
    with zipfile.ZipFile(cfg["historical_package"]) as archive:
        for trim in TRIMS:
            frame, member = archive_table(archive, trim, "eeg_onset_order_model_input.csv")
            members.append(member)
            required = {*KEYS, *DESIGN, "PreviousWWR", "PreviousComplexity", "PositionWithinBlock",
                "view_start_s", "view_end_s", "onset_trim_s", "Cond",
                "bad_eeg_quality", "eeg_subject_quality_exclusion",
                *[f"{m}_{s}" for m in METRICS for s in ("absolute", "relative")]}
            if required - set(frame):
                raise StageBlockedError(f"Historic fields missing: {sorted(required-set(frame))}")
            # Historical read.csv/write.csv encoded C0 as missing. Use its archived
            # condition label as evidence; never reconstruct predecessor from QC subset.
            missing = frame.Complexity.isna()
            if not frame.loc[missing, "Cond"].astype(str).str.endswith("C0").all():
                raise StageBlockedError("Missing Complexity without archived C0 evidence")
            frame.loc[missing, "Complexity"] = "C0"
            # The archive also writes the predecessor's empty-string C0 level as
            # blank. A present PreviousWWR distinguishes it from no predecessor.
            previous_c0 = frame.PreviousComplexity.isna() & frame.PreviousWWR.notna()
            frame.loc[previous_c0, "PreviousComplexity"] = "C0"
            has_previous = frame.PreviousWWR.notna()
            if not frame.loc[has_previous, "PreviousComplexity"].isin(["C0", "C1"]).all():
                raise StageBlockedError("Invalid archived predecessor complexity")
            if not frame.loc[frame.PositionWithinBlock.eq(1), ["PreviousWWR", "PreviousComplexity"]].isna().all().all():
                raise StageBlockedError("Unexpected predecessor at block start")
            if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
                raise StageBlockedError("Invalid or duplicate historic trial keys")
            if not frame.onset_trim_s.eq(trim).all():
                raise StageBlockedError("Historic trim labels disagree")
            for field in DESIGN:
                if field in FACTOR_LEVELS:
                    if not frame[field].isin(FACTOR_LEVELS[field]).all():
                        raise StageBlockedError(f"Invalid historic factor {field}")
                elif not np.isfinite(pd.to_numeric(frame[field], errors="coerce")).all():
                    raise StageBlockedError(f"Invalid historic covariate {field}")
            for metric in METRICS:
                for scale in ("relative", "absolute"):
                    x = pd.to_numeric(frame[f"{metric}_{scale}"], errors="coerce")
                    if not np.isfinite(x).all() or (x <= 0).any():
                        raise StageBlockedError("Invalid historic power; sample cannot be reduced")
                frame[f"log10_{metric}_absolute"] = np.log10(frame[f"{metric}_absolute"])
            expected = cfg.get("expected_sample", {"participants": 42, "trials": 461})
            if len(frame) != expected["trials"] or frame.Participant.nunique() != expected["participants"]:
                raise StageBlockedError("Historic sample does not match frozen contract")
            frames[trim] = frame.sort_values(KEYS).reset_index(drop=True)
    reference = frames[0]
    invariant = [*KEYS, *DESIGN, "PreviousWWR", "PreviousComplexity", "PositionWithinBlock",
                 "view_start_s", "view_end_s", "bad_eeg_quality", "eeg_subject_quality_exclusion"]
    for trim in TRIMS[1:]:
        if not frames[trim][invariant].equals(reference[invariant]):
            raise StageBlockedError(f"Frozen sample/design/epoch/QC mismatch at trim {trim}")
    return frames, members


def preflight(config, repo_root):
    cfg = config["denominator_sensitivity"]
    for field in ("historical_package", "preprocessed_root", "matlab", "psd_cache_dir", "candidate_factor_reference"):
        if not cfg.get(field):
            raise StageBlockedError(f"Missing config denominator_sensitivity.{field}")
    for path in (cfg["historical_package"], cfg["matlab"], config["rscript"], cfg["candidate_factor_reference"]):
        if not Path(path).is_file():
            raise StageBlockedError(f"Missing input/launcher: {path}")
    if cfg.get("relative_denominators_hz") != [[1, 45], [1, 40]] or cfg.get("qc_denominator_hz") != [1, 45]:
        raise StageBlockedError("Relative dual denominators and independent frozen QC denominator must be declared")
    frames, members = freeze_history(config)
    sources = []
    for participant in frames[0].Participant.unique():
        base = Path(cfg["preprocessed_root"])/str(participant)
        set_path, fdt_path = base.with_suffix(".set"), base.with_suffix(".fdt")
        if not set_path.is_file() or not fdt_path.is_file():
            raise StageBlockedError("Missing frozen participant .set/.fdt pair")
        sources.append({"participant": str(participant), "set_path": str(set_path), "fdt_path": str(fdt_path)})
    # Read-only executable/dependency checks; no files or analysis outputs written.
    root = Path(repo_root)
    check = subprocess.run([str(config["rscript"]), "-e",
        "stopifnot(all(vapply(c('lme4','clubSandwich','jsonlite','glmmTMB','emmeans','broom.mixed'), requireNamespace, logical(1), quietly=TRUE)))"],
        cwd=root, capture_output=True, text=True)
    if check.returncode:
        raise StageBlockedError(f"R preflight failed: {check.stderr[-1500:]}")
    # MATLAB owns SET decoding, including v7.3. ASCII base64 avoids Windows
    # locale conversion of participant filenames; no probe files are created.
    encoded = base64.b64encode(json.dumps(sources, ensure_ascii=False).encode("utf-8")).decode()
    expression = (
        "assert(license('test','Signal_Toolbox'),'Signal Toolbox unavailable');"
        f"s=jsondecode(native2unicode(matlab.net.base64decode('{encoded}'),'UTF-8'));"
        "for i=1:numel(s); E=load(s(i).set_path,'-mat','srate','pnts','nbchan','chanlocs','datfile');"
        "assert(E.srate>90 && E.pnts>0 && E.nbchan>0,'Invalid EEG header');"
        "declared=fullfile(fileparts(s(i).set_path),E.datfile);"
        "if isfile(declared);s(i).fdt_path=declared;end;"
        "d=dir(s(i).fdt_path);assert(d.bytes==4*E.nbchan*E.pnts,'FDT/header size mismatch');"
        "labels=upper(string({E.chanlocs.labels}));"
        "assert(any(ismember(labels,{'F3','F4'})) && any(ismember(labels,{'P3','PZ','P4'})) && any(ismember(labels,{'O1','OZ','O2'})),'Missing ROI');"
        "end;fprintf('EEG_PREFLIGHT:%s\\n',matlab.net.base64encode(unicode2native(jsonencode(s),'UTF-8')));"
    )
    probe = subprocess.run([cfg["matlab"], "-batch", expression], cwd=root, capture_output=True, encoding="utf-8", errors="replace")
    if probe.returncode:
        raise StageBlockedError(f"MATLAB header/dependency preflight failed: {probe.stdout[-1500:]} {probe.stderr[-1500:]}")
    marker = next((line.split("EEG_PREFLIGHT:", 1)[1] for line in probe.stdout.splitlines() if "EEG_PREFLIGHT:" in line), None)
    if marker is None: raise StageBlockedError("MATLAB preflight did not return header verification")
    sources = json.loads(base64.b64decode(marker).decode("utf-8"))
    return frames, sources, {"purpose": "frozen_sample_denominator_sensitivity", "historic_members": members,
        "common_trials": len(frames[0]), "common_participants": int(frames[0].Participant.nunique()),
        "sample_hash": sample_hash(frames[0]), "windows": list(TRIMS), "matlab_headers_and_signal_toolbox": True,
        "r_dependencies_available": True, "qc": "historic_values_and_inclusion_frozen",
        "factor_method": "reimplemented_equal_weight_marginal_CR2_HTZ"}


def cache_identity(sources, trials, method_hashes):
    body = {"sources": sources, "epochs": trials.to_dict("records"),
            "parameters": PSD_PARAMETERS, "implementation": method_hashes}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_cache(cache, key):
    manifest = Path(cache)/"cache_manifest.json"
    if not manifest.is_file():
        raise StageBlockedError("Incomplete PSD cache; refusing reuse")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta["cache_key"] != key or meta["status"] != "complete":
        raise StageBlockedError("PSD cache identity/status mismatch")
    if not meta["output_hashes"] or not any(x["path"].endswith(".mat") for x in meta["output_hashes"]):
        raise StageBlockedError("Cache lacks complete spectra")
    for record in meta["output_hashes"]:
        if not (Path(cache)/record["path"]).is_file() or file_sha256(Path(cache)/record["path"]) != record["sha256"]:
            raise StageBlockedError("PSD cache artifact hash mismatch")


def compute_psd(config, frames, sources, root, out):
    cfg = config["denominator_sensitivity"]
    for source in sources:
        source["set_sha256"] = file_sha256(source["set_path"])
        source["fdt_sha256"] = file_sha256(source["fdt_path"])
    trials = pd.concat([f[[*KEYS, "view_start_s", "view_end_s", "onset_trim_s"]] for f in frames.values()])
    trials = trials.sort_values([*KEYS, "onset_trim_s"]).reset_index(drop=True)
    code = {p: file_sha256(root/p) for p in METHOD_FILES if p.endswith(".m")}
    key = cache_identity(sources, trials, code)
    cache = Path(cfg["psd_cache_dir"])/key
    reused = cache.exists()
    if reused:
        verify_cache(cache, key)
    else:
        cache.mkdir(parents=True, exist_ok=False)
        trials.to_csv(cache/"frozen_epochs.csv", index=False, encoding="utf-8-sig")
        write_json(cache/"matlab_config.json", {"cache_key": key, "sources": sources,
            "trials": str(cache/"frozen_epochs.csv"), "cache_dir": str(cache)})
        quote = lambda p: str(p).replace("'", "''").replace("\\", "/")
        expression = f"addpath('{quote(root/'matlab/eeg_bandpower_pipeline')}'); run_eeg_denominator_psd('{quote(cache/'matlab_config.json')}')"
        with (out/"MATLAB_execution.log").open("w", encoding="utf-8") as log:
            subprocess.run([cfg["matlab"], "-batch", expression], stdout=log, stderr=subprocess.STDOUT, cwd=root, check=True)
        for source in sources:
            for kind in ("set", "fdt"):
                if file_sha256(source[f"{kind}_path"]) != source[f"{kind}_sha256"]:
                    raise StageBlockedError("Source waveform changed while creating cache")
        records = [{"path": p.name, "sha256": file_sha256(p)} for p in sorted(cache.glob("*")) if p.is_file()]
        write_json(cache/"cache_manifest.json", {"status": "complete", "cache_key": key,
            "parameters": PSD_PARAMETERS, "sources": sources, "output_hashes": records})
        verify_cache(cache, key)
    powers = pd.read_csv(cache/"paired_powers.csv")
    return powers, {"cache_dir": str(cache), "cache_key": key, "reused": reused, "sources": sources}


def paired_inputs(frames, powers):
    keys = [*KEYS, "onset_trim_s", "roi"]
    expected = pd.DataFrame([(*row, trim, roi) for trim, f in frames.items()
        for row in f[KEYS].itertuples(index=False, name=None) for roi in ("F", "P", "O")], columns=keys)
    if powers.duplicated(keys).any() or len(powers) != len(expected):
        raise StageBlockedError("Duplicate or missing PSD rows; frozen sample cannot be reduced")
    paired = expected.merge(powers, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise StageBlockedError("PSD keys differ from frozen sample")
    for col in ("total_1_45", "total_1_40", "theta", "alpha", "beta"):
        if not np.isfinite(paired[col]).all() or paired[col].le(0).any():
            raise StageBlockedError(f"Invalid PSD {col}; frozen sample cannot be reduced")
    if not np.isfinite(paired.power_40_45).all() or paired.power_40_45.lt(0).any() or (paired.total_1_40 > paired.total_1_45).any():
        raise StageBlockedError("Invalid band integration")
    versions = {"A": frames, "B": {}, "C": {}}
    for version, denom in (("B", "total_1_45"), ("C", "total_1_40")):
        for trim, original in frames.items():
            new = original.copy()
            for roi in ("F", "P", "O"):
                ps = paired.loc[paired.onset_trim_s.eq(trim) & paired.roi.eq(roi)].set_index(KEYS)
                ps = ps.reindex(pd.MultiIndex.from_frame(new[KEYS]))
                for band in ("theta", "alpha", "beta"):
                    m = f"{roi}_{band}"
                    new[f"{m}_absolute"] = ps[band].to_numpy()
                    if m in new:
                        new[m] = ps[band].to_numpy()
                    new[f"{m}_relative"] = (ps[band]/ps[denom]).to_numpy()
                    new[f"log10_{m}_absolute"] = np.log10(ps[band].to_numpy())
            versions[version][trim] = new
    for trim in TRIMS:
        b, c = versions["B"][trim], versions["C"][trim]
        changed = {f"{m}_relative" for m in METRICS}
        if not b.drop(columns=list(changed)).equals(c.drop(columns=list(changed))):
            raise StageBlockedError("B/C differ beyond relative powers")
    paired = paired.drop(columns="_merge")
    paired["percent_40_45"] = 100*paired.power_40_45/paired.total_1_45
    paired["denominator_reduction_percent"] = 100*(paired.total_1_45-paired.total_1_40)/paired.total_1_45
    paired["relative_increase_percent"] = 100*(paired.total_1_45/paired.total_1_40-1)
    return versions, paired


def factor_families(factors):
    keys = ["onset_trim_s", "outcome", "model", "term"]
    if factors.duplicated(keys).any():
        raise StageBlockedError("Duplicate factor test")
    tables = []; statuses = []
    for name, outcomes in (("factor_core_relative", [f"{m}_relative" for m in CORE]),
            ("factor_expanded_relative", [f"{m}_relative" for m in METRICS]),
            ("factor_expanded_absolute", [f"log10_{m}_absolute" for m in METRICS])):
        grid = pd.DataFrame(itertools.product(TRIMS, outcomes, ["Model1"], EFFECTS), columns=keys)
        table = grid.merge(factors, on=keys, how="left", validate="one_to_one")
        table["family_id"] = name; table["within_q"] = np.nan; table["joint_q"] = np.nan
        p = pd.to_numeric(table["p.value"], errors="coerce")
        valid = np.isfinite(p) & p.between(0, 1) & table.inference_valid.eq(True)
        valid &= np.isfinite(table.Fstat) & table.df_num.gt(0) & np.isfinite(table.df_denom) & table.df_denom.gt(0)
        for scope, members in [("joint", table.index), *[(f"window_{t}", table.index[table.onset_trim_s.eq(t)]) for t in TRIMS]]:
            complete = bool(valid.loc[members].all())
            statuses.append({"family_id": name, "scope": scope, "expected_tests": len(members),
                "valid_tests": int(valid.loc[members].sum()), "status": "complete" if complete else "incomplete_no_final_q"})
            if complete:
                table.loc[members, "joint_q" if scope == "joint" else "within_q"] = multipletests(p.loc[members], method="fdr_bh")[1]
        tables.append(table)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(statuses)


def run_models(frames, config, root, folder, contract, reuse_absolute=None):
    folder.mkdir()
    spec = outcome_spec()
    if reuse_absolute is not None:
        spec = spec.loc[spec.scale.eq("relative_power")]
    spec.to_csv(folder/"outcomes.csv", index=False)
    collected = {n: [] for n in ("coefficients", "diagnostics", "model_samples", "factor_tests", "contrast_matrices")}
    for trim, frame in frames.items():
        dest = folder/f"trim_{trim}s"; dest.mkdir()
        frame.to_csv(dest/"input.csv", index=False, encoding="utf-8-sig")
        # Portable Windows R can corrupt non-ASCII command-line filenames even
        # when CSV contents are read as UTF-8. Stage exact files under an ASCII
        # workspace and copy outputs back with Python's Unicode filesystem API.
        staging_root=root/".codex_tmp"; staging_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="eeg_r_",dir=staging_root) as temporary:
            stage=Path(temporary)
            for source,target in ((dest/"input.csv","input.csv"),(folder/"outcomes.csv","outcomes.csv"),
                                  (Path(contract),"contract.json")):
                shutil.copyfile(source,stage/target)
            for name in ("eeg_audit.R","eeg_factor_tests.R","common.R"):
                shutil.copyfile(root/"analysis/r"/name,stage/name)
            with (dest/"R_execution.log").open("w",encoding="utf-8") as log:
                subprocess.run([config["rscript"],"eeg_audit.R","input.csv",".","outcomes.csv","contract.json"],
                    cwd=stage,stdout=log,stderr=subprocess.STDOUT,check=True)
            for name in (*collected,"R_session_info"):
                filename=f"{name}.txt" if name=="R_session_info" else f"{name}.csv"
                shutil.copyfile(stage/filename,dest/filename)
        for name in collected:
            table = pd.read_csv(dest/f"{name}.csv")
            table["onset_trim_s"] = trim
            if reuse_absolute is not None:
                original = reuse_absolute[name]
                table = pd.concat([table, original.loc[original.onset_trim_s.eq(trim) & original.outcome.str.startswith("log10_")]], ignore_index=True)
                table.to_csv(dest/f"{name}.csv", index=False, encoding="utf-8-sig")
            collected[name].append(table)
    result = {name: pd.concat(tables, ignore_index=True) for name, tables in collected.items()}
    diag = result["diagnostics"].drop(columns=["inference_valid", "sample_hash"], errors="ignore")
    hashes = pd.DataFrame([{"onset_trim_s": t, "outcome": o, "model": m, "sample_hash": sample_hash(g)}
        for (t, o, m), g in result["model_samples"].groupby(["onset_trim_s", "outcome", "model"])])
    diag = diag.merge(hashes, on=["onset_trim_s", "outcome", "model"], how="left", validate="one_to_one")
    diag["inference_valid"] = diag.fit_status.eq("fit") & diag.converged.eq(True) & diag.singular.eq(False) & diag.rank_deficient.eq(False) & diag.cr2_status.eq("computed")
    result["diagnostics"] = diag
    for name in ("coefficients", "factor_tests"):
        table = result[name].drop(columns=["inference_valid", "sample_hash"], errors="ignore")
        result[name] = table.merge(diag[["onset_trim_s", "outcome", "model", "inference_valid", "sample_hash"]],
            on=["onset_trim_s", "outcome", "model"], validate="many_to_one")
    result["family_coefficients"], cs = family_tables(result["coefficients"])
    result["family_factors"], fs = factor_families(result["factor_tests"])
    result["family_status"] = pd.concat([cs, fs], ignore_index=True)
    unadjusted = result["coefficients"].copy()
    unadjusted["family_id"] = "all_model_coefficients_unadjusted"
    unadjusted["within_q"] = np.nan; unadjusted["joint_q"] = np.nan
    result["all_model_coefficients"] = unadjusted
    for name, table in result.items():
        table.to_csv(folder/f"{name}.csv", index=False, encoding="utf-8-sig")
    return result


def compare_tables(left, right, pair, kind):
    keys = ["onset_trim_s", "outcome", "model", "term", "family_id"]
    table = left.merge(right, on=keys, suffixes=("_from", "_to"), validate="one_to_one", how="outer", indicator=True)
    if not table._merge.eq("both").all():
        raise StageBlockedError("Unpaired model/test family identities")
    table = table.drop(columns="_merge"); table["pair"] = pair; table["test_kind"] = kind
    for col in ("estimate", "p.value", "within_q", "joint_q", "Fstat", "df", "df_denom"):
        if f"{col}_from" in table:
            table[f"delta_{col}"] = table[f"{col}_to"]-table[f"{col}_from"]
    for col in ("p.value", "within_q", "joint_q"):
        x, y = table[f"{col}_from"], table[f"{col}_to"]
        table[f"flip_{col}"] = (x.lt(.05) != y.lt(.05)).astype("boolean").mask(x.isna() | y.isna())
    x, y = table.estimate_from, table.estimate_to
    table["direction_changed"] = (np.sign(x) != np.sign(y)).astype("boolean").mask(x.isna() | y.isna())
    table["estimate_percent_change"] = (100*(y-x)/x.abs()).where(x.abs().ge(1e-6))
    return table


def historical_regression(config, result, out):
    rows = []
    with zipfile.ZipFile(config["denominator_sensitivity"]["historical_package"]) as archive:
        for trim in TRIMS:
            df, _ = archive_table(archive, trim, "eeg_cr2_robust_results.csv")
            df = df.loc[df.model.eq("Model1") & df.outcome.isin([f"{m}_relative" for m in CORE])].copy()
            df["onset_trim_s"] = trim; rows.append(df)
    keys = ["onset_trim_s", "outcome", "model", "term"]
    previous = pd.concat(rows)
    comparison = previous.merge(result["coefficients"], on=keys, suffixes=("_historic", "_new"), how="left", validate="one_to_one")
    comparison["beta_error"] = (comparison.estimate_historic-comparison.estimate_new).abs()
    comparison["p_error"] = (comparison["p.value_historic"]-comparison["p.value_new"]).abs()
    comparison.to_csv(out/"historical_coefficient_regression.csv", index=False, encoding="utf-8-sig")
    if len(comparison)!=240 or not comparison.beta_error.le(1e-8).all() or not comparison.p_error.le(1e-6).all():
        raise StageBlockedError("A historical 240-coefficient regression failed")
    candidate = pd.read_csv(config["denominator_sensitivity"]["candidate_factor_reference"])
    mapping = {"core_relative": "factor_core_relative", "expanded_relative": "factor_expanded_relative", "expanded_absolute": "factor_expanded_absolute"}
    candidate["family_id"] = candidate.family.map(mapping); candidate["term"] = candidate.effect
    keys = ["onset_trim_s", "outcome", "term", "family_id"]
    factor = candidate.merge(result["family_factors"], on=keys, how="outer", suffixes=("_reference", "_new"), validate="one_to_one", indicator=True)
    factor["p_error"] = (factor.raw_p-factor["p.value"]).abs()
    factor["joint_q_error"] = (factor.joint_q_reference-factor.joint_q_new).abs()
    factor.to_csv(out/"factor_independent_regression.csv", index=False, encoding="utf-8-sig")
    if len(factor)!=528 or not factor._merge.eq("both").all() or not factor.p_error.le(1e-6).all() or not factor.joint_q_error.le(1e-6).all():
        raise StageBlockedError("A factor-level independent reference regression failed")
    return {"core_coefficient_rows": 240, "max_beta_error": float(comparison.beta_error.max()),
        "max_p_error": float(comparison.p_error.max()), "factor_rows": 528,
        "factor_max_p_error": float(factor.p_error.max()), "factor_max_joint_q_error": float(factor.joint_q_error.max())}


def manuscript_focus(comparison):
    masks = {
        "表6_额区theta_WWR45×复杂度": comparison.family_id.eq("condition_core_relative") & comparison.outcome.eq("F_theta_relative") & comparison.term.eq("WWRWWR45:ComplexityC1"),
        "枕区theta_复杂度边际效应": comparison.family_id.eq("factor_core_relative") & comparison.outcome.eq("O_theta_relative") & comparison.term.eq("Complexity"),
        "alpha_Block": comparison.family_id.isin(["temporal_relative", "temporal_absolute"]) & comparison.outcome.str.contains("_alpha_") & comparison.term.eq("Block"),
        "theta_Block": comparison.family_id.isin(["temporal_relative", "temporal_absolute"]) & comparison.outcome.str.contains("_theta_") & comparison.term.eq("Block"),
        "alpha_Position": comparison.family_id.isin(["temporal_relative", "temporal_absolute"]) & comparison.outcome.str.contains("_alpha_") & comparison.term.eq("PositionWithinBlockCentered"),
        "顶区beta_WWR总体效应": comparison.family_id.eq("factor_expanded_relative") & comparison.outcome.eq("P_beta_relative") & comparison.term.eq("WWR"),
        "上一场景": comparison.family_id.eq("previous_core_relative"),
    }
    rows = []
    for claim, mask in masks.items():
        part = comparison.loc[mask].copy(); part["manuscript_claim"] = claim; rows.append(part)
    return pd.concat(rows, ignore_index=True)


def finish_report(out, results, powers, regression, provenance):
    comparisons = []
    for a,b in (("A","B"),("B","C"),("A","C")):
        for name, kind in (("family_coefficients", "coefficient"), ("family_factors", "factor"),
                           ("all_model_coefficients", "coefficient_unadjusted")):
            comparisons.append(compare_tables(results[a][name], results[b][name], f"{a}→{b}", kind))
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(out/"model_comparisons.csv", index=False, encoding="utf-8-sig")
    focus = manuscript_focus(comparison)
    focus.to_csv(out/"manuscript_comparisons.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for (pair, family), group in comparison.groupby(["pair","family_id"]):
        summaries.append({"pair": pair, "family_id": family, "tests": len(group),
            "p_flips": int(group["flip_p.value"].sum()), "within_q_flips": int(group.flip_within_q.sum()),
            "joint_q_flips": int(group.flip_joint_q.sum()), "direction_changes": int(group.direction_changed.sum()),
            "max_abs_delta_p": float(group["delta_p.value"].abs().max()),
            "max_abs_delta_joint_q": float(group.delta_joint_q.abs().max()) if group.delta_joint_q.notna().any() else None})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/"comparison_summary.csv", index=False, encoding="utf-8-sig")
    fraction_rows=[]
    for trim in TRIMS:
        for roi in ("ALL", "F", "P", "O"):
            p = powers.loc[powers.onset_trim_s.eq(trim)]
            if roi != "ALL": p = p.loc[p.roi.eq(roi)]
            row={"onset_trim_s":trim,"roi":roi,"n_spectra":len(p)}
            for col in ("percent_40_45","denominator_reduction_percent","relative_increase_percent"):
                for label, fun in (("mean",np.mean),("median",np.median),("min",np.min),("max",np.max)):
                    row[f"{col}_{label}"]=float(fun(p[col]))
            row["pooled_40_45_percent"]=float(100*p.power_40_45.sum()/p.total_1_45.sum())
            fraction_rows.append(row)
    fractions=pd.DataFrame(fraction_rows)
    fractions.to_csv(out/"power_fraction_summary.csv",index=False,encoding="utf-8-sig")
    changed=[]
    for trim in TRIMS:
        a,b=results["A"]["inputs"][trim], results["B"]["inputs"][trim]
        for metric in METRICS:
            for scale in ("relative","absolute"):
                delta=b[f"{metric}_{scale}"]-a[f"{metric}_{scale}"]
                changed.append({"onset_trim_s":trim,"metric":metric,"scale":scale,
                    "changed_above_1e_minus10":int(delta.abs().gt(1e-10).sum()),"max_absolute_delta":float(delta.abs().max())})
    drift=pd.DataFrame(changed);drift.to_csv(out/"input_reproduction_differences.csv",index=False,encoding="utf-8-sig")
    valid=all(r["diagnostics"].inference_valid.all() and r["family_status"].status.eq("complete").all() for r in results.values())
    status="complete" if valid else "analysis_issues"
    lines=["# EEG 1–40 Hz 分母敏感性分析", "", f"状态：{status}。冻结 0805 的{provenance['common_participants']}人、{provenance['common_trials']}共同试次，0/5/10/15 s四窗口。",
        "", "A为0805历史1–45 Hz；B为当前预处理源文件复现1–45 Hz；C与B使用完全相同PSD和θ/α/β分子，仅改为1–40 Hz分母。B→C为分母效应，A→B为输入复现差异。",
        "", "factor-level 使用重新实现的等权边际 CR2/HTZ，已对照独立核对表；不声称找到历史factor-level脚本。参与者是聚类单位。上一场景保持PreviousWWR + PreviousComplexity，论文交互项描述与历史代码不一致。",
        "", "预处理来源：现有.set/.fdt，无重新滤波或ICA。既有核查为band-pass 0.5–40 Hz、notch 49–51 Hz、average reference、runica(PCA=7)。已计算ICA，历史未发现pop_subcomp或标记删除；不能称为已删除ICA伪迹。QC和纳入名单使用历史值。",
        "", "## 主要比较", "", "|比较|检验族|原始p翻转|窗口q翻转|联合q翻转|方向变化|", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"|{row['pair']}|{row['family_id']}|{row['p_flips']}|{row['within_q_flips']}|{row['joint_q_flips']}|{row['direction_changes']}|")
    lines += ["", "## 正文逐项核对", "", "以下逐项给出四窗口联合BH判断；单自由度方向可比较，多自由度总体F不报告虚构β。"]
    for (pair,claim), g in focus.groupby(["pair","manuscript_claim"],sort=False):
        lines.append(f"\n- {pair}，{claim}：{len(g)}项；方向变化{int(g.direction_changed.sum())}项，原始p翻转{int(g['flip_p.value'].sum())}项，窗口q翻转{int(g.flip_within_q.sum())}项，联合q翻转{int(g.flip_joint_q.sum())}项；原联合显著{int(g.joint_q_from.lt(.05).sum())}项，新联合显著{int(g.joint_q_to.lt(.05).sum())}项。")
    lines += ["", "## 40–45 Hz 与实际分母变化", "", "|窗口s|谱数|40–45占比均值%|实际分母减少均值%|relative增加均值%|", "|---:|---:|---:|---:|---:|"]
    for _,r in fractions.loc[fractions.roi.eq("ALL")].iterrows():
        lines.append(f"|{r.onset_trim_s:g}|{r.n_spectra:g}|{r.percent_40_45_mean:.6f}|{r.denominator_reduction_percent_mean:.6f}|{r.relative_increase_percent_mean:.6f}|")
    lines += ["", "频率掩码含端点，未插值。500 Hz、NFFT1024时，1–40末点39.55078125 Hz，40–45首点40.0390625 Hz；两段之间梯形面积导致40–45占比与实际分母减少不同。每个试次、窗口和ROI独立积分，未使用统一比例缩放。",
        "", "## 验证与溯源", "", f"历史240条核心Model1回归：β最大绝对误差{regression['max_beta_error']:.3g}，p最大绝对误差{regression['max_p_error']:.3g}。独立factor核对528条，p最大误差{regression['factor_max_p_error']:.3g}。",
        "", "论文的小幅p/q差异保留，不调整统计方法追求逐位一致。全部对照、置信区间、自由度、对比矩阵、诊断和输入哈希在本目录。既有正式多模态结果未覆盖。"]
    (out/"中文简报.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(out/"summary.json", {"status":status,"regression":regression,"provenance":provenance,
        "comparison_summary":summaries,"main_comparison":"B→C", "factor_method":"reimplemented_equal_weight_marginal_CR2_HTZ"})
    return status


def run_eeg_denominator(config, *, config_path, outdir, repo_root, r_required=True):
    if not r_required:
        raise StageBlockedError("Sensitivity analysis requires R; use --dry-run for read-only preflight")
    root=Path(repo_root);out=Path(outdir)
    if out.exists():raise StageBlockedError("Use a new sensitivity output directory")
    frames,sources,provenance=preflight(config,root)
    dirty=subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True)
    if dirty.strip():raise StageBlockedError("Commit implementation first; real sensitivity analysis requires a clean tracked/untracked Git workspace")
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    records=[{"path":str(root/p),"sha256":file_sha256(root/p)} for p in METHOD_FILES]
    records += [{"path":str(config_path),"sha256":file_sha256(config_path)},
        {"path":config["denominator_sensitivity"]["historical_package"],"sha256":file_sha256(config["denominator_sensitivity"]["historical_package"])},
        {"path":config["denominator_sensitivity"]["candidate_factor_reference"],"sha256":file_sha256(config["denominator_sensitivity"]["candidate_factor_reference"])}]
    out.mkdir(parents=True)
    manifest={"status":"running","git_sha":sha,"purpose":"frozen_sample_denominator_sensitivity",
        "input_code_hashes":records,"config":config,"sample":provenance}
    write_json(out/"run_manifest.json",manifest)
    contract={"factor_levels":FACTOR_LEVELS,"seed":20260906,"purpose":manifest["purpose"],
        "factor_method":"reimplemented_equal_weight_marginal_CR2_HTZ", "previous_formula":"PreviousWWR + PreviousComplexity",
        "psd_parameters":PSD_PARAMETERS,"sample":provenance}
    write_json(out/"analysis_contract.json",contract)
    try:
        powers,cache=compute_psd(config,frames,sources,root,out)
        versions,powers=paired_inputs(frames,powers)
        powers.to_csv(out/"paired_psd_integrals.csv",index=False,encoding="utf-8-sig")
        write_json(out/"psd_cache_reference.json",cache)
        results={}
        for version in ("A","B","C"):
            results[version]=run_models(versions[version],config,root,out/version,out/"analysis_contract.json",
                reuse_absolute=results.get("B") if version=="C" else None)
            results[version]["inputs"]=versions[version]
            if version=="A":regression=historical_regression(config,results[version],out)
        for name in ("coefficients","factor_tests"):
            b=results["B"][name].loc[results["B"][name].outcome.str.startswith("log10_")].reset_index(drop=True)
            c=results["C"][name].loc[results["C"][name].outcome.str.startswith("log10_")].reset_index(drop=True)
            pd.testing.assert_frame_equal(b,c,check_exact=True)
        status=finish_report(out,results,powers,regression,provenance)
        for r in records:
            if file_sha256(r["path"])!=r["sha256"]:raise StageBlockedError("Input or implementation changed during execution")
        for s in sources:
            for kind in ("set","fdt"):
                if file_sha256(s[f"{kind}_path"])!=s[f"{kind}_sha256"]:raise StageBlockedError("Source changed during execution")
        manifest.update(status=status,psd_cache=cache,regression=regression,absolute_results="B/C checked identical then reused")
        write_json(out/"run_manifest.json",manifest)
        write_json(out/"output_hashes.json",[{"path":p.relative_to(out).as_posix(),"sha256":file_sha256(p)}
            for p in sorted(out.rglob("*")) if p.is_file()])
        return {"manifest":out/"run_manifest.json"}
    except Exception as exc:
        manifest.update(status="failed",error=str(exc));write_json(out/"run_manifest.json",manifest);raise
