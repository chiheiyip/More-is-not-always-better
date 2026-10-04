"""Read-only EEG request verification and transactional, current-request delivery.

The old calculation is never relabelled as a new calculation. Both language
readers consume source files, not each other's converted model inputs.
"""
from __future__ import annotations

import csv
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t as student_t, f as fisher_f
from statsmodels.stats.multitest import multipletests

from .state import file_sha256, git_commit
from .eeg_denominator import METHOD_FILES, verify_cache

KEY = ["Participant", "GlobalTrialOrder"]
TEST_KEY = ["onset_trim_s", "outcome", "model", "term", "family_id"]
BANDS = {"total_1_45": (1,45), "total_1_40": (1,40), "power_40_45": (40,45),
         "theta": (4,7), "alpha": (8,12), "beta": (13,30)}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")


def raw_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")


def unique_keys(frame, keys):
    if frame[keys].isna().any().any() or frame[keys].astype(str).eq("").any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate keys: {keys}")


def compare_cells(left, right, *, keys=None, numeric=False):
    if list(left.columns) != list(right.columns):
        raise ValueError("Column identity/order mismatch")
    if keys:
        unique_keys(left, keys); unique_keys(right, keys)
        left = left.sort_values(keys).reset_index(drop=True)
        right = right.sort_values(keys).reset_index(drop=True)
    if left.shape != right.shape:
        raise ValueError("Row/column count mismatch")
    for col in left:
        if numeric and pd.api.types.is_numeric_dtype(left[col]):
            if not np.allclose(left[col], right[col], rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError(f"Numeric mismatch in {col}")
        elif not left[col].fillna("").astype(str).equals(right[col].fillna("").astype(str)):
            raise ValueError(f"Cell mismatch in {col}")
    return int(left.size)


def bh_checks(frame):
    unique_keys(frame, TEST_KEY)
    records = []
    for family, group in frame.groupby("family_id", sort=False):
        for scope, members, column in [("joint", group, "joint_q"), *[(f"window_{w}", g, "within_q") for w,g in group.groupby("onset_trim_s")]]:
            p = members["p.value"].to_numpy(float)
            valid = np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and members.inference_valid.eq(True).all()
            q = multipletests(p, method="fdr_bh")[1] if valid else np.full(len(p), np.nan)
            if not np.allclose(q, members[column], rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError(f"BH mismatch: {family}/{scope}")
            records.append({"family_id":family,"scope":scope,"tests":len(p),"valid":bool(valid),"max_q_error":float(np.nanmax(np.abs(q-members[column]))) if valid else None})
    return records


def independent_design_column(frame,label):
    if label=="(Intercept)":return np.ones(len(frame))
    if ":" in label:
        columns=[independent_design_column(frame,p) for p in label.split(":")]
        return np.prod(columns,axis=0)
    if label in ("Block","PositionWithinBlockCentered"):
        return frame[label].to_numpy(float)
    for field in sorted(("WWR","Complexity","ExperienceGroup","Gender","OrderGroup","PreviousWWR","PreviousComplexity"),key=len,reverse=True):
        if label.startswith(field):return frame[field].eq(label[len(field):]).to_numpy(float)
    raise ValueError(f"Unknown design coefficient {label}")


def questionnaire_identity(value):
    # Independently reproduce the existing discovery.py ethnic-name rule.
    parts=[p.strip() for p in re.split(r"[·•]",str(value).strip()) if p.strip()]
    return parts[0] if parts else ""


def set_sample_count(path):
    try:
        value=loadmat(path,variable_names=["pnts"],squeeze_me=True)["pnts"]
    except NotImplementedError:
        import h5py
        with h5py.File(path,"r") as source:
            value=source["pnts"][()]
    number=float(np.asarray(value).item())
    if not np.isfinite(number) or number<=0 or number!=int(number):
        raise ValueError("Invalid SET sample count")
    return int(number)


def preprocessing_flags(text):
    band=[];notch=[]
    for match in re.finditer(r"pop_eegfiltnew\(\s*EEG\s*,([^)]*)\)",text):
        arguments=match.group(1)
        named=re.search(r"'locutoff'\s*,\s*([\d.]+)\s*,\s*'hicutoff'\s*,\s*([\d.]+)",arguments)
        positional=re.match(r"\s*([\d.]+)\s*,\s*([\d.]+)",arguments)
        parsed=named or positional
        if parsed is None:continue
        low,high=map(float,parsed.groups())
        if (low,high)==(.5,40):band.append(match.start())
        if (low,high)==(49,51):notch.append(match.start())
    return {"bandpass":bool(band),"notch":bool(notch),
            "order":"band-pass→notch" if band and notch and band[-1]<notch[-1] else "notch→band-pass" if band and notch else "requires_review",
            "average_reference":bool(re.search(r"pop_reref\(\s*EEG\s*,\s*\[\s*\]",text)),
            "ICA_run":"pop_runica" in text,"ICA_component_removal":"pop_subcomp" in text}


def preprocessing_records(cache):
    rows=[]
    for index,wave in enumerate(cache["sources"],1):
        metadata=loadmat(Path(cache["cache_dir"])/f"spectra_{index:03d}.mat",squeeze_me=True,struct_as_record=False)["metadata"]
        rows.append({"Participant":wave["participant"],**preprocessing_flags(str(metadata.preprocessing_history)),"reference":str(metadata.reference)})
    return rows


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("outputs_root", "source_run", "request_md", "questionnaire_file", "historical_package", "rscript", "r_library", "replay_rscript", "delivery_root", "archive_root", "node", "runtime_modules"):
        p=Path(config[key])
        config[key] = str((p if p.is_absolute() else Path(path).resolve().parent/p).resolve())
    return config


def preflight(config, repo):
    required = [config[k] for k in ("outputs_root","source_run","request_md","questionnaire_file","historical_package","rscript","replay_rscript","node","runtime_modules")]
    if not all(Path(p).exists() for p in required):
        raise ValueError("Missing request input/runtime")
    destination, archive = Path(config["delivery_root"]), Path(config["archive_root"])
    if destination == archive or destination in archive.parents or archive in destination.parents:
        raise ValueError("Archive must be outside and separate from delivery")
    run = Path(config["source_run"])
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Source analysis is incomplete")
    files = sorted(str(p) for p in destination.iterdir()) if destination.exists() else []
    return {"source_run":str(run),"analysis_git_sha":manifest["git_sha"],"delivery_root":str(destination),
            "archive_root":str(archive),"existing_entries_to_archive":files,
            "reuse":"waveforms, complete PSD, previously verified model outputs and bootstrap",
            "new_work":"independent file reads, integrals, BH, cited model refits, focused handoff"}


def r_environment(config):
    env = os.environ.copy()
    for field in ("R_HOME", "LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_MONETARY", "LC_TIME", "LANG"):
        env.pop(field, None)
    # Keep the installed 4.5 user library as well as the request-specific library.
    env["R_LIBS_USER"] = str(config["r_library"]) + ";" + str(Path.home()/"AppData/Local/R/win-library/4.5")
    return env


def r_call(config, repo, script, payload, output, label):
    jobs = repo/".codex_tmp/request_jobs"/output.name
    jobs.mkdir(parents=True,exist_ok=True)
    job = jobs / f"{label}_job.json"
    dump(job, payload)
    shutil.copyfile(job,output/f"{label}_job.json")
    env=r_environment(config)
    if config.get("original_environment"):
        env.pop("R_LIBS_USER",None)
    with (output/f"{label}_execution.log").open("w",encoding="utf-8") as log:
        subprocess.run([config["rscript"], "--vanilla", str(repo/script), str(job)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def source_inventory(config, repo):
    source = Path(config["source_run"])
    manifest = json.loads((source/"run_manifest.json").read_text(encoding="utf-8"))
    records = json.loads((source/"output_hashes.json").read_text(encoding="utf-8"))
    for r in records:
        if file_sha256(source/r["path"]) != r["sha256"]:
            raise ValueError(f"Historical output changed: {r['path']}")
    method_checks=[]
    for relative in METHOD_FILES:
        current, previous = repo/relative, source/"code_snapshot"/relative
        before = previous.read_text(encoding="utf-8").rstrip()
        after = current.read_text(encoding="utf-8").rstrip()
        if relative == "analysis/r/common.R":
            after = after.split("read_teacher_csv <- function(path)")[0].rstrip()
        if before != after:
            raise ValueError(f"Inference code differs: {relative}")
        method_checks.append({"path":relative,"sha256":file_sha256(current),"same_statistical_source":True})
    cache = manifest["psd_cache"]
    verify_cache(cache["cache_dir"],cache["cache_key"])
    protected = {Path(config[k]) for k in ("request_md","questionnaire_file","historical_package")}
    for r in cache["sources"]:
        for kind in ("set","fdt"):
            p = Path(r[f"{kind}_path"])
            if file_sha256(p) != r[f"{kind}_sha256"]:
                raise ValueError("Waveform source changed")
            protected.add(p)
    protected.update(Path(cache["cache_dir"]).glob("*"))
    protected.update(source/r["path"] for r in records)
    bootstrap=Path(config["outputs_root"])/"12_teacher_analysis/06_eeg_primary"
    protected.update(bootstrap/name for name in ("07_eeg_cluster_bootstrap_5000.csv","08_eeg_bootstrap_failures.csv","eeg_primary_analysis_reuse.json"))
    hashes=[{"path":str(p),"sha256":file_sha256(p),"size_bytes":p.stat().st_size} for p in sorted(protected) if p.is_file()]
    return manifest, hashes, method_checks


def python_spectra(cache):
    powers = pd.read_csv(Path(cache["cache_dir"])/"paired_powers.csv",encoding="utf-8-sig")
    unique_keys(powers, KEY+["onset_trim_s","roi"])
    rows=[]
    for index, source in enumerate(cache["sources"],1):
        d=loadmat(Path(cache["cache_dir"])/f"spectra_{index:03d}.mat",squeeze_me=True,struct_as_record=False)
        if str(d["metadata"].source.participant) != source["participant"]:
            raise ValueError("MAT participant mismatch")
        spectra=np.asarray(d["spectra"],dtype=object)
        for r,roi in enumerate(("F","P","O")):
            sub=powers.loc[powers.Participant.eq(source["participant"]) & powers.roi.eq(roi)]
            if spectra.shape[0] != len(sub):
                raise ValueError("Spectrum trial count mismatch")
            for j,(_,record) in enumerate(sub.iterrows()):
                s=spectra[j,r]
                if s.first_sample != record.first_sample or s.last_sample != record.last_sample:
                    raise ValueError("MAT/CSV endpoint mismatch")
                if s.window!=record.window or s.overlap!=record.overlap or s.nfft!=record.nfft:
                    raise ValueError("Welch parameters differ")
                expected_window=min(int(s.finite_samples),max(int(np.floor(2*record.srate+.5)),8))
                if s.window!=expected_window or s.overlap!=int(expected_window//2) or s.nfft!=2**int(np.ceil(np.log2(expected_window))):
                    raise ValueError("Welch parameter rule differs")
                if not set(str(x).upper() for x in np.atleast_1d(s.channels)) <= set({"F":["F3","F4"],"P":["P3","PZ","P4"],"O":["O1","OZ","O2"]}[roi]):
                    raise ValueError("ROI channel identity differs")
                f,pxx=np.asarray(s.f).ravel(),np.asarray(s.pxx).ravel()
                if not np.isfinite(pxx).all() or np.any(pxx<0) or not np.all(np.diff(f)>0):
                    raise ValueError("Invalid frequency/power array")
                row={k:record[k] for k in KEY+["onset_trim_s","roi"]}
                for name,(low,high) in BANDS.items():
                    mask=(f>=low)&(f<=high)
                    row[name]=float(np.trapz(pxx[mask],f[mask]))
                rows.append(row)
    return pd.DataFrame(rows)


def verify(config, config_path, repo):
    preflight(config,repo)
    if subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip():
        raise ValueError("Real request execution requires a committed clean repository")
    root=Path(config["outputs_root"])/"teacher_request_runs"/config["run_id"]
    root.mkdir(parents=True,exist_ok=False)
    dump(root/"request_config.json",config)
    manifest,protected,methods=source_inventory(config,repo)
    dump(root/"input_hashes_before.json",protected)
    source=Path(config["source_run"])
    reads=[]
    for version in "ABC":
        for trim in (0,5,10,15):
            reads.append({"id":f"{version}_{trim}","path":str(source/version/f"trim_{trim}s/input.csv"),"keys":KEY})
        for kind in ("family_coefficients","family_factors","contrast_matrices"):
            reads.append({"id":f"{version}_{kind}","path":str(source/version/f"{kind}.csv"),"keys":None})
    q=pd.read_excel(config["questionnaire_file"],sheet_name=0,keep_default_na=False)
    wanted=[next(c for c in q if str(c).startswith(prefix) and not str(c).endswith("_word")) for prefix in ("Q1.0_","Q1.4_","Q1.5_")]
    questionnaire=q[wanted].astype(str)
    payload={"reads":reads,"questionnaire_file":config["questionnaire_file"],"questionnaire_columns":wanted,
             "source_run":str(source),"cache":manifest["psd_cache"],"output":str(root)}
    r_call(config,repo,Path("analysis/r/eeg_request_read_verify.R"),payload,root,"read")
    checks=[]
    for record in reads:
        left,right=raw_csv(record["path"]),raw_csv(root/"r_reads"/(record["id"]+".csv"))
        count=compare_cells(left,right,keys=record["keys"])
        checks.append({"file":record["path"],"rows":len(left),"columns":len(left.columns),"cells":count,"exact":True})
    compare_cells(questionnaire,raw_csv(root/"r_questionnaire.csv"))
    qrows=[]
    for _,row in questionnaire.iterrows():
        answer=row[wanted[1]]
        group="Low" if any(x in answer for x in ("从不","极少","偶尔")) else "High" if any(x in answer for x in ("有时","经常")) else "Unknown"
        qrows.append({"Participant":questionnaire_identity(row[wanted[0]]),"Q1.4":answer,"ExperienceGroup":group})
    qgroups=pd.DataFrame(qrows)
    observed=pd.read_csv(source/"A/trim_0s/input.csv")[KEY+["ExperienceGroup"]].drop_duplicates("Participant")
    relevant=qgroups.loc[qgroups.Participant.isin(observed.Participant)]
    unique_keys(relevant,["Participant"])
    groupcheck=observed.merge(relevant,on="Participant",validate="one_to_one",how="left",suffixes=("_frozen","_Q1_4"))
    if groupcheck["ExperienceGroup_Q1_4"].isna().any() or not groupcheck.ExperienceGroup_frozen.equals(groupcheck.ExperienceGroup_Q1_4):
        groupcheck.to_csv(root/"Q1_4_discrepancies.csv",index=False,encoding="utf-8-sig")
        raise ValueError("Q1.4 differs from frozen groups; inputs preserved")
    groupcheck.to_csv(root/"Q1_4_group_check.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"raw_questionnaire_name":questionnaire[wanted[0]],
                  "Participant":questionnaire[wanted[0]].map(questionnaire_identity)}).to_csv(root/"questionnaire_name_mapping.csv",index=False,encoding="utf-8-sig")
    rgroup=raw_csv(root/"r_Q1_4_groups.csv")
    compare_cells(relevant.reset_index(drop=True),rgroup.loc[rgroup.Participant.isin(observed.Participant)].reset_index(drop=True),keys=["Participant"])
    python=python_spectra(manifest["psd_cache"])
    python.to_csv(root/"python_psd_integrals.csv",index=False,encoding="utf-8-sig")
    rpower=pd.read_csv(root/"r_psd_integrals.csv")
    compare_cells(python,rpower,keys=KEY+["onset_trim_s","roi"],numeric=True)
    original=pd.read_csv(source/"paired_psd_integrals.csv")
    compare_cells(python,original[python.columns],keys=KEY+["onset_trim_s","roi"],numeric=True)
    sample=[]; bh=[]
    waveform_lengths={s["participant"]:set_sample_count(s["set_path"]) for s in manifest["psd_cache"]["sources"]}
    frozen_keys=None
    for trim in (0,5,10,15):
        frames={v:pd.read_csv(source/v/f"trim_{trim}s/input.csv") for v in "ABC"}
        a,b,c=(frames[v] for v in "ABC")
        for v,frame in frames.items():
            unique_keys(frame,KEY)
            keys=set(map(tuple,frame[KEY].to_numpy()))
            if len(frame)!=461 or frame.Participant.nunique()!=42 or (frozen_keys is not None and keys!=frozen_keys):
                raise ValueError("Frozen common sample differs")
            frozen_keys=keys
            if not np.allclose(frame.analysis_start_s,frame.view_start_s+trim,atol=1e-12):
                raise ValueError("Onset trim boundary differs")
            sample.append({"version":v,"window_s":trim,"participants":42,"trials":461,"common_keys":True})
        relative=[f"{r}_{band}_relative" for r in "FPO" for band in ("theta","alpha","beta")]
        compare_cells(b.drop(columns=relative),c.drop(columns=relative),keys=KEY,numeric=True)
        changing={f"{r}_{band}{suffix}" for r in "FPO" for band in ("theta","alpha","beta") for suffix in ("","_absolute","_relative")}
        changing.update(f"log10_{r}_{band}_absolute" for r in "FPO" for band in ("theta","alpha","beta"))
        compare_cells(a.drop(columns=list(changing)),b.drop(columns=list(changing)),keys=KEY,numeric=True)
        for v,denominator in (("B","total_1_45"),("C","total_1_40")):
            frame=frames[v].set_index(KEY)
            for roi in "FPO":
                ps=python.loc[python.roi.eq(roi)&python.onset_trim_s.eq(trim)].set_index(KEY).reindex(frame.index)
                saved=original.loc[original.roi.eq(roi)&original.onset_trim_s.eq(trim)].set_index(KEY).reindex(frame.index)
                first=np.maximum(1,np.floor(frame.view_start_s*saved.srate+.5))+np.floor(trim*saved.srate+.5)
                last=np.minimum(frame.reset_index().Participant.map(waveform_lengths).to_numpy(),np.floor(frame.view_end_s*saved.srate+.5))
                if not np.array_equal(first,saved.first_sample) or not np.array_equal(last,saved.last_sample):
                    raise ValueError("Inclusive onset/epoch endpoints differ")
                for band in ("theta","alpha","beta"):
                    if not np.allclose(frame[f"{roi}_{band}_relative"],ps[band]/ps[denominator],rtol=1e-10,atol=1e-12):
                        raise ValueError("Relative power denominator/numerator mismatch")
    for v in "ABC":
        for kind in ("coefficients","factors"):
            frame=pd.read_csv(source/v/f"family_{kind}.csv")
            for record in bh_checks(frame):
                record.update(version=v,test_kind=kind);bh.append(record)
            rbh=pd.read_csv(root/"r_bh"/f"{v}_{kind}.csv")
            cols=TEST_KEY+["within_q","joint_q"]
            compare_cells(frame[cols],rbh[cols],keys=TEST_KEY,numeric=True)
            if kind=="coefficients":
                p=2*student_t.sf(np.abs(frame.estimate/frame["std.error"]),frame.df)
            else:
                p=fisher_f.sf(frame.Fstat,frame.df_num,frame.df_denom)
            if not np.allclose(p,frame["p.value"],rtol=1e-8,atol=1e-12):
                raise ValueError("Test statistic/p mismatch")
    focus=pd.read_csv(source/"manuscript_comparisons.csv")
    comparisons=pd.read_csv(source/"model_comparisons.csv")
    flips=comparisons.loc[comparisons.pair.eq("B→C")&(comparisons.flip_joint_q.eq(True)|comparisons.direction_changed.eq(True))]
    requests=pd.concat([focus[["onset_trim_s","outcome","model"]],flips[["onset_trim_s","outcome","model"]]]).drop_duplicates()
    requests.to_csv(root/"cited_models.csv",index=False)
    r_call(config,repo,Path("analysis/r/eeg_request_model_verify.R"),
           {"source_run":str(source),"requests":str(root/"cited_models.csv"),"output":str(root),"design":str(source/"analysis_contract.json")},root,"models")
    coef=pd.read_csv(root/"r_refit_coefficients.csv"); factors=pd.read_csv(root/"r_refit_factors.csv")
    regress=[]
    for version in "ABC":
        for kind,new in (("coefficients",coef),("factor_tests",factors)):
            old=pd.read_csv(source/version/f"{kind}.csv")
            keys=["onset_trim_s","outcome","model","term"]
            check=new.loc[new.version.eq(version)].merge(old,on=keys,suffixes=("_refit","_source"),validate="one_to_one")
            if len(check)!=len(new.loc[new.version.eq(version)]):
                raise ValueError("Refit identity mismatch")
            for metric,tolerance in (("estimate",1e-8),("p.value",1e-6)):
                delta=(check[f"{metric}_refit"]-check[f"{metric}_source"]).abs()
                error=float(delta.max()) if delta.notna().any() else 0.0
                regress.append({"version":version,"kind":kind,"metric":metric,"rows":len(check),"max_abs_error":error,"tolerance":tolerance,"passed":bool(error<=tolerance)})
    pd.DataFrame(regress).to_csv(root/"R45_model_regression.csv",index=False)
    environment_difference=not all(r["passed"] for r in regress)
    model_output=root
    if environment_difference:
        original=root/"original_R_refit";original.mkdir()
        replay={**config,"rscript":config["replay_rscript"],"original_environment":True}
        r_call(replay,repo,Path("analysis/r/eeg_request_model_verify.R"),
               {"source_run":str(source),"requests":str(root/"cited_models.csv"),"output":str(original),"design":str(source/"analysis_contract.json")},original,"models")
        model_output=original
        coef=pd.read_csv(original/"r_refit_coefficients.csv");factors=pd.read_csv(original/"r_refit_factors.csv")
        regress=[]
        for version in "ABC":
            for kind,new in (("coefficients",coef),("factor_tests",factors)):
                old=pd.read_csv(source/version/f"{kind}.csv")
                keys=["onset_trim_s","outcome","model","term"]
                check=new.loc[new.version.eq(version)].merge(old,on=keys,suffixes=("_refit","_source"),validate="one_to_one")
                if len(check)!=len(new.loc[new.version.eq(version)]):raise ValueError("Original-R refit identity mismatch")
                for metric,tolerance in (("estimate",1e-8),("p.value",1e-6)):
                    delta=(check[f"{metric}_refit"]-check[f"{metric}_source"]).abs()
                    error=float(delta.max()) if delta.notna().any() else 0.0
                    regress.append({"version":version,"kind":kind,"metric":metric,"rows":len(check),"max_abs_error":error,"tolerance":tolerance,"passed":bool(error<=tolerance)})
    pd.DataFrame(regress).to_csv(root/"R_model_regression.csv",index=False)
    if not all(r["passed"] for r in regress):
        raise ValueError("Refit still differs beyond tolerance in original environment; stop for investigation")
    matrices=pd.read_csv(model_output/"r_refit_matrices.csv")
    design=pd.read_csv(model_output/"r_refit_design.csv")
    design_keys=["version","onset_trim_s","outcome","model"]
    design_checks=[]
    for identity,group in design.groupby(design_keys,sort=False):
        version,trim,outcome,model=identity
        frame=pd.read_csv(source/version/f"trim_{trim}s/input.csv")
        if model=="PreviousScene":frame=frame.loc[frame.PositionWithinBlock.gt(1)]
        if model=="Block1":frame=frame.loc[frame.Block.eq(1)]
        group=group.sort_values(KEY).reset_index(drop=True);frame=frame.sort_values(KEY).reset_index(drop=True)
        if list(map(tuple,group[KEY].to_numpy()))!=list(map(tuple,frame[KEY].to_numpy())):
            raise ValueError("Independent model sample differs")
        checked=0
        for column in group.columns:
            if column in design_keys+KEY or group[column].isna().all():continue
            expected=independent_design_column(frame,column)
            if not np.allclose(expected,group[column],rtol=0,atol=1e-12):raise ValueError("Independent model design differs")
            checked+=len(group)
        design_checks.append({"version":version,"onset_trim_s":trim,"outcome":outcome,"model":model,"design_cells":checked,"exact":True})
    pd.DataFrame(design_checks).to_csv(root/"design_matrix_checks.csv",index=False)
    for version in "ABC":
        old=pd.read_csv(source/version/"contrast_matrices.csv")
        new=matrices.loc[matrices.version.eq(version)]
        matrix_keys=["onset_trim_s","outcome","term","contrast_row","coefficient"]
        check=new.merge(old,on=matrix_keys,suffixes=("_independent","_source"),validate="one_to_one")
        if len(check)!=len(new) or not np.allclose(check.weight_independent,check.weight_source,atol=1e-12,rtol=1e-10):
            raise ValueError("Independent equal-weight contrast matrix differs")
    # Independent comparison reconstruction from authoritative version tables.
    comparison_checks=[]
    for pair in ("A→B","B→C","A→C"):
        x,y=pair.split("→")
        expected=comparisons.loc[comparisons.pair.eq(pair)]
        for kind,filename in (("coefficient","family_coefficients"),("factor","family_factors"),("coefficient_unadjusted","all_model_coefficients")):
            left=pd.read_csv(source/x/f"{filename}.csv")
            right=pd.read_csv(source/y/f"{filename}.csv")
            merge=left.merge(right,on=TEST_KEY,suffixes=("_from","_to"),validate="one_to_one")
            selected=expected.loc[expected.test_kind.eq(kind)]
            for stat in ("estimate","p.value","within_q","joint_q"):
                columns=TEST_KEY+[f"{stat}_from",f"{stat}_to"]
                compare_cells(merge[columns],selected[columns],keys=TEST_KEY,numeric=True)
            comparison_checks.append({"pair":pair,"kind":kind,"rows":len(selected),"paired":True})
    history=preprocessing_records(manifest["psd_cache"])
    pd.DataFrame(history).to_csv(root/"preprocessing_history_check.csv",index=False,encoding="utf-8-sig")
    bootstrap=Path(config["outputs_root"])/"12_teacher_analysis/06_eeg_primary"
    boot=pd.read_csv(bootstrap/"07_eeg_cluster_bootstrap_5000.csv")
    proof=json.loads((bootstrap/"eeg_primary_analysis_reuse.json").read_text(encoding="utf-8"))
    script=(repo/"analysis/r/eeg_primary_analysis.R").read_text(encoding="utf-8")
    if "sample(participants, length(participants), replace = TRUE)" not in script or boot.successful.lt(1).any():
        raise ValueError("Bootstrap provenance invalid")
    dump(root/"bootstrap_provenance.json",{"unit":"Participant","rerun":False,"original_reuse_proof":proof,
        "minimum_successful":int(boot.successful.min()),"maximum_failed_replicates":int(boot.failed_replicates.max()),
        "statistical_script_sha256":file_sha256(repo/"analysis/r/eeg_primary_analysis.R")})
    for record in protected:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("Protected source changed during verification")
    dump(root/"input_hashes_after.json",protected)
    pd.DataFrame(checks).to_csv(root/"read_cell_checks.csv",index=False)
    pd.DataFrame(bh).to_csv(root/"BH_checks.csv",index=False)
    pd.DataFrame(sample).to_csv(root/"sample_checks.csv",index=False)
    dump(root/"comparison_checks.json",comparison_checks)
    summary={"status":"verified","git_sha":git_commit(repo),"config_sha256":file_sha256(config_path),
             "source_analysis_git_sha":manifest["git_sha"],"source_run":str(source),"input_files_verified":len(protected),
             "source_files_unchanged":True,"python_r_cells_checked":sum(r["cells"] for r in checks)+int(questionnaire.size),
             "spectra_checked":len(python),"R_refit_models":int(coef[["version","onset_trim_s","outcome","model"]].drop_duplicates().shape[0]),
             "R_model_regression":regress,"Q1_4_participants_checked":len(groupcheck),"methods":methods,
             "R45_environment_difference":environment_difference,"model_verification_environment":json.loads((model_output/"model_R_environment.json").read_text()),
             "python_version":sys.version,"r_runtime":json.loads((root/"R_environment.json").read_text()),
             "reuse":"complete PSD and verified prior outputs; no new filter, ICA, QC or bootstrap"}
    dump(root/"verification_summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("status","python_r_cells_checked","spectra_checked","R_refit_models")}),flush=True)
    return root


def request_root(config):
    return Path(config["outputs_root"])/"teacher_request_runs"/config["run_id"]


def statistical_verifier_source(source):
    tree=ast.parse(source)
    names={"raw_csv","unique_keys","compare_cells","bh_checks","independent_design_column","questionnaire_identity","set_sample_count","r_environment","r_call","source_inventory","python_spectra","verify"}
    selected={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    verifier=selected["verify"]
    start=verifier.index("    history=");end=verifier.index("    bootstrap=",start)
    selected["verify"]=verifier[:start]+"    # preprocessing evidence stage\n"+verifier[end:]
    if set(selected)!=names:raise ValueError("Missing statistical verifier function")
    return selected


def refresh_evidence(config,repo):
    """Refresh processing metadata only, with exact statistical-code reuse guards."""
    root=request_root(config)
    previous=json.loads((root/"verification_summary.json").read_text())
    if previous["status"]!="verified" or json.loads((root/"request_config.json").read_text())!=config:
        raise ValueError("Only a verified unchanged request can reuse statistical checks")
    if subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip():raise ValueError("Clean committed repository required")
    path="src/paper_analysis/teacher/request_handoff.py"
    old=subprocess.check_output(["git","show",f"{previous['git_sha']}:{path}"],cwd=repo).decode("utf-8")
    if statistical_verifier_source(old)!=statistical_verifier_source((repo/path).read_text(encoding="utf-8")):
        raise ValueError("Statistical verification code changed; full affected-stage verification required")
    for name in ("analysis/r/eeg_request_read_verify.R","analysis/r/eeg_request_model_verify.R"):
        before=subprocess.check_output(["git","show",f"{previous['git_sha']}:{name}"],cwd=repo).decode("utf-8")
        if before.rstrip()!=(repo/name).read_text(encoding="utf-8").rstrip():raise ValueError("Independent R verification code changed")
    manifest,protected,methods=source_inventory(config,repo)
    if protected!=json.loads((root/"input_hashes_after.json").read_text()):raise ValueError("Verified source hashes changed")
    outputs=[{"path":str(p),"sha256":file_sha256(p)} for p in sorted(root.rglob("*")) if p.is_file() and (p.suffix==".csv" or p.name in {"R_environment.json","model_R_environment.json"}) and not p.name.startswith("preprocessing_history") and p.name not in {"key_numbers.csv","paragraph_source_map.csv"} and "handoff_staging" not in p.parts]
    dump(root/"verification_before_evidence_refresh.json",previous)
    shutil.copyfile(root/"preprocessing_history_check.csv",root/"preprocessing_history_before_refresh.csv")
    pd.DataFrame(preprocessing_records(manifest["psd_cache"])).to_csv(root/"preprocessing_history_check.csv",index=False,encoding="utf-8-sig")
    for item in outputs:
        if file_sha256(item["path"])!=item["sha256"]:raise ValueError("Reused verification output changed")
    for item in protected:
        if file_sha256(item["path"])!=item["sha256"]:raise ValueError("Protected source changed")
    previous.update(statistics_verification_git_sha=previous.get("statistics_verification_git_sha",previous["git_sha"]),git_sha=git_commit(repo),methods=methods,evidence_refresh="preprocessing metadata only; independently verified statistical outputs reused")
    dump(root/"verification_output_reuse_hashes.json",outputs)
    dump(root/"verification_summary.json",previous)
    print(f"Processing metadata refreshed; {previous['R_refit_models']} independently verified model fits retained",flush=True)


def require_verified(config):
    root=request_root(config)
    summary=json.loads((root/"verification_summary.json").read_text())
    if summary["status"]!="verified":
        raise ValueError("Request verification has not passed")
    if json.loads((root/"request_config.json").read_text())!=config:
        raise ValueError("Request configuration changed after verification")
    repo=Path(__file__).resolve().parents[3]
    if summary["git_sha"]!=git_commit(repo):
        raise ValueError("Build/publication must use the verified committed version")
    return root,summary


def typed(name,frame):
    return {"name":name,"columns":frame.columns.tolist(),
            "rows":frame.astype(object).where(pd.notna(frame),None).values.tolist()}


def prepare_handoff(config,repo):
    root,verification=require_verified(config)
    source=Path(config["source_run"])
    stage=root/"handoff_staging"
    stage.mkdir(exist_ok=False)
    flat=stage/"filesource_flat";flat.mkdir()
    items=[]
    def add(path,category):
        path=Path(path)
        if path.is_file() and not any(r["source_path"]==str(path) for r in items):
            number=f"SRC{len(items)+1:04d}"
            name=f"{number}__{category}__{path.name}"
            target=flat/name
            shutil.copyfile(path,target)
            assert file_sha256(path)==file_sha256(target)
            items.append({"SourceID":number,"source_path":str(path),"flat_name":name,
                          "category":category,"size_bytes":path.stat().st_size,"sha256":file_sha256(path)})
    for version in "ABC":
        for trim in (0,5,10,15): add(source/version/f"trim_{trim}s/input.csv",f"{version}_{trim}s")
        for name in ("coefficients","factor_tests","contrast_matrices","diagnostics","family_coefficients","family_factors","family_status"):
            add(source/version/f"{name}.csv",version)
    for name in ("analysis_contract.json","summary.json","run_manifest.json","paired_psd_integrals.csv",
                 "power_fraction_summary.csv","model_comparisons.csv","manuscript_comparisons.csv","manuscript_conclusion_summary.csv",
                 "comparison_summary.csv","input_reproduction_differences.csv","historical_coefficient_regression.csv",
                 "factor_independent_regression.csv","reuse_verification.json","psd_cache_reference.json"):
        add(source/name,"analysis")
    for name in ("verification_summary.json","R_environment.json","R_session_info.txt","input_hashes_before.json","input_hashes_after.json",
                 "read_cell_checks.csv","Q1_4_group_check.csv","questionnaire_name_mapping.csv","preprocessing_history_check.csv","sample_checks.csv","BH_checks.csv",
                 "python_psd_integrals.csv","r_psd_integrals.csv","r_refit_coefficients.csv","r_refit_factors.csv","r_refit_matrices.csv",
                 "r_refit_diagnostics.csv","R_model_regression.csv","R45_model_regression.csv","model_R_environment.json","design_matrix_checks.csv","comparison_checks.json","cited_models.csv"):
        add(root/name,"verification")
    add(root/"bootstrap_provenance.json","verification")
    for name in ("verification_before_evidence_refresh.json","verification_output_reuse_hashes.json"):
        if (root/name).is_file():add(root/name,"verification_provenance")
    if (root/"original_R_refit").exists():
        for name in ("r_refit_coefficients.csv","r_refit_factors.csv","r_refit_matrices.csv","r_refit_diagnostics.csv","model_R_environment.json"):
            add(root/"original_R_refit"/name,"original_R_verification")
    add(config["questionnaire_file"],"questionnaire_source")
    add(config["request_md"],"teacher_request")
    bootroot=Path(config["outputs_root"])/"12_teacher_analysis/06_eeg_primary"
    for name in ("07_eeg_cluster_bootstrap_5000.csv","08_eeg_bootstrap_failures.csv"):
        add(bootroot/name,"bootstrap_provenance")
    dump(root/"source_manifest.json",items)
    def sid(path):
        return next(r["SourceID"] for r in items if r["source_path"]==str(Path(path)))
    def refs(*paths): return "、".join(sid(p) for p in paths)
    report=[];mapping=[];numbers=[]
    def line(text="",paths=(),fields="",filters="",quantitative=False):
        ids=refs(*paths) if paths else ""
        annotation=f"〔{ids}〕" if ids else ""
        report.append(text[:-1]+annotation+"|" if ids and text.startswith("|") and text.endswith("|") else text+annotation)
        if text and not text.startswith("#") and not (text.startswith("|") and "---" in text):
            mapping.append({"MD行号":len(report),"段落或表格行":text,"Source ID":ids or "综合归纳",
                            "工作表或字段":fields,"筛选条件":filters,"口径":"本次冻结EEG核查" if paths else "由前述数据综合归纳",
                            "定量":quantitative})
    summary=json.loads((source/"summary.json").read_text())
    history=pd.read_csv(root/"preprocessing_history_check.csv")
    order="；".join(f"{k}：{v}人" for k,v in history.order.value_counts().items())
    all_filters=bool(history.bandpass.all() and history.notch.all())
    if not all_filters or not history.average_reference.all() or not history.ICA_run.all():
        raise ValueError("Preprocessing evidence incomplete; do not publish unverified processing claims")
    line("# EEG Hz参数核查与1–40 Hz敏感性分析")
    line()
    line(f"本次日期：{config['stamp']}。核验执行代码：{verification['git_sha']}；复用的敏感性计算代码：{verification['source_analysis_git_sha']}。",
         [root/"verification_summary.json"],"git_sha/source_analysis_git_sha")
    if "statistics_verification_git_sha" in verification:
        line(f"统计双路核验代码：{verification['statistics_verification_git_sha']}；后续只更新处理历史解析和交付展示。统计代码和R核验代码逐项一致，源文件与已核验输出哈希重新检查后复用。",
             [root/"verification_summary.json",root/"verification_output_reuse_hashes.json"],"statistics_verification_git_sha/evidence_refresh")
    line("本次合并尚未发送的分母敏感性结果与老师新发的EEG核查要求。历史1–45 Hz主结果暂时保留，1–40 Hz作为敏感性分析；最终Methods与独立最终审计版本尚未冻结。")
    line(f"主要正文结论{'发生变化' if summary['major_manuscript_conclusions_changed'] else '保持一致'}；B→C有{summary['joint_q_flips_B_to_C']}项联合q越过0.05、{summary['direction_changes_B_to_C']}项系数方向变化，不能概括为全部统计结果完全不变。",
         [source/"summary.json",source/"model_comparisons.csv"],"major_manuscript_conclusions_changed/joint_q_flips_B_to_C/direction_changes_B_to_C","pair=B→C",True)
    line()
    line("## 处理参数与分析版本")
    line("|项目|核查结果|");line("|---|---|")
    facts=[("band-pass","0.5–40 Hz"),("notch","49–51 Hz"),("执行顺序",order),
           ("PSD输入","现有预处理.set/.fdt；已滤波、average reference及运行ICA。本次不重新滤波。"),
           ("ICA删除证据","历史中发现删除指令" if history.ICA_component_removal.any() else "未发现ICA成分删除指令；不能将运行ICA等同于完成成分剔除。"),
           ("版本","A历史1–45；B复现1–45；C与B使用同一PSD和频带分子，仅将分母改成1–40。"),
           ("样本","42人、461共同试次，0/5/10/15 s四窗口；QC名单、协变量、前序场景固定。"),
           ("Q1.4经验组","前两档Low、后两档High；附件ExerciseFrequency措辞在本代码中对应乒乓球经验，Q1.5不替代。"),
           ("ROI","F=F3/F4；P=P3/PZ/P4；O=O1/OZ/O2；先在时域平均，再计算Welch PSD。"),
           ("频带","θ 4–7、α 8–12、β 13–30 Hz；频率掩码及trapz积分保持原口径。")]
    for label,value in facts:
        paths=[root/"preprocessing_history_check.csv"] if label in {"band-pass","notch","执行顺序","PSD输入","ICA删除证据"} else [root/"Q1_4_group_check.csv"] if label=="Q1.4经验组" else [source/"analysis_contract.json",root/"sample_checks.csv"]
        line(f"|{label}|{value}|",paths,label,quantitative=True)
    line("B→C用于判断分母影响，A→B单独记录当前源文件与历史输入的复现差异，A→C用于核对论文。",
         [source/"input_reproduction_differences.csv",source/"model_comparisons.csv"],"pair")
    line("主模型为WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender + Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)。CR2按参与者聚类；六类factor-level采用等权边际CR2/HTZ。保留Model0、PreviousScene及Block1补充模型。",
         [source/"analysis_contract.json",root/"design_matrix_checks.csv",root/"r_refit_diagnostics.csv"],"formula/factor_levels/design/sample")
    line("四窗口联合BH族规模：系数级核心relative 144项，扩展relative/absolute各324项；factor-level核心96项、扩展relative/absolute各216项；temporal relative/absolute各72项；核心previous-scene 48项。各版本分开校正，窗口内与联合q均保留。",
         [root/"BH_checks.csv",source/"A/family_status.csv",source/"B/family_status.csv",source/"C/family_status.csv"],"family_id/scope/tests",quantitative=True)
    line("场景、事件边界、Round/Block、Position、OrderGroup及前序变量在A/B/C冻结输入中逐字段一致；Q1.0含中点的全名按仓库既有姓名规则匹配，原问卷姓名和Q1.4回答保持原样。此处核对冻结输入和PSD片段对应，不将其扩大描述为已完成附件阶段B的最终独立审计。",
         [root/"read_cell_checks.csv",root/"sample_checks.csv",root/"questionnaire_name_mapping.csv",source/"analysis_contract.json"],"model_input fields/epoch endpoints/name mapping")
    line()
    line("## 高频贡献与分母减少")
    line("|窗口s|40–45占1–45总功率均值%|实际分母减少均值%|relative增加均值%|");line("|---:|---:|---:|---:|")
    fractions=pd.read_csv(source/"power_fraction_summary.csv")
    for _,r in fractions.loc[fractions.roi.eq("ALL")].iterrows():
        line(f"|{int(r.onset_trim_s)}|{r.percent_40_45_mean:.6f}|{r.denominator_reduction_percent_mean:.6f}|{r.relative_increase_percent_mean:.6f}|",
             [source/"power_fraction_summary.csv",root/"python_psd_integrals.csv",root/"r_psd_integrals.csv"],"percent_40_45_mean/denominator_reduction_percent_mean/relative_increase_percent_mean",f"roi=ALL; onset_trim_s={int(r.onset_trim_s)}",True)
        for field in ("percent_40_45_mean","denominator_reduction_percent_mean","relative_increase_percent_mean"):
            numbers.append({"项目":field,"窗口s":int(r.onset_trim_s),"版本":"B/C","原值":float(r[field]),"Source ID":sid(source/"power_fraction_summary.csv"),"字段":field,"筛选":"roi=ALL"})
    line("40–45 Hz的独立积分与两分母之差不是同一数量：离散频率掩码边界间的梯形面积使两者不同。已逐片段积分，不使用统一百分比缩放。",
         [source/"paired_psd_integrals.csv",root/"python_psd_integrals.csv",root/"r_psd_integrals.csv"],"total_1_45/total_1_40/power_40_45")
    line()
    line("## Results 3.3与正文关键结果")
    line("下表为A→C正文对照；q为各自检验族的四窗口联合BH。多自由度总体检验报告F，不填造单一β。")
    line("|结果|窗口s|A β或F|C β或F|A p/q|C p/q|方向及联合显著性|");line("|---|---:|---:|---:|---|---|---|")
    focus=pd.read_csv(source/"manuscript_comparisons.csv")
    labels={"表6_额区theta_WWR45×复杂度":"表6额区θ交互","枕区theta_复杂度边际效应":"枕区θ复杂度边际效应","顶区beta_WWR总体效应":"顶区β的WWR总体效应"}
    selected=focus.loc[focus.pair.eq("A→C")&focus.manuscript_claim.isin(labels)]
    for _,r in selected.iterrows():
        b=r.estimate_from if pd.notna(r.estimate_from) else r.Fstat_from
        c=r.estimate_to if pd.notna(r.estimate_to) else r.Fstat_to
        changed=r.flip_joint_q==True or r.direction_changed==True
        line(f"|{labels[r.manuscript_claim]}|{int(r.onset_trim_s)}|{b:.7g}|{c:.7g}|{r['p.value_from']:.7g}/{r.joint_q_from:.7g}|{r['p.value_to']:.7g}/{r.joint_q_to:.7g}|{'改变' if changed else '未改变'}|",
             [source/"manuscript_comparisons.csv"],"estimate/Fstat/p.value/joint_q",f"pair=A→C; claim={r.manuscript_claim}; window={int(r.onset_trim_s)}",True)
        for version,suffix in (("A","from"),("C","to")):
            for field in ("estimate","Fstat","p.value","joint_q"):
                value=r.get(f"{field}_{suffix}")
                if pd.notna(value): numbers.append({"项目":labels[r.manuscript_claim],"窗口s":int(r.onset_trim_s),"版本":version,"原值":float(value),"Source ID":sid(source/"manuscript_comparisons.csv"),"字段":f"{field}_{suffix}","筛选":f"pair=A→C; claim={r.manuscript_claim}"})
    claims=pd.read_csv(source/"manuscript_conclusion_summary.csv")
    line("时序、枕区θ复杂度、顶区β WWR及上一场景结论按以下完整正文对照核查。")
    line("|正文项目|B→C判断|联合q翻转数|方向变化数|");line("|---|---|---:|---:|")
    for _,r in claims.loc[claims["比较"].eq("B→C")].iterrows():
        line(f"|{r['正文结论']}|{r['判断']}|{int(r['联合q翻转数'])}|{int(r['方向变化数'])}|",
             [source/"manuscript_conclusion_summary.csv",source/"manuscript_comparisons.csv"],"判断/联合q翻转数/方向变化数",f"比较=B→C; 正文结论={r['正文结论']}",True)
    line()
    line("## 边界变化与来源限制")
    comparisons=pd.read_csv(source/"model_comparisons.csv")
    changes=comparisons.loc[comparisons.pair.eq("B→C")&(comparisons.flip_joint_q.eq(True)|comparisons.direction_changed.eq(True))]
    for _,r in changes.iterrows():
        q=f"{r.joint_q_from:.9g}→{r.joint_q_to:.9g}" if pd.notna(r.joint_q_from) and pd.notna(r.joint_q_to) else "不适用（此补充系数未加入校正族）"
        line(f"{int(r.onset_trim_s)} s，{r.outcome}，{r.model}，{r.term}：β {r.estimate_from:.9g}→{r.estimate_to:.9g}，原始p {r['p.value_from']:.9g}→{r['p.value_to']:.9g}，联合q {q}。",
             [source/"model_comparisons.csv"],"estimate/p.value/joint_q",f"pair=B→C; window={r.onset_trim_s}; outcome={r.outcome}; model={r.model}; term={r.term}; family={r.family_id}",True)
    line("边界q翻转不能据此增加跨窗口稳定效应的表述；接近零且不显著的变号不构成新的实质方向证据。")
    line("factor-level采用等权边际CR2/HTZ重新实现并独立核对；历史原脚本仍未找到。PreviousScene保持PreviousWWR + PreviousComplexity，论文交互项描述与实际执行代码的差异继续记录。",
         [source/"analysis_contract.json",source/"factor_independent_regression.csv"],"factor_method/previous_formula")
    line()
    line("## Python与R复核结果")
    line(f"独立读取核对{verification['python_r_cells_checked']}个单元格、逐一核对{verification['spectra_checked']}个PSD片段，核对{verification['Q1_4_participants_checked']}人的Q1.4分组，并独立复拟合{verification['R_refit_models']}个正文或边界变化涉及的模型。所有已登记源文件在处理前后哈希一致。",
         [root/"verification_summary.json",root/"read_cell_checks.csv",root/"R_model_regression.csv"],"python_r_cells_checked/spectra_checked/Q1_4_participants_checked/R_refit_models",quantitative=True)
    line("Python负责源表读取、片段/样本/字段核对、PSD积分、功率与统计量关系、完整BH和结果配对；R分别直接读取源CSV/XLSX/MAT、独立积分和BH，并使用lme4/clubSandwich复拟合CR2/HTZ。两种语言未共用解析后的模型输入。",
         [root/"read_cell_checks.csv",root/"BH_checks.csv",root/"R_environment.json"],"reader/runtime")
    if verification["R45_environment_difference"]:
        line("系统R 4.5.3复拟合与历史环境存在超出预设容差的数值差异，已单列保留。使用原R环境独立复拟合后通过历史回归；正文对照仍采用已核验的原结果，不以更新软件环境重定义主分析。",
             [root/"R45_model_regression.csv",root/"R_model_regression.csv",root/"verification_summary.json"],"R45_environment_difference/model_verification_environment")
    line("原PSD、未涉及新增复拟合的模型和既有bootstrap通过来源核查后复用。bootstrap按参与者抽样；本次未重复运行抽样，也未把旧执行标成新计算。",
         [bootroot/"07_eeg_cluster_bootstrap_5000.csv",bootroot/"08_eeg_bootstrap_failures.csv",root/"verification_summary.json"],"bootstrap provenance")
    markdown="\n".join(report)+"\n"
    (root/"论文数据分析结果报告.md").write_text(markdown,encoding="utf-8")
    shutil.copyfile(root/"论文数据分析结果报告.md",stage/"论文数据分析结果报告.md")
    pd.DataFrame(mapping).to_csv(root/"paragraph_source_map.csv",index=False,encoding="utf-8-sig")
    number_table=pd.DataFrame(numbers)
    rfit_root=root/"original_R_refit" if verification["R45_environment_difference"] else root
    rcoeff=pd.read_csv(rfit_root/"r_refit_coefficients.csv");rfactors=pd.read_csv(rfit_root/"r_refit_factors.csv")
    pintegrals=pd.read_csv(root/"python_psd_integrals.csv");rintegrals=pd.read_csv(root/"r_psd_integrals.csv")
    for i,n in number_table.iterrows():
        if n["版本"]=="B/C":
            vals=[]
            for data in (pintegrals,rintegrals):
                g=data.loc[data.onset_trim_s.eq(n["窗口s"])]
                val=100*g.power_40_45/g.total_1_45 if n["字段"]=="percent_40_45_mean" else 100*(g.total_1_45-g.total_1_40)/g.total_1_45 if n["字段"]=="denominator_reduction_percent_mean" else 100*(g.total_1_45/g.total_1_40-1)
                vals.append(float(val.mean()))
            tolerance=1e-10*abs(n["原值"])+1e-12
        else:
            claim=next(k for k,v in labels.items() if v==n["项目"])
            frow=selected.loc[selected.manuscript_claim.eq(claim)&selected.onset_trim_s.eq(n["窗口s"])].iloc[0]
            keycols={"version":n["版本"],"onset_trim_s":n["窗口s"],"outcome":frow.outcome,"model":frow.model,"term":frow.term}
            field=n["字段"].rsplit("_",1)[0]
            ref=rfactors if frow.test_kind=="factor" else rcoeff
            mask=pd.Series(True,index=ref.index)
            for c,v in keycols.items():mask &= ref[c].eq(v)
            if field=="joint_q":
                kind="factors" if frow.test_kind=="factor" else "coefficients"
                table=pd.read_csv(source/n["版本"]/f"family_{kind}.csv")
                members=table.loc[table.family_id.eq(frow.family_id)].copy()
                members["recomputed_q"]=multipletests(members["p.value"],method="fdr_bh")[1]
                hit=members.loc[members.onset_trim_s.eq(n["窗口s"])&members.outcome.eq(frow.outcome)&members.model.eq(frow.model)&members.term.eq(frow.term)].iloc[0]
                rtable=pd.read_csv(root/"r_bh"/f"{n['版本']}_{kind}.csv")
                rhit=rtable.loc[rtable.onset_trim_s.eq(n["窗口s"])&rtable.outcome.eq(frow.outcome)&rtable.model.eq(frow.model)&rtable.term.eq(frow.term)&rtable.family_id.eq(frow.family_id)].iloc[0]
                vals=[float(hit.recomputed_q),float(rhit.joint_q)]
            else:
                if mask.sum()!=1:raise ValueError("Key numeric refit not uniquely identified")
                vals=[float(n["原值"]),float(ref.loc[mask,field].iloc[0])]
            tolerance=1e-8 if field=="estimate" else 1e-6 if field in {"p.value","joint_q"} else 1e-8*max(1,abs(n["原值"]))
        for column,value in zip(("Python独立复核值","R独立复核值"),vals):number_table.loc[i,column]=value
        error=max(abs(v-n["原值"]) for v in vals)
        number_table.loc[i,"最大绝对差"]=error;number_table.loc[i,"容差"]=tolerance;number_table.loc[i,"通过"]=bool(error<=tolerance)
    if not number_table["通过"].all():raise ValueError("A report key number failed independent verification")
    number_table.to_csv(root/"key_numbers.csv",index=False,encoding="utf-8-sig")
    filelist=pd.DataFrame([{"Source ID":r["SourceID"],"原始绝对路径":r["source_path"],"平铺文件名":r["flat_name"],"阶段":r["category"],"大小字节":r["size_bytes"],"SHA256":r["sha256"],"打开文件":f'=HYPERLINK("filesource_flat/{r["flat_name"]}","{r["SourceID"]}")'} for r in items])
    sample=pd.read_csv(root/"sample_checks.csv")
    environment=pd.DataFrame([("来源运行",str(source)),("本次核验运行",config["run_id"]),("核验代码SHA",verification["git_sha"]),("原分析代码SHA",verification["source_analysis_git_sha"]),("主结果", "暂保留历史1–45 Hz"),("敏感性","1–40 Hz；Methods尚未冻结"),("阅读顺序","先读MD，再按Source ID查索引和filesource_flat"),("参与者标识","来源文件按原样保留，个体资料按研究数据约定保存")],columns=["项目","说明"])
    powers=pd.read_csv(source/"paired_psd_integrals.csv")
    calculation=powers[KEY+["onset_trim_s","roi","total_1_45","total_1_40","power_40_45","theta","alpha","beta"]].copy()
    calculation["40–45占比%"]=[f"=100*G{i}/E{i}" for i in range(2,len(calculation)+2)]
    calculation["实际分母减少%"]=[f"=100*(E{i}-F{i})/E{i}" for i in range(2,len(calculation)+2)]
    calculation["relative增加%"]=[f"=100*(E{i}/F{i}-1)" for i in range(2,len(calculation)+2)]
    omitted=pd.DataFrame([{"资料":"预处理.set/.fdt与完整MAT PSD","未复制原因":"大型原始波形与PSD保留本地；原文件路径和SHA256见来源清单","来源入口":r["path"],"SHA256":r["sha256"],"大小字节":r["size_bytes"]} for r in json.loads((root/"input_hashes_before.json").read_text()) if Path(r["path"]).suffix.lower() in {".set",".fdt",".mat"}])
    glossary=pd.DataFrame([("Participant/GlobalTrialOrder","参与者与全局试次键，禁止依赖行号合并"),("WWR","15/45/75三水平分类变量"),("Complexity","C0/C1；历史空白C0仅按已核对条件恢复"),("ExperienceGroup","Q1.4乒乓球经验四档的前两档Low/后两档High"),("CR2","按参与者聚类的稳健协方差；系数用Satterthwaite"),("HTZ","因子等权边际Wald检验的自由度近似"),("within_q/joint_q","窗口内BH与四窗口联合BH；不同检验族单列"),("NA/空白","不适用或推断无效；多自由度检验无单一β，未校正项无q")],columns=["术语","说明"])
    tables=[typed("先看这里",environment),typed("段落来源映射",pd.DataFrame(mapping)),typed("关键数字核对",number_table),
            typed("文件总清单",filelist),typed("样本与数据粒度",sample),typed("功率与统计复核",calculation),
            typed("差异与阴性结果",changes[[c for c in ("onset_trim_s","outcome","model","term","family_id","estimate_from","estimate_to","p.value_from","p.value_to","within_q_from","within_q_to","joint_q_from","joint_q_to") if c in changes]]),
            typed("未复制原始数据",omitted),typed("术语与字段说明",glossary)]
    dump(root/"handoff_tables.json",{"tables":tables})
    guide=["# 本次EEG核查交接说明","",
           "本次材料合并此前尚未发送的1–40 Hz分母敏感性分析，以及新附件要求的Hz参数和分母无关核查。主要结论请先读论文数据分析结果报告.md。历史1–45 Hz主结果暂时保留，1–40 Hz是敏感性结果，最终Methods尚未冻结。","",
           "## 阅读与追溯","",
           "MD中的Source ID对应数据来源交接索引.xlsx。索引按段落、表格行及字段登记具体来源；文件总清单提供相对链接。打开链接即可读取filesource_flat中按原始字节复制的源表。解压ZIP后保持文件夹结构。","",
           "## 本次主要发现","",
           f"主要正文结论{'发生变化，需逐项查看' if summary['major_manuscript_conclusions_changed'] else '保持一致'}。全部配对检验中，有{summary['joint_q_flips_B_to_C']}项联合q跨越0.05、{summary['direction_changes_B_to_C']}项系数方向变化；是否接近零、是否显著及对应模型见MD。40–45 Hz独立积分占比与两分母减少比例不同，不能按统一比例缩放。","",
           "## 独立复核与统计口径","",
           f"Python与R分别读取源文件，核对{verification['python_r_cells_checked']}个单元格和{verification['spectra_checked']}个PSD片段；R独立复拟合{verification['R_refit_models']}个被正文或边界变化引用的模型。输入、分类、样本、频带积分、检验族和源文件哈希均核对。原准确结果与bootstrap按来源证明复用。","",
           "CR2按参与者聚类，单系数采用Satterthwaite推断；factor-level采用等权边际CR2/HTZ。窗口内与四窗口联合BH分别列出，多自由度总体检验用F，不提供虚构的单一β。正文显著性依据各自完整检验族的joint_q。","",
           "## 尚需明确的事项","",
           "factor-level是重新实现并独立核对的代码，历史原脚本尚未找到。上一场景沿用PreviousWWR + PreviousComplexity，论文交互项描述与执行结构的差异已记录。运行ICA不等于已完成成分删除；历史中未见删除指令。","",
           "本次仅包含老师当前要求。其他多模态结果仍在正式分析目录；旧交付已移至目录之外归档。个体级源文件包含参与者标识，按此前研究数据约定保存。"]
    (root/"handoff_guide.md").write_text("\n".join(guide)+"\n",encoding="utf-8")
    dump(root/"handoff_prepared.json",{"sources":len(items),"staging":str(stage),"report_sha256":file_sha256(stage/"论文数据分析结果报告.md")})
    print(f"prepared_sources {len(items)}",flush=True)
    return stage


def make_document(config,repo,maker):
    root,_=require_verified(config)
    maker((root/"handoff_guide.md").read_text(encoding="utf-8"),root/"handoff_staging/数据来源交接说明.docx")


def archive_publish(stage, destination, archive, validator=None):
    stage,destination,archive=map(lambda p:Path(p).resolve(),(stage,destination,archive))
    if destination==archive or destination in archive.parents or archive in destination.parents or stage==destination or destination in stage.parents:
        raise ValueError("Unsafe publication paths")
    archive.mkdir(parents=True,exist_ok=False)
    destination.mkdir(parents=True,exist_ok=True)
    old=list(destination.iterdir()); new=list(stage.iterdir())
    moved_old=[]; moved_new=[]
    try:
        for p in old:
            shutil.move(str(p),str(archive/p.name)); moved_old.append(p.name)
        for p in new:
            shutil.move(str(p),str(destination/p.name)); moved_new.append(p.name)
        if validator is not None:
            validator(destination, archive)
    except BaseException:
        for name in reversed(moved_new): shutil.move(str(destination/name),str(stage/name))
        for name in reversed(moved_old): shutil.move(str(archive/name),str(destination/name))
        raise
    return {str(destination/name):str(archive/name) for name in moved_old}


def publish_handoff(config,repo):
    root,verification=require_verified(config); stage=root/"handoff_staging"
    expected={"论文数据分析结果报告.md","数据来源交接索引.xlsx","数据来源交接说明.docx","filesource_flat"}
    if {p.name for p in stage.iterdir()} != expected:
        raise ValueError("Delivery staging must contain exactly the four specified entries")
    if not (root/"artifact_verification.json").is_file():
        raise ValueError("Rendered artifacts must be verified before publishing")
    qa=json.loads((root/"artifact_verification.json").read_text())
    for name in ("数据来源交接索引.xlsx","数据来源交接说明.docx"):
        if qa["files"][name]["sha256"]!=file_sha256(stage/name) or not qa["files"][name]["visually_reviewed"]:
            raise ValueError("Artifact verification missing or stale")
    manifest=json.loads((root/"source_manifest.json").read_text())
    for r in manifest:
        if file_sha256(stage/"filesource_flat"/r["flat_name"])!=r["sha256"] or file_sha256(r["source_path"])!=r["sha256"]:
            raise ValueError("Source/copy changed after preparation")
    if file_sha256(root/"论文数据分析结果报告.md")!=file_sha256(stage/"论文数据分析结果报告.md"):
        raise ValueError("Report copy differs")
    # Manifest resides with sources; no additional top-level reader clutter.
    files=[{"path":p.relative_to(stage).as_posix(),"sha256":file_sha256(p)} for p in sorted(stage.rglob("*")) if p.is_file()]
    dump(stage/"filesource_flat/交付文件哈希.json",files)
    archive_file=stage/f"结果发送_完整交接包_EEG_Hz核查_{config['stamp']}.zip"
    with zipfile.ZipFile(archive_file,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p!=archive_file:z.write(p,p.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive_file) as z:
        if z.testzip() is not None:raise ValueError("ZIP integrity failure")
        import hashlib
        for r in files:
            if hashlib.sha256(z.read(r["path"])).hexdigest()!=r["sha256"]:raise ValueError("ZIP hash mismatch")
    # Verify all old files have survived unchanged after migration.
    destination=Path(config["delivery_root"])
    old_hashes={str(p.relative_to(destination)):file_sha256(p) for p in destination.rglob("*") if p.is_file()}
    archive=Path(config["archive_root"])/config["run_id"]
    def validate_published(current, historical):
        if not all(file_sha256(historical/p)==sha for p,sha in old_hashes.items()):
            raise ValueError("Archived old delivery differs")
        if not all(file_sha256(current/r["path"])==r["sha256"] for r in files):
            raise ValueError("Published delivery differs")
    moved=archive_publish(stage,destination,archive,validate_published)
    dump(root/"archive_path_mapping.json",moved)
    pointer={"status":"complete","run_root":str(root),"source_run":config["source_run"],"delivery_root":str(destination),
             "zip":str(destination/archive_file.name),"zip_sha256":file_sha256(destination/archive_file.name),
             "archive_root":str(archive),"git_sha":verification["git_sha"],"source_analysis_git_sha":verification["source_analysis_git_sha"],
             "major_conclusions_changed":json.loads((Path(config["source_run"])/"summary.json").read_text())["major_manuscript_conclusions_changed"],
             "sources":len(manifest),"archived_files":len(old_hashes)}
    dump(root/"publication.json",pointer)
    outputs=Path(config["outputs_root"])
    dump(outputs/"eeg_request_handoff_latest.json",pointer)
    shutil.copyfile(root/"论文数据分析结果报告.md",outputs/"老师本次EEG核查报告.md")
    summary_path=outputs/"realdata_run_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"));summary["current_teacher_request"]=pointer
    if "teacher_delivery" in summary:
        previous=summary["teacher_delivery"]
        previous["archive_mapping"]=str(root/"archive_path_mapping.json")
        summary.setdefault("teacher_delivery_history",[]).append(previous)
    summary["teacher_delivery"]={**pointer,"package":str(destination),"publication_git_sha":verification["git_sha"]}
    dump(summary_path,summary)
    sensitivity_pointer=outputs/"eeg_denominator_sensitivity_latest.json"
    previous=json.loads(sensitivity_pointer.read_text(encoding="utf-8"))
    previous.update(delivery_root=str(destination),delivery_zip=pointer["zip"],request_verification_run=str(root))
    dump(sensitivity_pointer,previous)
    readme=outputs/"README_当前有效结果.md"
    readme.write_text("# 当前有效数据分析结果\n\n完整正式运行：teacher_latest_20261003；论文数据分析结果报告.md保留完整多模态内容。\n\n"
        "老师本次EEG核查阅读入口：老师本次EEG核查报告.md；逐项独立核验与交付入口：eeg_request_handoff_latest.json。\n\n"
        f"当前老师交付目录：{destination}；历史交付已在目录之外归档：{archive}。\n\n"
        "历史1–45 Hz主结果暂保留，1–40 Hz是敏感性分析，最终Methods未冻结。\n",encoding="utf-8")
    (outputs/"README_老师本次EEG交付.md").write_text(f"# 老师本次EEG交付\n\n阅读老师本次EEG核查报告.md；核验运行：{config['run_id']}。\n\n交付目录：{destination}。旧交付完整归档：{archive}。\n\n完整多模态报告仍为论文数据分析结果报告.md；历史1–45为主结果，1–40为敏感性，最终Methods未冻结。\n",encoding="utf-8")
    print(json.dumps(pointer,ensure_ascii=True),flush=True)
    return pointer
