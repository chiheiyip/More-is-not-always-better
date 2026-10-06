import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from paper_analysis.teacher.independent_eye import (
    RAW_COLUMNS, deduplicate_fixations, assign_events, trial_metrics, make_masks,
    read_questionnaire, unique, canonical_name,
)
from paper_analysis.teacher.independent_eeg import reconstruct_qc, read_raw_acquisition, match_scene_epochs, current_model_inputs
from paper_analysis.teacher.independent_eye import source_identity
from paper_analysis.teacher.incremental_audit import compare_values, r_env
from paper_analysis.teacher.incremental_evidence import pair_statistics, bh_checks


def test_independent_cli_does_not_import_original_eye_processing():
    repo = Path(__file__).resolve().parents[1]
    code = "import sys;sys.path.insert(0,'src');import paper_analysis.teacher.incremental_audit;assert 'paper_analysis.teacher.eye' not in sys.modules"
    result = subprocess.run([os.sys.executable,"-c",code],cwd=repo,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_fixations_deduplicate_duration_and_record_coordinate_conflict():
    raw = pd.DataFrame({c: [0, 0, 0] for c in RAW_COLUMNS})
    raw["Fixation Index"] = [1, 1, 2]
    raw["Fixation Point X[px]"] = [1, 1.5, 3]
    raw["Fixation Point Y[px]"] = [2, 2, 3]
    raw["Fixation Duration[ms]"] = [100, 100, 200]
    raw["Recording Time Stamp[ms]"] = [1000, 1004, 1300]
    events, issues = deduplicate_fixations(raw)
    assert len(events) == 2 and events.FixationDuration.sum() == 300
    assert events.FixationX.iloc[0] == 1.25 and issues.empty
    raw.loc[1, "Fixation Point X[px]"] = 10
    _, issues = deduplicate_fixations(raw)
    assert issues.Issue.tolist() == ["coordinate_conflict"]


def test_offstimulus_and_structural_missing_are_distinct():
    events = pd.DataFrame({"FixationX": [0, 1, -10], "FixationY": [0, 0, 0],
                           "FixationDuration": [100, 200, 300], "FixationStartMS": [0, 200, 300]})
    events = assign_events(events, np.array([[1, 4]], dtype=np.uint8))
    result = trial_metrics(events, {"TableAreaShare": .2, "WindowAreaShare": .3, "EquipmentAreaShare": np.nan, "BackgroundAreaShare": .5}, 1, .8, 0)
    assert result["ValidSceneTFD"] == 300
    assert result["TableShare"] == pytest.approx(1/3)
    assert np.isnan(result["EquipmentVisited"]) and np.isnan(result["EquipmentShare"])
    assert result["WindowVisited"] == 0 and np.isnan(result["WindowTTFF"])
    assert np.isnan(result["LogWindowEnrichment"])


def test_measurement_mask_size_overlap_and_background(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "base.png")
    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(tmp_path / "valid.png")
    annotation = {"image": {"width": 8, "height": 8}, "aoi_classes": {"table": [{"points": [[1,1],[3,1],[3,3],[1,3]]}], "window": [{"points": [[2,2],[4,2],[4,4],[2,4]]}]}}
    (tmp_path / "aoi.json").write_text(json.dumps(annotation))
    row = {"AOIFile": str(tmp_path / "aoi.json"), "BaseImageFile": str(tmp_path / "base.png"), "ValidSceneFile": str(tmp_path / "valid.png"), "Complexity": 0, "ProjectionType": "unknown", "OverlapResolution": "table_window_equipment_priority"}
    code, shares, detail = make_masks(row)
    assert detail["OverlapPixelsBeforeResolution"] == 4 and code[2,2] == 1
    assert sum(v for v in shares.values() if np.isfinite(v)) == pytest.approx(1)
    with pytest.raises(ValueError, match="verified equirectangular"):
        make_masks(row, spherical=True)


def test_q1_4_cannot_be_replaced_by_q1_5(tmp_path):
    # XLSX test fixture only; production artifact authoring uses Artifact Tool.
    p = tmp_path / "original.xlsx"
    pd.DataFrame({"Q1.0_姓名": ["甲", "乙·丙"], "Q1.4_运动": ["偶尔（每月1–2次）", "有时（每月3-4次）"], "Q1.5_经验": ["经常", "从不"], "Q1.4_运动_word": ["经常", "从不"]}).to_excel(p, index=False)
    q = read_questionnaire(p)
    assert q.Participant.tolist() == ["甲", "乙"] and q.ExerciseFrequency.tolist() == ["Low", "High"]
    assert canonical_name("卢诗晗-600-600") == "卢诗晗"
    assert canonical_name("甲-研究记录") == "甲-研究记录"


def test_comparison_matches_by_key_and_exposes_missingness():
    a = pd.DataFrame({"Participant": ["甲", "乙"], "GlobalTrialOrder": [1, 1], "x": [np.nan, .2]})
    _, summary = compare_values(a, a.iloc[::-1], ["x"])
    assert summary.MismatchedRows.iloc[0] == 0
    b = a.copy(); b.loc[0,"x"] = 0
    _, summary = compare_values(a, b, ["x"])
    assert summary.MismatchedRows.iloc[0] == 1
    with pytest.raises(ValueError): unique(pd.concat([a,a]), ["Participant","GlobalTrialOrder"])


def test_paired_missing_q_does_not_fabricate_a_threshold_crossing():
    a=pd.DataFrame({'key':[1,2],'estimate':[1,np.nan],'raw_p':[.0499,np.nan],'joint_q':[np.nan,.1]})
    b=pd.DataFrame({'key':[2,1],'estimate':[2,1],'raw_p':[.01,.0501],'joint_q':[np.nan,.02]})
    out=pair_statistics(a,b,['key'])
    assert out.raw_p_crosses_005.sum()==1 and not out.joint_q_crosses_005.any()
    assert not out.loc[out.key.eq(2),'DirectionChanged'].any()


def test_missing_current_family_requires_all_q_to_remain_unavailable():
    frame=pd.DataFrame({'version':['D']*3,'family_id':['test']*3,'onset_trim_s':[0]*3,'raw_p':[.01,.04,np.nan],'inference_valid':[True,True,False],'joint_q':[np.nan]*3,'within_q':[np.nan]*3})
    assert bh_checks(frame,'coefficient').FullFamilySize.iloc[0]==3
    frame.loc[0,'joint_q']=.02
    with pytest.raises(ValueError,match='joint BH'):bh_checks(frame,'coefficient')


def test_variant_qc_common_identity_and_subject_exclusion():
    data = []
    for trim in [0,5,10,15]:
        for p in ["a","b"]:
            for trial in range(1,5):
                bad = p == "b" and trim == 15 and trial <= 2
                data.append({"Participant": p, "GlobalTrialOrder": trial, "onset_trim_s": trim, "analysis_dur_s": 30,
                             "segment_valid_duration": True, "nan_fraction": .5 if bad else 0, "flat_fraction": 0,
                             "hf_ratio_20_40Hz": .1, "rms_mean_uV": 5, "peak_to_peak_uV": 10})
    cfg = {"min_segment_duration_s": 1, "nan_fraction_threshold": .2, "flat_fraction_threshold": .2,
           "robust_k": 3.5, "robust_min_n": 4, "bad_scene_fraction_threshold": .3}
    d, _ = reconstruct_qc(pd.DataFrame(data), cfg)
    assert d.loc[d.Participant.eq("a"), "CommonQCIncluded"].all()
    assert not d.loc[d.Participant.eq("b"), "CommonQCIncluded"].any()
    assert d.loc[d.Participant.eq("b") & d.onset_trim_s.eq(15), "eeg_subject_quality_exclusion"].all()


def test_original_easy_pulses_blank_rows_and_info_identity(tmp_path):
    p = tmp_path / "original.easy"
    rows = []
    for i, marker in enumerate([0,7,7,0,8,0]):
        rows.append("\t".join(map(str,[0]*11+[marker,1000+2*i])))
    p.write_text("\n\n".join(rows),encoding="utf-8")
    p.with_suffix(".info").write_text("Number of EEG channels: 8\nNumber of records of EEG: 6\nEEG sampling rate: 500 Samples/second\n")
    metadata, events = read_raw_acquisition({"candidate_easy_path": str(p),"participant_id":"甲"})
    assert metadata["Samples"] == metadata["InfoSamples"] == 6
    assert metadata["TimestampRegular"] and metadata["InfoEEGChannels"] == 8
    assert events.Marker.tolist() == ["7","8"] and events.LatencySample.tolist() == [2,5]


def test_generic_export_identity_remains_unverified_and_explicit():
    raw = pd.DataFrame({"User":["预实验"]})
    record = pd.Series({"Participant":"甲","CSVFile":"raw_甲_123.csv"})
    with pytest.raises(ValueError): source_identity(raw,record,{})
    result = source_identity(raw,record,{"unresolved_export_labels":{"甲":["预实验"]}})
    assert result[2].endswith("raw_identity_unverified") and result[1]=="预实验"
    record.CSVFile="raw_乙_123.csv"
    with pytest.raises(ValueError): source_identity(raw,record,{"unresolved_export_labels":{"甲":["预实验"]}})


def test_scene_epoch_matching_does_not_shift_after_extra_trigger():
    raw = pd.DataFrame({"Participant":["a"]*6,"Marker":["7","8","7","7","8","9"],"LatencySample":[10,20,25,30,40,50]})
    sets = pd.DataFrame({"Participant":["a"]*4,"Marker":["7","8","7","8"],"LatencySample":[10,20,30,40]})
    match = match_scene_epochs(raw,sets)
    assert len(match)==2 and match.EndpointMatchStatus.eq("both").all()


def test_prior_relocation_requires_exact_bytes(tmp_path):
    from paper_analysis.teacher.independent_eeg import verify_prior
    from paper_analysis.teacher.independent_eye import sha
    original=tmp_path/'expired.md';copy=tmp_path/'preserved.md';copy.write_text('original request')
    (tmp_path/'verification_summary.json').write_text(json.dumps({'status':'verified','source_files_unchanged':True,'methods':[]}))
    (tmp_path/'input_hashes_after.json').write_text(json.dumps([{'path':str(original),'sha256':sha(copy)}]));(tmp_path/'verification_output_reuse_hashes.json').write_text('[]')
    config={'prior_verification':str(tmp_path),'prior_source_relocations':{str(original):str(copy)}}
    _,checks=verify_prior(config,tmp_path);assert checks[0]['status']=='relocated_identical_bytes'
    copy.write_text('different');
    with pytest.raises(ValueError,match='changed'): verify_prior(config,tmp_path)


def test_current_qc_inputs_preserve_common_keys_and_absolute_powers(tmp_path):
    rows=[]
    for trim in [0,5,10,15]:
        for trial in [1,2]:
            row=dict(Participant="a",GlobalTrialOrder=trial,onset_trim_s=trim,CommonQCIncluded=True,WWR=15,Complexity=0,ExerciseFrequency="High",Gender="Female",Block=1,PositionWithinBlock=trial,PositionWithinBlockCentered=trial-3.5,OrderGroup="order1",PreviousWWR=np.nan if trial==1 else 15,PreviousComplexity=np.nan if trial==1 else 0)
            for roi in ["F","P","O"]:
                for band in ["theta","alpha","beta"]:
                    row[f"{roi}_{band}_absolute"]=2;row[f"{roi}_{band}_relative"]=.2;row[f"{roi}_{band}_relative_1_40"]=.21
            rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path/"independent_eeg_trial_QC.csv",index=False)
    root=current_model_inputs({},tmp_path)
    a=pd.read_csv(root/"D_current_QC_1_45/trim_0s/input.csv");b=pd.read_csv(root/"E_current_QC_1_40/trim_0s/input.csv")
    assert a.WWR.eq("WWR15").all() and a.ExperienceGroup.eq("High").all()
    pd.testing.assert_frame_equal(a.filter(regex="log10_"),b.filter(regex="log10_"))
    assert np.allclose(a.F_theta_relative,.2,rtol=1e-10,atol=1e-12) and np.allclose(b.F_theta_relative,.21,rtol=1e-10,atol=1e-12)


def test_real_r_current_qc_htz_absolute_reuse_and_missing_family(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    rs=Path("C:/Program Files/R/R-4.5.3/bin/x64/Rscript.exe")
    if not rs.exists(): pytest.skip("System R unavailable")
    rng=np.random.default_rng(92);rows=[]
    for person in range(32):
        shift=rng.normal(0,.4);conditions=[(w,c) for _ in range(2) for w in [15,45,75] for c in [0,1]];rng.shuffle(conditions)
        for trial,(w,c) in enumerate(conditions,1):
            row=dict(Participant=f"p{person}",GlobalTrialOrder=trial,WWR=f"WWR{w}",Complexity=f"C{c}",ExperienceGroup=["High","Low"][person%2],Gender=["Female","Male"][(person//2)%2],OrderGroup=["new order2","order1","order2"][person%3],Block=(trial-1)//6+1,PositionWithinBlock=(trial-1)%6+1,PositionWithinBlockCentered=(trial-1)%6-2.5,PreviousWWR="WWR15",PreviousComplexity="C0")
            row['P_theta_relative']=shift+.01*w+rng.normal(0,.25);row['log10_P_theta_absolute']=shift+.02*w+rng.normal(0,.3);rows.append(row)
    for version in ['D_current_QC_1_45','E_current_QC_1_40']:
        target=tmp_path/version/'trim_0s';target.mkdir(parents=True);frame=pd.DataFrame(rows)
        if version.startswith('E'): frame.P_theta_relative*=1.001
        frame.to_csv(target/'input.csv',index=False)
    contract={'factor_levels':{'WWR':['WWR15','WWR45','WWR75'],'Complexity':['C0','C1'],'ExperienceGroup':['High','Low'],'Gender':['Female','Male'],'OrderGroup':['new order2','order1','order2'],'PreviousWWR':['WWR15','WWR45','WWR75'],'PreviousComplexity':['C0','C1']}}
    (tmp_path/'contract.json').write_text(json.dumps(contract));job={'inputs':str(tmp_path),'outdir':str(tmp_path),'contract':str(tmp_path/'contract.json'),'test_trim':0,'test_outcomes':['P_theta_relative','log10_P_theta_absolute']};(tmp_path/'job.json').write_text(json.dumps(job))
    result=subprocess.run([str(rs),'--vanilla',str(repo/'analysis/r/eeg_independent_current_qc_models.R'),str(tmp_path/'job.json')],env=r_env({'r_library':str(repo/'.codex_tmp/r45-lib')}),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    factors=pd.read_csv(tmp_path/'current_QC_factor_tests.csv');assert len(factors)==24 and np.isfinite(factors.raw_p).all()
    families=pd.read_csv(tmp_path/'current_QC_coefficient_families.csv');assert len(families.query('family_id=="coefficient_expanded_relative"'))==648
    assert families.joint_q.isna().all() # missing models remain in full family
    co=pd.read_csv(tmp_path/'current_QC_coefficients.csv');a=co.query('version=="D_current_QC_1_45" and outcome=="log10_P_theta_absolute"');b=co.query('version=="E_current_QC_1_40" and outcome=="log10_P_theta_absolute"')
    assert np.array_equal(a.estimate,b.estimate)


def test_real_r_independent_models_and_complete_families(tmp_path):
    rs = Path("C:/Program Files/R/R-4.5.3/bin/x64/Rscript.exe")
    if not rs.exists(): pytest.skip("System R unavailable")
    repo = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(73)
    records = []
    for person in range(30):
        intercept = rng.normal(0,.2)
        conditions = [(w,c) for _ in range(2) for c in [0,1] for w in [15,45,75]]
        rng.shuffle(conditions)
        for trial in range(12):
            w,c = conditions[trial]
            z = intercept + .2*([15,45,75].index(w)) + rng.normal(0,.35)
            records.append({"Participant": f"p{person}", "GlobalTrialOrder": trial+1, "WWR": w,
                            "Complexity": c, "ExerciseFrequency": ["Low","High"][person%2],
                            "Gender": ["Male","Female"][(person//2)%2], "OrderGroup": ["order1","order2","new order2"][person%3],
                            "Block": trial//6+1, "PositionWithinBlockCentered": trial%6-2.5, "QC60": True,
                            "TableShare": 1/(1+np.exp(-z)), "WindowShare": 1/(1+np.exp(z+.3)), "RawCompetition": z,
                            "LogTableEnrichment": z+.1, "LogWindowEnrichment": -z, "AdjustedCompetition": 2*z+.1})
    pd.DataFrame(records).to_csv(tmp_path / "input.csv", index=False)
    job = {"input": str(tmp_path / "input.csv"), "outdir": str(tmp_path), "mode": "test"}
    (tmp_path / "job.json").write_text(json.dumps(job))
    env = r_env({"r_library": str(repo / ".codex_tmp/r45-lib")})
    result = subprocess.run([str(rs), "--vanilla", str(repo / "analysis/r/eye_independent_audit_analysis.R"), str(tmp_path / "job.json")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    factors = pd.read_csv(tmp_path / "independent_factor_tests.csv")
    assert len(factors) == 36 and factors.q.notna().all()
    assert factors.family_size.eq(3).all()
    contrasts = pd.read_csv(tmp_path / "independent_contrasts.csv")
    assert len(contrasts) == 24 and contrasts.contrast.eq("C0-C1").sum() == 6
    # Boundary data intentionally repeats each trial once per changed mask.
    boundary=pd.concat([pd.DataFrame(records).assign(AuditVariant='boundary_+5'),pd.DataFrame(records).assign(AuditVariant='boundary_-5')],ignore_index=True)
    boundary.to_csv(tmp_path/'boundary.csv',index=False);job['input']=str(tmp_path/'boundary.csv');job['mode']='boundary';(tmp_path/'job.json').write_text(json.dumps(job))
    result=subprocess.run([str(rs),'--vanilla',str(repo/'analysis/r/eye_independent_audit_analysis.R'),str(tmp_path/'job.json')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert len(pd.read_csv(tmp_path/'08_model_specification_audit.csv'))==12


def test_real_matlab_source_audit_roi_order_endpoints_and_dual_denominators(tmp_path):
    from scipy.io import savemat
    matlab = Path("D:/Program Files/MATLAB/R2026a/bin/matlab.exe")
    if not matlab.exists(): pytest.skip("MATLAB unavailable")
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "waveforms"; source.mkdir()
    fs, n = 500, 16000
    t = np.arange(n)/fs
    base = 2*np.sin(2*np.pi*6*t) + np.sin(2*np.pi*20*t) + .5*np.sin(2*np.pi*43*t)
    data = np.tile(base, (8,1))
    data[1] = -2*np.sin(2*np.pi*6*t)+np.sin(2*np.pi*20*t)+.5*np.sin(2*np.pi*43*t)
    data.astype('<f4').flatten(order='F').tofile(source / "synthetic.fdt")
    events = np.array([(7,1),(8,n)],dtype=[('type','O'),('latency','O')])
    channels = np.array([(x,) for x in ['F3','F4','P3','Pz','P4','O1','Oz','O2']],dtype=[('labels','O')])
    savemat(source / "synthetic.set", {'data':'synthetic.fdt','srate':fs,'nbchan':8,'pnts':n,'event':events,'chanlocs':channels},appendmat=False)
    output = tmp_path / "results";output.mkdir()
    job = tmp_path / "job.json";job.write_text(json.dumps({'preprocessed_root':str(source),'outdir':str(output)}))
    expression = f"addpath('{(repo/'matlab/eeg_bandpower_pipeline').as_posix()}');run_eeg_independent_source_audit('{job.as_posix()}')"
    result = subprocess.run([str(matlab),'-batch',expression],capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
    d = pd.read_csv(output / "independent_eeg_source_features.csv")
    assert len(d)==4 and d.AnalysisStartSample.tolist()==[1,2501,5001,7501]
    # Hamming leakage from the uncancelled 20/43-Hz signals is finite; averaging
    # PSDs instead of channel waveforms would retain approximately 2 theta units.
    assert (d.F_theta_absolute < d.P_theta_absolute * 1e-4).all()
    assert np.allclose(d.P_theta_absolute,2,rtol=.02)
    assert np.allclose(d.P_theta_relative_1_40,.8,rtol=.02)
    assert (d.P_total_1_45>d.P_total_1_40).all()
