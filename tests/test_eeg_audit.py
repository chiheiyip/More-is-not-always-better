from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from paper_analysis.teacher.contracts import canonicalize_trials, normalize_complexity
from paper_analysis.teacher.eeg_audit import (
    CORE, METRICS, TRIMS, CONDITION, TEMPORAL, PREVIOUS, FACTOR_LEVELS,
    family_tables, outcome_spec, prepare_inputs, run_eeg_audit,
)
from paper_analysis.teacher.state import StageBlockedError


@pytest.mark.parametrize("value,expected", [(0,"C0"),(0.0,"C0"),("0","C0"),("C0","C0"),(1,"C1"),(None,""),(np.nan,""),(pd.NA,"")])
def test_complexity_zero_is_not_missing(value, expected):
    assert normalize_complexity(value) == expected


def synthetic_config(tmp_path, n=36, unicode_ids=False):
    rng = np.random.default_rng(830)
    prefix = "参与者" if unicode_ids else "S"
    registry = pd.DataFrame([
        {"Participant": f"{prefix}{i:02}", "Gender": "Female" if i % 2 else "Male",
         "ExperienceGroup": "High" if (i//2)%2 else "Low",
         "OrderGroup": ["new order2", "order1", "order2"][i%3], "IncludeEEGValid": True}
        for i in range(n)
    ])
    rows = []
    for i in range(n):
        intercept = rng.normal(0, .15, 18)
        conditions = [(w,c) for w in (15,45,75) for c in (0,1)]
        order = rng.permutation(6)
        for b in (1,2):
            for position, ci in enumerate(order, 1):
                w,c = conditions[ci]
                relative = .5 + intercept[:9] + rng.normal(0,.03,9) + .006*position
                absolute = np.exp(2 + intercept[9:] + rng.normal(0,.08,9) + .02*b)
                for trim in TRIMS:
                    row = {"Participant":f"{prefix}{i:02}", "Block":b,"PositionWithinBlock":position,
                           "GlobalTrialOrder":(b-1)*6+position,"WWR":w,"Complexity":c,
                           "onset_trim_s":trim,"bad_eeg_quality":False,"eeg_subject_quality_exclusion":False}
                    for j,m in enumerate(METRICS):
                        row[f"{m}_relative"] = relative[j] + trim*.0001
                        row[f"{m}_absolute"] = absolute[j] + trim*.001
                    rows.append(row)
    registry.to_csv(tmp_path/"registry.csv",index=False)
    pd.DataFrame(rows).to_csv(tmp_path/"input.csv",index=False)
    pd.DataFrame([{"Parameter":"relative_power_denominator_hz","Value":"1-45"}]).to_csv(tmp_path/"preprocessing.csv",index=False)
    config = {"participant_information":str(tmp_path/"registry.csv"),
              "rscript":str(Path(__file__).resolve().parents[1]/"scripts/portable_rscript.cmd"),
              "eeg":{"onset_sensitivity_trial_file":str(tmp_path/"input.csv"),"onset_trim_variants_s":list(TRIMS),
                     "onset_trim_strategy":"parallel","core_metrics":list(CORE),
                     "preprocessing_audit_file":str(tmp_path/"preprocessing.csv"),"preprocessing_confirmed":True,
                     "preprocessing_parameters":{"relative_power_denominator_hz":"1-45"}}}
    (tmp_path/"config.json").write_text(json.dumps(config),encoding="utf-8")
    return config


def test_common_sample_and_predecessor_before_qc(tmp_path):
    config = synthetic_config(tmp_path, 4)
    path = tmp_path/"input.csv"; raw = pd.read_csv(path)
    mask = raw.Participant.eq("S00") & raw.GlobalTrialOrder.eq(2) & raw.onset_trim_s.eq(5)
    raw.loc[mask,"F_theta_absolute"] = 0
    raw.to_csv(path,index=False)
    prepared, exclusions = prepare_inputs(config)
    assert {len(f) for f in prepared.values()} == {47}
    assert exclusions.reason.str.contains("nonpositive").any()
    f = prepared[0]
    third = f.loc[f.Participant.eq("S00") & f.GlobalTrialOrder.eq(3)].iloc[0]
    second = raw.loc[raw.Participant.eq("S00") & raw.GlobalTrialOrder.eq(2)].iloc[0]
    assert third.PreviousWWR == f"WWR{second.WWR}"
    assert f.loc[f.PositionWithinBlock.eq(1),"PreviousWWR"].isna().all()


def test_missing_trial_not_bridged():
    frame = pd.DataFrame({"Participant":["A"]*3,"Block":[1,1,2],"PositionWithinBlock":[1,3,1],"WWR":[15,45,75],"Complexity":[0,1,0]})
    actual = canonicalize_trials(frame)
    assert actual.PreviousWWR.isna().all()


@pytest.mark.parametrize("case", ["duplicate","missing_window","missing_outcome","empty_sample"])
def test_bad_inputs_fail(tmp_path, case):
    config=synthetic_config(tmp_path,4); path=tmp_path/"input.csv"; raw=pd.read_csv(path)
    if case=="duplicate":raw=pd.concat([raw,raw.iloc[:1]])
    if case=="missing_window":raw=raw.loc[raw.onset_trim_s.ne(15)]
    if case=="missing_outcome":raw=raw.drop(columns="P_alpha_relative")
    if case=="empty_sample":raw["P_alpha_absolute"]=0
    raw.to_csv(path,index=False)
    with pytest.raises(StageBlockedError):prepare_inputs(config)


def coefficient_fixture():
    rows=[]
    for trim in TRIMS:
        for spec in outcome_spec().to_dict("records"):
            for term in (*CONDITION,*TEMPORAL):
                rows.append({"onset_trim_s":trim,"outcome":spec["outcome"],"model":"Model1","term":term,"p.value":.01+len(rows)*.00001,"estimate":.3,"inference_valid":True})
            if spec["core"]:
                for term in PREVIOUS:
                    rows.append({"onset_trim_s":trim,"outcome":spec["outcome"],"model":"PreviousScene","term":term,"p.value":.03,"estimate":-.1,"inference_valid":True})
    return pd.DataFrame(rows)


def test_explicit_family_sizes_and_failure_isolation():
    source=coefficient_fixture(); tables,status=family_tables(source)
    assert status.loc[status.scope.eq("joint"),"expected_tests"].tolist()==[72,72,144,324,324,48]
    assert status.status.eq("complete").all()
    missing=source.loc[~(source.outcome.eq("F_alpha_relative") & source.term.eq("Block") & source.onset_trim_s.eq(5))]
    tables,status=family_tables(missing)
    relative=tables.loc[tables.family_id.eq("temporal_relative")]
    assert relative.joint_q.isna().all()
    assert relative.loc[relative.onset_trim_s.eq(5),"within_q"].isna().all()
    assert relative.loc[relative.onset_trim_s.eq(0),"within_q"].notna().all()
    assert tables.loc[tables.family_id.eq("temporal_absolute"),"joint_q"].notna().all()
    assert relative.corrected_detected.isna().all()


def test_duplicate_coefficients_rejected():
    source=coefficient_fixture()
    with pytest.raises(StageBlockedError):family_tables(pd.concat([source,source.iloc[:1]]))


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_p_does_not_reduce_family(bad):
    source=coefficient_fixture()
    source.loc[0,"p.value"]=bad
    table,status=family_tables(source)
    selected=table.loc[table.family_id.eq("condition_core_relative")]
    assert len(selected)==144
    assert selected.joint_q.isna().all()
    assert status.loc[status.family_id.eq("condition_core_relative") & status.scope.eq("joint"),"valid_tests"].iloc[0]==143


def test_bad_diagnostic_suppresses_final_q():
    source=coefficient_fixture(); source.loc[0,"inference_valid"]=False
    table,_=family_tables(source)
    assert table.loc[table.family_id.eq("condition_core_relative"),"joint_q"].isna().all()


def test_dry_run_is_read_only(tmp_path, capsys):
    import runpy
    config=synthetic_config(tmp_path,4)
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/run_teacher_analysis.py"))
    out=tmp_path/"not_created"
    code=module["main"](["eeg-audit","--config",str(tmp_path/"config.json"),"--outdir",str(out),"--dry-run"])
    assert code==0 and not out.exists()
    assert json.loads(capsys.readouterr().out)["common_trials"]==48
    with pytest.raises(StageBlockedError):
        run_eeg_audit(config,config_path=tmp_path/"config.json",outdir=out,repo_root=tmp_path,r_required=False)
    assert not out.exists()


def test_history_comparison_does_not_duplicate_other_windows(tmp_path, monkeypatch):
    import runpy
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    compare=runpy.run_path(str(root/"scripts/validate_eeg_audit.py"))["compare_history"]
    config=synthetic_config(tmp_path,4)
    prepared,_=prepare_inputs(config)
    out=tmp_path/"validation"; historical=tmp_path/"old"
    out.mkdir(); new=[]
    for trim in TRIMS:
        olddir=historical/"onset_window_models"/f"trim_{trim}s"; olddir.mkdir(parents=True)
        newdir=out/f"trim_{trim}s"; newdir.mkdir()
        prepared[trim].to_csv(olddir/"eeg_onset_order_model_input.csv",index=False)
        prepared[trim].to_csv(newdir/"input.csv",index=False)
        rows=pd.DataFrame([{"outcome":f"{m}_relative","model":"Model1","term":"Block","estimate":trim+.1,"std.error":.01,"df":30.,"p.value":.04} for m in CORE])
        rows.to_csv(olddir/"eeg_cr2_robust_results.csv",index=False)
        rows["onset_trim_s"]=trim; new.append(rows)
    pd.concat(new).to_csv(out/"coefficients.csv",index=False)
    summary=compare(out,historical)
    assert summary["matched_rows"]==16
    assert summary["unmatched_rows"]==0
    assert summary["comparison_status"]=="concordant_within_numerical_tolerance"


def test_real_r_synthetic_integration(tmp_path):
    root=Path(__file__).resolve().parents[1]
    if not (root/".r-env/Lib/R/bin/Rscript.exe").exists():pytest.skip("Portable R not installed")
    config=synthetic_config(tmp_path, unicode_ids=True)
    out=tmp_path/"validation"
    run_eeg_audit(config,config_path=tmp_path/"config.json",outdir=out,repo_root=root)
    diag=pd.read_csv(out/"diagnostics.csv")
    assert len(diag.loc[diag.model.eq("Model1")])==72
    assert diag.loc[diag.model.eq("Model1"),"n_trials"].eq(432).all()
    assert diag.loc[diag.model.eq("Model1"),"n_participants"].eq(36).all()
    assert len(diag.loc[diag.model.ne("Model1")])==48
    coef=pd.read_csv(out/"coefficients.csv")
    assert len(coef)>0
    assert set(CONDITION).issubset(set(coef.term))
    assert set(PREVIOUS).issubset(set(coef.term))
    assert pd.read_csv(out/"family_status.csv").status.eq("complete").all()
    assert json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))["status"]=="validated"
    np.testing.assert_allclose(coef.CI_low,coef.estimate-t.ppf(.975,coef.df)*coef["std.error"],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(coef.CI_high,coef.estimate+t.ppf(.975,coef.df)*coef["std.error"],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(coef["p.value"],2*t.sf(abs(coef.estimate/coef["std.error"]),coef.df),rtol=1e-9,atol=1e-12)
    source=coefficient_fixture(); audit,_=family_tables(source)
    audit.to_csv(tmp_path/"bh_input.csv",index=False)
    check=tmp_path/"check_bh.R"
    check.write_text("args<-commandArgs(TRUE); x<-read.csv(args[1]); for (f in unique(x$family_id)) {z<-x[x$family_id==f,]; stopifnot(max(abs(p.adjust(z$p.value,'BH')-z$joint_q))<1e-12)}",encoding="utf-8")
    subprocess.run([config["rscript"],str(check),str(tmp_path/"bh_input.csv")],check=True)
    assert json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))["purpose"]=="code_validation_not_formal_results"
    with pytest.raises(StageBlockedError):run_eeg_audit(config,config_path=tmp_path/"config.json",outdir=out,repo_root=root)
