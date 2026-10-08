from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from paper_analysis.teacher.fresh import set_record, stage, verify_inputs, resolved_config
from paper_analysis.teacher.state import StageBlockedError, file_sha256


def test_resume_rejects_changed_output_or_identity(tmp_path):
    def produce():
        (tmp_path / "stage1").mkdir()
        (tmp_path / "stage1/data.csv").write_text("x\n1\n")
    stage(tmp_path,"stage1","identity",produce,False)
    stage(tmp_path,"stage1","identity",lambda:pytest.fail("re-executed"),True)
    with pytest.raises(StageBlockedError,match="source/code/config"):
        stage(tmp_path,"stage1","changed",produce,True)
    (tmp_path / "stage1/data.csv").write_text("x\n2\n")
    with pytest.raises(StageBlockedError,match="artifact changed"):
        stage(tmp_path,"stage1","identity",produce,True)


def test_set_checks_referenced_fdt_and_event_pairs(tmp_path):
    p=tmp_path/"participant.set";fdt=tmp_path/"actual_reference.fdt"
    labels=["F3","F4","P3","Pz","P4","O1","Oz","O2"]
    payload={"data":"actual_reference.fdt","nbchan":8,"pnts":300,"trials":1,"srate":500,
        "chanlocs":np.array([{"labels":v} for v in labels],dtype=object),
        "event":np.array([{"type":7 if i%2==0 else 8,"latency":i+1} for i in range(24)],dtype=object)}
    savemat(p,payload);fdt.write_bytes(bytes(8*300*4))
    alias={p.name:{"declared":fdt.name,"actual":"participant.fdt"}}
    with pytest.raises(StageBlockedError,match="alias required"):set_record(p)
    fdt.rename(tmp_path/"participant.fdt");fdt=tmp_path/"participant.fdt"
    assert set_record(p,alias)["fdt_path"]==str(fdt)
    fdt.write_bytes(bytes(4))
    with pytest.raises(StageBlockedError,match="dimensions"):
        set_record(p,alias)
    fdt.write_bytes(bytes(8*300*4));payload["event"][1]={"type":7,"latency":2};savemat(p,payload)
    with pytest.raises(StageBlockedError,match="events"):
        set_record(p,alias)


def test_changed_input_cannot_silently_reuse(tmp_path):
    p=tmp_path/"中文.csv";p.write_text("\ufeffParticipant,value\n甲,1\n",encoding="utf-8")
    record={"path":str(p),"sha256":file_sha256(p)}
    verify_inputs([record]);p.write_text("Participant,value\n甲,2\n",encoding="utf-8")
    with pytest.raises(StageBlockedError,match="Source changed"):
        verify_inputs([record])


def test_fresh_config_cuts_historical_dependencies_and_questionnaire_models(tmp_path):
    cfg={"participant_information":"old.xlsx","trial_order_mapping":"old.xlsx","scene_aoi_mapping":"old.xlsx",
        "eeg":{"trial_file":"history.csv","onset_analysis_dir":"history","preprocessing_audit_file":"history.csv"},
        "eye":{},"stage3":{"s3_trial_file":"old_questionnaire.csv"},"denominator_sensitivity":{"historical_package":"old.zip"},
        "synchronized_timebin_file":"old.csv","clock_scene_qc_file":"old.csv"}
    result=resolved_config(cfg,tmp_path)
    assert "denominator_sensitivity" not in result
    assert result["stage3"]["s3_trial_file"]==""
    assert result["eeg"]["onset_analysis_dir"]==""
    for path in [result["eeg"]["trial_file"],result["eeg"]["onset_sensitivity_trial_file"],result["synchronized_timebin_file"],result["clock_scene_qc_file"]]:
        assert Path(path).is_relative_to(tmp_path)
    assert cfg["eeg"]["trial_file"]=="history.csv"


def test_partial_stage_is_preserved(tmp_path):
    (tmp_path/"stage1").mkdir();(tmp_path/"stage1/data.csv").write_text("incomplete")
    with pytest.raises(StageBlockedError,match="Unsealed stage"):
        stage(tmp_path,"stage1","identity",lambda:None,True)
    assert (tmp_path/"stage1/data.csv").read_text()=="incomplete"


def test_hdf_set_and_additional_block_end_markers(tmp_path):
    import h5py
    p=tmp_path/"中文.set";p.with_suffix(".fdt").write_bytes(bytes(8*300*4))
    with h5py.File(p,"w") as f:
        def char(name,value):
            d=f.create_dataset(name,data=np.array([ord(v) for v in value],dtype=np.uint16).reshape(-1,1))
            d.attrs["MATLAB_class"]=b"char";return d.ref
        char("data","中文.fdt");char("history","original preprocessing")
        for k,v in {"nbchan":8,"pnts":300,"trials":1,"srate":500}.items():f.create_dataset(k,data=[[v]])
        refs=[char(f"refs/channel{i}",v) for i,v in enumerate(["F3","F4","P3","Pz","P4","O1","Oz","O2"])]
        f.create_dataset("chanlocs/labels",data=np.array(refs,dtype=h5py.ref_dtype).reshape(-1,1))
        refs=[char(f"refs/event{i}",v) for i,v in enumerate(["7","8"]*12+["8","6"])]
        f.create_dataset("event/type",data=np.array(refs,dtype=h5py.ref_dtype).reshape(1,-1))
    assert set_record(p)["srate"]==500


def test_cli_prohibits_historical_reuse(tmp_path):
    import runpy
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/run_teacher_analysis.py"))
    cfg=tmp_path/"config.json";cfg.write_text("{}")
    with pytest.raises(ValueError,match="prohibits historical reuse"):
        module["main"](["all-results","--scope","eye-eeg","--fresh-from-source","--reuse-valid","--config",str(cfg),"--dry-run"])
