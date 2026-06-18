from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_analysis.diagnostics.pipeline import run_diagnostics
from paper_analysis.eeg.pipeline import run_eeg_pipeline
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline
from paper_analysis.figures import run_figure_pipeline
from paper_analysis.fusion.pipeline import run_fusion_pipeline
from paper_analysis.intake.pipeline import build_manifests
from paper_analysis.questionnaire.pipeline import run_questionnaire_pipeline
from paper_analysis.reporting.pipeline import build_paper_outputs
from paper_analysis.stats.models import run_statistical_models
from paper_analysis.utils.coding import experience_group


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "paper"


def test_experience_group_uses_two_by_two_frequency_split() -> None:
    assert experience_group("从不或极少（每月＜1次）") == "Low"
    assert experience_group("偶尔（每月1–2次）") == "Low"
    assert experience_group("有时（每月3-4次）") == "High"
    assert experience_group("经常（每月≥5次）") == "High"


def test_eeg_model_config_covers_roi_band_grid() -> None:
    config = json.loads(Path("configs/model_families.json").read_text(encoding="utf-8"))
    eeg_outcomes = set()
    for family in config["families"]:
        if family["name"] == "eeg_roi_band_primary":
            eeg_outcomes.update(family["outcomes"])
    assert eeg_outcomes == {
        "eeg_F_theta",
        "eeg_F_alpha",
        "eeg_F_beta",
        "eeg_P_theta",
        "eeg_P_alpha",
        "eeg_P_beta",
        "eeg_O_theta",
        "eeg_O_alpha",
        "eeg_O_beta",
    }


def test_full_pipeline_builds_paper_outputs(tmp_path: Path) -> None:
    intake = build_manifests(
        participants_csv=FIXTURES / "participants.csv",
        scene_manifest_csv=FIXTURES / "scene_manifest.csv",
        outdir=tmp_path / "outputs" / "01_sample_qc",
    )
    questionnaire = run_questionnaire_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        questionnaire_wide=FIXTURES / "questionnaire" / "questionnaire_wide.csv",
        outdir=tmp_path / "outputs" / "02_questionnaire",
    )
    eye = run_eye_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        outdir=tmp_path / "outputs" / "03_eye_tracking",
    )
    eeg = run_eeg_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        eeg_scene_csv=FIXTURES / "eeg" / "eeg_scene.csv",
        outdir=tmp_path / "outputs" / "04_eeg",
    )
    fusion = run_fusion_pipeline(
        questionnaire_long=questionnaire["questionnaire_long"],
        eye_aoi_trial_long=eye["eye_aoi_trial_long"],
        eeg_trial_long=eeg["eeg_trial_long"],
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        outdir=tmp_path / "outputs" / "05_multimodal_fusion",
        expected_scenes_per_subject=3,
    )
    stats = run_statistical_models(
        master_csv=fusion["analysis_master_long"],
        model_config=Path("configs/model_families.json"),
        outdir=tmp_path / "outputs" / "06_models",
    )
    diagnostics = run_diagnostics(
        master_csv=fusion["analysis_master_long"],
        participants_csv=intake["participants_standardized"],
        outdir=tmp_path / "outputs" / "06_robustness",
    )
    reporting = build_paper_outputs(
        model_results_csv=stats["model_results"],
        diagnostics_dir=tmp_path / "outputs" / "06_robustness",
        reviewer_map=Path("configs/reviewer_response_map.json"),
        outdir=tmp_path / "outputs" / "07_paper_tables",
    )
    figures = run_figure_pipeline(
        outputs_root=tmp_path / "outputs",
        figure_contracts_config=Path("configs/figure_contracts.json"),
        outdir=tmp_path / "outputs" / "10_figures",
    )

    master = pd.read_csv(fusion["analysis_master_long"])
    assert {"participant_id", "scene_id", "q_S1", "eeg_O_theta", "FCR", "attention_share"}.issubset(master.columns)
    assert pd.read_csv(fusion["aligned_scene"]).shape[0] >= 6
    assert {"bin_index", "bin_start_ms", "bin_end_ms", "eeg_O_theta"}.issubset(pd.read_csv(fusion["aligned_timebin"]).columns)
    assert {"duration_delta_s", "duration_mismatch"}.issubset(pd.read_csv(fusion["sync_qc"]).columns)
    assert {"time_sync_slope", "time_sync_offset_ms"}.issubset(pd.read_csv(fusion["time_sync_map"]).columns)
    assert pd.read_csv(intake["group_balance"])["factor"].isin(["ExperienceGroup"]).any()
    assert pd.read_csv(eye["aoi_validation_summary"])["class_name"].isin(["table", "window"]).any()
    assert pd.read_csv(stats["model_diagnostics"]).shape[0] > 0
    assert pd.read_csv(stats["eeg_core_lmm"]).shape[0] > 0
    assert pd.read_csv(stats["eeg_peak_index"]).shape[0] > 0
    assert pd.read_csv(stats["eeg_trial_index_models"]).shape[0] > 0
    assert pd.read_csv(diagnostics["nonlinear_wwr_sensitivity"]).shape[0] > 0
    assert pd.read_csv(reporting["claim_strength_table"]).shape[0] > 0
    assert {"figure_id", "source_data", "review_risk"}.issubset(pd.read_csv(reporting["figure_contracts_index"]).columns)
    assert {"figure_id", "source_file"}.issubset(pd.read_csv(reporting["source_data_index"]).columns)
    assert {"issue_id", "response_readiness"}.issubset(pd.read_csv(reporting["reviewer_issue_matrix"]).columns)
    assert {"dataset_id", "access_route", "identifier"}.issubset(pd.read_csv(reporting["data_availability_index"]).columns)
    assert reporting["data_availability_statement"].exists()
    figure_manifest = pd.read_csv(figures["figure_manifest"])
    figure_qa = pd.read_csv(figures["figure_qa"])
    assert {"figure_id", "svg", "pdf", "tiff", "png", "source_csv", "qa_status"}.issubset(figure_manifest.columns)
    assert figure_manifest.shape[0] == 5
    assert figure_qa["qa_status"].isin(["pass"]).all()
    for _, row in figure_manifest.iterrows():
        assert Path(row["svg"]).exists()
        assert Path(row["pdf"]).exists()
        assert Path(row["tiff"]).exists()
        assert Path(row["png"]).exists()
        assert Path(row["source_csv"]).exists()


def test_questionnaire_wide_to_long_keeps_design_columns(tmp_path: Path) -> None:
    intake = build_manifests(FIXTURES / "participants.csv", FIXTURES / "scene_manifest.csv", tmp_path / "sample_qc")
    questionnaire = run_questionnaire_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        questionnaire_wide=FIXTURES / "questionnaire" / "questionnaire_wide.csv",
        outdir=tmp_path / "questionnaire",
    )
    q = pd.read_csv(questionnaire["questionnaire_long"])
    assert len(q) == 6
    assert {"WWR", "Complexity", "ExperienceGroup", "Gender", "Age"}.issubset(q.columns)
    assert {"Afford4", "Afford4_n_valid"}.issubset(q.columns)
    desc = pd.read_csv(questionnaire["s_items_descriptives"])
    assert {"median", "ci95_low", "ci95_high", "skewness", "kurtosis", "shapiro_p"}.issubset(desc.columns)
    reliability = pd.read_csv(questionnaire["questionnaire_reliability"])
    assert reliability["scale"].isin(["S1_S4_Afford4_candidate"]).any()
    poly = pd.read_csv(questionnaire["questionnaire_wwr_polynomial_contrasts"])
    assert poly["contrast"].isin(["Linear", "Quadratic"]).any()
    assert set(q["Gender"].dropna()) == {"Female", "Male"}
    assert q[["participant_id", "scene_id"]].duplicated().sum() == 0


def test_questionnaire_enhanced_outputs_handle_ipq_b_items_and_s5_scale(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene = tmp_path / "scene_manifest.csv"
    questionnaire = tmp_path / "questionnaire_long.csv"
    participants.write_text(
        "participant_id,ExperienceGroup,SportFreqGroup,Gender,Age,exclude\n"
        "P01,Low,Low,Female,22,false\n"
        "P02,High,High,Male,24,false\n"
        "P03,High,Low,Female,23,false\n",
        encoding="utf-8",
    )
    scene.write_text(
        "participant_id,scene_id,WWR,Complexity,Cond,block,position\n"
        "P01,1,15,0,C0,1,1\nP01,2,45,1,C1,1,2\nP01,3,75,0,C0,1,3\n"
        "P02,1,15,0,C0,1,1\nP02,2,45,1,C1,1,2\nP02,3,75,0,C0,1,3\n"
        "P03,1,15,0,C0,1,1\nP03,2,45,1,C1,1,2\nP03,3,75,0,C0,1,3\n",
        encoding="utf-8",
    )
    questionnaire.write_text(
        "participant_id,scene_id,S1,S2,S3,S4,S5,B1,B2,B3,IPQ1,IPQ2,IPQ3,IPQ4,IPQ5,IPQ6\n"
        "P01,1,5,5,5,5,8,,,,4,4,5,5,4,5\n"
        "P01,2,6,6,6,6,9,6,6,5,4,4,5,5,4,5\n"
        "P01,3,4,4,4,4,7,,,,4,4,5,5,4,5\n"
        "P02,1,4,4,4,4,7,,,,5,5,5,6,5,6\n"
        "P02,2,5,5,5,5,8,5,5,5,5,5,5,6,5,6\n"
        "P02,3,3,3,3,3,6,,,,5,5,5,6,5,6\n"
        "P03,1,3,3,3,3,6,,,,6,6,6,6,6,6\n"
        "P03,2,4,4,4,4,7,4,4,4,6,6,6,6,6,6\n"
        "P03,3,2,2,2,2,5,,,,6,6,6,6,6,6\n",
        encoding="utf-8",
    )
    out = run_questionnaire_pipeline(participants, scene, tmp_path / "questionnaire", questionnaire_long=questionnaire)
    q = pd.read_csv(out["questionnaire_long"])
    assert {"S5_7", "Bmean", "Afford4", "IPQ_mean"}.issubset(q.columns)
    assert q["S5_7"].notna().any()
    assert q.loc[q["Cond"].eq("C1"), "Bmean"].notna().all()
    b_qc = pd.read_csv(out["questionnaire_b_item_qc"])
    assert b_qc.loc[0, "status"] == "pass"
    ipq = pd.read_csv(out["ipq_subject_level"])
    assert ipq.shape[0] == 3
    rel = pd.read_csv(out["questionnaire_reliability"])
    assert rel["scale"].isin(["IPQ1_IPQ6_subject_level"]).any()
    poly = pd.read_csv(out["questionnaire_wwr_polynomial_contrasts"])
    assert set(poly["contrast"].dropna()) >= {"Linear", "Quadratic"}
    assert poly["claim_strength"].dropna().eq("trend_only").all()


def test_eeg_pipeline_robust_qc_flags_trials_and_keeps_legacy_audit(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene = tmp_path / "scene_manifest.csv"
    eeg = tmp_path / "eeg_scene.csv"
    participants.write_text(
        "participant_id,eeg_subject_id,exclude\n"
        "P01,P01,false\n"
        "P02,P02,false\n",
        encoding="utf-8",
    )
    scene.write_text(
        "participant_id,scene_id,WWR,Complexity,block,position\n"
        "P01,1,15,0,1,1\n"
        "P01,2,45,0,1,2\n"
        "P02,1,15,1,1,1\n"
        "P02,2,45,1,1,2\n",
        encoding="utf-8",
    )
    eeg.write_text(
        "participant_id,scene_id,view_dur_s,O_theta,hf_ratio_20_40Hz,rms_mean_uV,peak_to_peak_uV,nan_fraction,flat_fraction\n"
        "P01,1,3,1.0,0.10,10,50,0,0\n"
        "P01,2,3,1.1,0.11,11,51,0,0\n"
        "P02,1,3,1.2,0.12,12,52,0,0\n"
        "P02,2,3,1.3,0.80,13,53,0,0\n",
        encoding="utf-8",
    )

    robust = run_eeg_pipeline(
        participants_csv=participants,
        scene_manifest_csv=scene,
        eeg_scene_csv=eeg,
        outdir=tmp_path / "eeg_robust",
        eeg_qc_config={"policy": "robust", "robust_min_n": 4, "bad_scene_fraction_threshold": 0.9},
    )
    robust_trial = pd.read_csv(robust["eeg_trial_long"])
    bad = robust_trial.set_index(["participant_id", "scene_id"]).loc[("P02", 2)]
    assert bool(bad["bad_eeg_quality"]) is True
    assert bool(bad["eeg_legacy_hf_flag"]) is True
    assert "robust_hf_ratio_20_40Hz" in bad["eeg_qc_reasons"]
    assert "legacy_hf_ratio" not in bad["eeg_qc_reasons"]

    legacy = run_eeg_pipeline(
        participants_csv=participants,
        scene_manifest_csv=scene,
        eeg_scene_csv=eeg,
        outdir=tmp_path / "eeg_legacy",
        eeg_qc_config={"policy": "legacy_0_4", "robust_min_n": 4, "bad_scene_fraction_threshold": 0.9},
    )
    legacy_bad = pd.read_csv(legacy["eeg_trial_long"]).set_index(["participant_id", "scene_id"]).loc[("P02", 2)]
    assert "legacy_hf_ratio" in legacy_bad["eeg_qc_reasons"]


def test_eeg_pipeline_subject_quality_exclusion_and_old_csv_compatibility(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene = tmp_path / "scene_manifest.csv"
    eeg = tmp_path / "eeg_scene.csv"
    participants.write_text("participant_id,eeg_subject_id,exclude\nP01,P01,false\n", encoding="utf-8")
    scene.write_text(
        "participant_id,scene_id,WWR,Complexity\n"
        "P01,1,15,0\n"
        "P01,2,45,0\n"
        "P01,3,75,0\n",
        encoding="utf-8",
    )
    eeg.write_text(
        "participant_id,scene_id,view_dur_s,O_theta,segment_valid_duration\n"
        "P01,1,3,1.0,true\n"
        "P01,2,3,1.1,false\n"
        "P01,3,3,1.2,false\n",
        encoding="utf-8",
    )

    out = run_eeg_pipeline(
        participants_csv=participants,
        scene_manifest_csv=scene,
        eeg_scene_csv=eeg,
        outdir=tmp_path / "eeg_subject",
        eeg_qc_config={"policy": "robust", "bad_scene_fraction_threshold": 0.3},
    )
    trial = pd.read_csv(out["eeg_trial_long"])
    assert trial["eeg_subject_quality_exclusion"].astype(str).str.lower().isin(["true", "1"]).all()
    assert trial["bad_eeg_quality"].astype(str).str.lower().isin(["true", "1"]).all()

    old_eeg = tmp_path / "old_eeg_scene.csv"
    old_eeg.write_text(
        "participant_id,scene_id,view_dur_s,O_theta\n"
        "P01,1,3,1.0\n"
        "P01,2,3,1.1\n"
        "P01,3,3,1.2\n",
        encoding="utf-8",
    )
    old = run_eeg_pipeline(participants, scene, old_eeg, tmp_path / "old_eeg")
    old_trial = pd.read_csv(old["eeg_trial_long"])
    assert old_trial["eeg_qc_policy"].eq("unavailable").all()
    assert old_trial["bad_eeg_quality"].astype(str).str.lower().isin(["false", "0"]).all()


def test_fusion_qc_filters_analysis_master_and_keeps_audit_tables(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene_manifest = tmp_path / "scene_manifest.csv"
    questionnaire = tmp_path / "questionnaire_long.csv"
    eye_metrics = tmp_path / "eye_aoi_trial_long.csv"
    eeg = tmp_path / "eeg_trial_long.csv"
    eye1 = tmp_path / "P01_scene01.csv"
    eye2 = tmp_path / "P01_scene02.csv"
    eye3 = tmp_path / "P02_scene01.csv"
    eye1.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n0,1,1\n1000,1,1\n2000,1,1\n", encoding="utf-8")
    eye2.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n0,1,1\n5000,1,1\n", encoding="utf-8")
    eye3.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n0,1,1\n1000,1,1\n2000,1,1\n", encoding="utf-8")
    participants.write_text(
        "participant_id,eeg_subject_id,eye_subject_id,ExperienceGroup,Gender,exclude\n"
        "P01,P01,P01,Low,F,false\n"
        "P02,P02,P02,High,M,false\n",
        encoding="utf-8",
    )
    scene_manifest.write_text(
        "participant_id,scene_id,condition_id,WWR,Complexity,Cond,block,position,eye_csv_path,aoi_json_path\n"
        f"P01,1,C0_W15,15,0,C0,1,1,{eye1.as_posix()},\n"
        f"P01,2,C1_W45,45,1,C1,1,2,{eye2.as_posix()},\n"
        f"P02,1,C0_W15,15,0,C0,1,1,{eye3.as_posix()},\n",
        encoding="utf-8",
    )
    questionnaire.write_text(
        "participant_id,scene_id,S1\n"
        "P01,1,5\n"
        "P01,2,4\n"
        "P02,1,3\n",
        encoding="utf-8",
    )
    eye_metrics.write_text(
        "participant_id,scene_id,class_name,FCR,TFD_ms,TTFF_ms,attention_share,visited\n"
        "P01,1,whole_scene,1,100,0,1,true\n"
        "P01,2,whole_scene,1,100,0,1,true\n"
        "P02,1,whole_scene,1,100,0,1,true\n",
        encoding="utf-8",
    )
    eeg.write_text(
        "participant_id,scene_id,view_dur_s,O_theta\n"
        "P01,1,2,1.1\n"
        "P01,2,2,1.2\n"
        "P02,1,2,1.3\n",
        encoding="utf-8",
    )

    fusion = run_fusion_pipeline(
        questionnaire_long=questionnaire,
        eye_aoi_trial_long=eye_metrics,
        eeg_trial_long=eeg,
        participants_csv=participants,
        scene_manifest_csv=scene_manifest,
        outdir=tmp_path / "fusion",
        expected_scenes_per_subject=2,
        duration_tolerance_s=2.0,
    )

    master = pd.read_csv(fusion["analysis_master_long"])
    pre_qc = pd.read_csv(fusion["analysis_master_long_pre_qc"])
    qc = pd.read_csv(fusion["analysis_qc_exclusions"])
    kept_keys = set(zip(master["participant_id"], master["scene_id"]))
    reasons = qc.set_index(["participant_id", "scene_id"])["analysis_exclusion_reasons"]

    assert pre_qc[["participant_id", "scene_id"]].drop_duplicates().shape[0] == 3
    assert kept_keys == {("P01", 1)}
    assert "duration_mismatch" in reasons.loc[("P01", 2)]
    assert "scene_count_mismatch" in reasons.loc[("P02", 1)]


def test_fusion_qc_excludes_bad_eeg_quality(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene_manifest = tmp_path / "scene_manifest.csv"
    questionnaire = tmp_path / "questionnaire_long.csv"
    eye_metrics = tmp_path / "eye_aoi_trial_long.csv"
    eeg = tmp_path / "eeg_trial_long.csv"
    eye = tmp_path / "P01_scene01.csv"
    eye.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n0,1,1\n1000,1,1\n2000,1,1\n", encoding="utf-8")
    participants.write_text("participant_id,eeg_subject_id,eye_subject_id,exclude\nP01,P01,P01,false\n", encoding="utf-8")
    scene_manifest.write_text(
        "participant_id,scene_id,WWR,Complexity,block,position,eye_csv_path,aoi_json_path\n"
        f"P01,1,15,0,1,1,{eye.as_posix()},\n",
        encoding="utf-8",
    )
    questionnaire.write_text("participant_id,scene_id,S1\nP01,1,5\n", encoding="utf-8")
    eye_metrics.write_text(
        "participant_id,scene_id,class_name,FCR,TFD_ms,TTFF_ms,attention_share,visited\n"
        "P01,1,whole_scene,1,100,0,1,true\n",
        encoding="utf-8",
    )
    eeg.write_text(
        "participant_id,scene_id,view_dur_s,O_theta,bad_eeg_quality,eeg_qc_reasons,eeg_qc_policy\n"
        "P01,1,2,1.0,true,robust_hf_ratio_20_40Hz,robust\n",
        encoding="utf-8",
    )

    fusion = run_fusion_pipeline(
        questionnaire_long=questionnaire,
        eye_aoi_trial_long=eye_metrics,
        eeg_trial_long=eeg,
        participants_csv=participants,
        scene_manifest_csv=scene_manifest,
        outdir=tmp_path / "fusion_bad_eeg",
        expected_scenes_per_subject=1,
        duration_tolerance_s=2.0,
    )
    master = pd.read_csv(fusion["analysis_master_long"])
    qc = pd.read_csv(fusion["analysis_qc_exclusions"])
    assert master.empty
    assert bool(qc.loc[0, "bad_eeg_quality"]) is True
    assert "bad_eeg_quality" in qc.loc[0, "analysis_exclusion_reasons"]
    assert qc.loc[0, "eeg_qc_reasons"] == "robust_hf_ratio_20_40Hz"
