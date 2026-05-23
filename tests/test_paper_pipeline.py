from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_analysis.diagnostics.pipeline import run_diagnostics
from paper_analysis.eeg.pipeline import run_eeg_pipeline
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline
from paper_analysis.fusion.pipeline import run_fusion_pipeline
from paper_analysis.intake.pipeline import build_manifests
from paper_analysis.questionnaire.pipeline import run_questionnaire_pipeline
from paper_analysis.reporting.pipeline import build_paper_outputs
from paper_analysis.stats.models import run_statistical_models


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "paper"


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

    master = pd.read_csv(fusion["analysis_master_long"])
    assert {"participant_id", "scene_id", "q_S1", "eeg_O_theta", "FCR", "attention_share"}.issubset(master.columns)
    assert pd.read_csv(fusion["aligned_scene"]).shape[0] >= 6
    assert {"bin_index", "bin_start_ms", "bin_end_ms", "eeg_O_theta"}.issubset(pd.read_csv(fusion["aligned_timebin"]).columns)
    assert {"duration_delta_s", "duration_mismatch"}.issubset(pd.read_csv(fusion["sync_qc"]).columns)
    assert {"time_sync_slope", "time_sync_offset_ms"}.issubset(pd.read_csv(fusion["time_sync_map"]).columns)
    assert pd.read_csv(intake["group_balance"])["factor"].isin(["ExperienceGroup"]).any()
    assert pd.read_csv(eye["aoi_validation_summary"])["class_name"].isin(["table", "window"]).any()
    assert pd.read_csv(stats["model_diagnostics"]).shape[0] > 0
    assert pd.read_csv(diagnostics["nonlinear_wwr_sensitivity"]).shape[0] > 0
    assert pd.read_csv(reporting["claim_strength_table"]).shape[0] > 0
    assert {"figure_id", "source_data", "review_risk"}.issubset(pd.read_csv(reporting["figure_contracts_index"]).columns)
    assert {"figure_id", "source_file"}.issubset(pd.read_csv(reporting["source_data_index"]).columns)
    assert {"issue_id", "response_readiness"}.issubset(pd.read_csv(reporting["reviewer_issue_matrix"]).columns)
    assert {"dataset_id", "access_route", "identifier"}.issubset(pd.read_csv(reporting["data_availability_index"]).columns)
    assert reporting["data_availability_statement"].exists()


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
    assert q[["participant_id", "scene_id"]].duplicated().sum() == 0
