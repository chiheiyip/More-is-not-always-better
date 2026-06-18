#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.diagnostics.pipeline import run_diagnostics
from paper_analysis.eeg.pipeline import run_eeg_pipeline
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline
from paper_analysis.figures import run_figure_pipeline
from paper_analysis.fusion.pipeline import run_fusion_pipeline
from paper_analysis.intake.pipeline import build_manifests
from paper_analysis.questionnaire.pipeline import run_questionnaire_pipeline
from paper_analysis.reporting.pipeline import build_paper_outputs
from paper_analysis.stats.models import run_statistical_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete paper-aligned analysis pipeline.")
    parser.add_argument("--config", default="configs/paths.example.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    outputs = Path(config.get("outputs_root", "outputs"))

    intake = build_manifests(
        participants_csv=config["participants"],
        scene_manifest_csv=config["scene_manifest"],
        outdir=outputs / "01_sample_qc",
    )
    questionnaire = run_questionnaire_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        questionnaire_wide=config.get("questionnaire_wide"),
        questionnaire_long=config.get("questionnaire_long"),
        outdir=outputs / "02_questionnaire",
        with_significance=config.get("questionnaire_significance", True),
        afford4_min_items=config.get("afford4_min_items", 3),
        wwr_levels=tuple(config.get("wwr_levels", [15.0, 45.0, 75.0])),
        skip_reliability=config.get("skip_questionnaire_reliability", False),
    )
    eye = run_eye_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        outdir=outputs / "03_eye_tracking",
    )
    eeg = run_eeg_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        eeg_scene_csv=config["eeg_scene_csv"],
        outdir=outputs / "04_eeg",
        eeg_qc_config=config.get("eeg_qc_config", "configs/eeg_qc.json"),
    )
    fusion = run_fusion_pipeline(
        questionnaire_long=questionnaire["questionnaire_long"],
        eye_aoi_trial_long=eye["eye_aoi_trial_long"],
        eeg_trial_long=eeg["eeg_trial_long"],
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        outdir=outputs / "05_multimodal_fusion",
        expected_scenes_per_subject=config.get("expected_scenes_per_subject", 12),
        bin_size_ms=config.get("bin_size_ms", 2000),
        duration_tolerance_s=config.get("duration_tolerance_s", 10.0),
    )
    stats = run_statistical_models(
        master_csv=fusion["analysis_master_long"],
        model_config="configs/model_families.json",
        outdir=outputs / "06_models",
    )
    diagnostics = run_diagnostics(
        master_csv=fusion["analysis_master_long"],
        participants_csv=intake["participants_standardized"],
        outdir=outputs / "06_robustness",
    )
    reporting = build_paper_outputs(
        model_results_csv=stats["model_results"],
        diagnostics_dir=outputs / "06_robustness",
        reviewer_map="configs/reviewer_response_map.json",
        outdir=outputs / "07_paper_tables",
    )
    figures = run_figure_pipeline(
        outputs_root=outputs,
        figure_contracts_config="configs/figure_contracts.json",
        outdir=outputs / "10_figures",
    )

    for group in [intake, questionnaire, eye, eeg, fusion, stats, diagnostics, reporting, figures]:
        for name, path in group.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
