#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.diagnostics.pipeline import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-driven diagnostics and sensitivity checks.")
    parser.add_argument("--master", default="outputs/05_multimodal_fusion/analysis_master_long.csv")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--outdir", default="outputs/06_robustness")
    parser.add_argument("--questionnaire", default="outputs/02_questionnaire/questionnaire_long.csv")
    parser.add_argument("--eye-dynamic", default="outputs/03_eye_tracking/eye_trial_dynamic_metrics.csv")
    parser.add_argument("--eeg", default="outputs/04_eeg/eeg_trial_long.csv")
    parser.add_argument("--eeg-qc", default="outputs/04_eeg/eeg_scene_qc.csv")
    parser.add_argument("--scene-manifest", default="outputs/01_sample_qc/scene_manifest_standardized.csv")
    parser.add_argument("--model-results", default="outputs/06_models/model_results.csv")
    parser.add_argument("--model-diagnostics", default="outputs/06_models/model_diagnostics.csv")
    args = parser.parse_args()
    for name, path in run_diagnostics(
        args.master,
        args.participants,
        args.outdir,
        questionnaire_csv=args.questionnaire,
        eye_dynamic_csv=args.eye_dynamic,
        eeg_csv=args.eeg,
        eeg_scene_qc_csv=args.eeg_qc,
        scene_manifest_csv=args.scene_manifest,
        model_results_csv=args.model_results,
        model_diagnostics_csv=args.model_diagnostics,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
