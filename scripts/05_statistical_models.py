#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.stats.canonical import run_canonical_analysis
from paper_analysis.stats.models import run_statistical_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical modality-specific statistical analysis.")
    parser.add_argument("--master", default="outputs/05_multimodal_fusion/analysis_master_long.csv")
    parser.add_argument("--model_config", default="configs/model_families.json")
    parser.add_argument("--outdir", default="outputs/06_models")
    parser.add_argument("--questionnaire", default="outputs/02_questionnaire/questionnaire_long.csv")
    parser.add_argument("--eye-aoi", default="outputs/03_eye_tracking/eye_aoi_trial_long.csv")
    parser.add_argument("--eye-dynamic", default="outputs/03_eye_tracking/eye_trial_dynamic_metrics.csv")
    parser.add_argument("--eye-qc", default="outputs/03_eye_tracking/eye_qc.csv")
    parser.add_argument("--eeg", default="outputs/04_eeg/eeg_trial_long.csv")
    parser.add_argument("--eeg-qc", default="outputs/04_eeg/eeg_scene_qc.csv")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--mde-simulations", type=int, default=1000)
    parser.add_argument("--legacy-models", action="store_true")
    args = parser.parse_args()
    outputs = run_canonical_analysis(
        args.questionnaire, args.eye_aoi, args.eye_dynamic, args.eye_qc,
        args.eeg, args.eeg_qc, args.participants, args.outdir,
        mde_simulations=args.mde_simulations,
    )
    if args.legacy_models:
        legacy = run_statistical_models(args.master, args.model_config, Path(args.outdir) / "legacy")
        outputs.update({f"legacy_{name}": path for name, path in legacy.items()})
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
