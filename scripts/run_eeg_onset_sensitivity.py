#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.stats.onset_sensitivity import run_onset_sensitivity_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EEG onset-window model, QC and equivalence analyses.")
    parser.add_argument("--sensitivity-trials", default="outputs/04_eeg/eeg_onset_sensitivity_trial_long.csv")
    parser.add_argument("--common-qc", default="outputs/04_eeg/eeg_onset_common_qc.csv")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--scene-manifest", default="outputs/01_sample_qc/scene_manifest_standardized.csv")
    parser.add_argument("--eeg-analysis-config", default="configs/eeg_analysis.json")
    parser.add_argument("--outdir", default="outputs/06_robustness/eeg_onset")
    parser.add_argument(
        "--reuse-equivalence",
        help=(
            "Optional validated historical 10-vs-15 equivalence CSV. This "
            "compatibility-only artifact is copied with explicit provenance."
        ),
    )
    args = parser.parse_args()
    outputs = run_onset_sensitivity_analysis(
        args.sensitivity_trials, args.common_qc, args.participants,
        args.scene_manifest, args.outdir, args.eeg_analysis_config,
        reuse_equivalence_csv=args.reuse_equivalence,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
