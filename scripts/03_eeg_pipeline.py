#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.eeg.pipeline import run_eeg_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize EEG scene-level metrics and build EEG QC summaries.")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--scene_manifest", default="outputs/01_sample_qc/scene_manifest_standardized.csv")
    parser.add_argument("--eeg_scene_csv", required=True)
    parser.add_argument("--eeg_qc_config", default="configs/eeg_qc.json")
    parser.add_argument("--outdir", default="outputs/04_eeg")
    args = parser.parse_args()
    for name, path in run_eeg_pipeline(args.participants, args.scene_manifest, args.eeg_scene_csv, args.outdir, args.eeg_qc_config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
