#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.alignment import run_alignment_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate eye-to-EEG time mapping and alignment QC.")
    parser.add_argument("--participants", default="manifests/generated/participants.csv")
    parser.add_argument("--scene_manifest", default="manifests/generated/scene_manifest.csv")
    parser.add_argument("--eeg_scene_csv", default="outputs/eeg/summary/all_subjects_scene_level.csv")
    parser.add_argument("--outdir", default="outputs/fusion")
    parser.add_argument("--columns_map", default=None)
    parser.add_argument("--duration_outlier_s", type=float, default=5.0)
    parser.add_argument("--residual_outlier_ms", type=float, default=3000.0)
    args = parser.parse_args()

    out = run_alignment_qc(
        participants_csv=args.participants,
        scene_manifest_csv=args.scene_manifest,
        eeg_scene_csv=args.eeg_scene_csv,
        outdir=args.outdir,
        columns_map=args.columns_map,
        duration_outlier_s=args.duration_outlier_s,
        residual_outlier_ms=args.residual_outlier_ms,
    )
    for name, path in out.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
