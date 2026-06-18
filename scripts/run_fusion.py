#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.fusion import run_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EEG + eye-tracking fusion outputs.")
    parser.add_argument("--participants", default="manifests/participants.csv")
    parser.add_argument("--scene_manifest", default="manifests/scene_manifest.csv")
    parser.add_argument("--eeg_scene_csv", default="outputs/eeg/summary/all_subjects_scene_level.csv")
    parser.add_argument("--eye_aoi_class_csv", default="outputs/eye/batch_aoi_metrics_by_class.csv")
    parser.add_argument("--outdir", default="outputs/fusion")
    parser.add_argument("--columns_map", default=None)
    parser.add_argument("--bin_size_ms", type=int, default=2000)
    parser.add_argument("--duration_tolerance_s", type=float, default=10.0)
    parser.add_argument("--expected_scenes_per_subject", type=int, default=12)
    parser.add_argument("--dwell_mode", default="fixation", choices=["row", "fixation"])
    parser.add_argument("--screen_w", type=int, default=None)
    parser.add_argument("--screen_h", type=int, default=None)
    parser.add_argument("--require_validity", action="store_true")
    args = parser.parse_args()

    out = run_fusion(
        participants_csv=args.participants,
        scene_manifest_csv=args.scene_manifest,
        eeg_scene_csv=args.eeg_scene_csv,
        eye_aoi_class_csv=args.eye_aoi_class_csv,
        outdir=args.outdir,
        columns_map=args.columns_map,
        bin_size_ms=args.bin_size_ms,
        duration_tolerance_s=args.duration_tolerance_s,
        expected_scenes_per_subject=args.expected_scenes_per_subject,
        dwell_mode=args.dwell_mode,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        require_validity=args.require_validity,
    )
    for name, path in out.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
