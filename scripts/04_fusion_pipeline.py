#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.fusion.pipeline import run_fusion_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse questionnaire, eye-tracking, and EEG trial-level outputs.")
    parser.add_argument("--questionnaire_long", default="outputs/02_questionnaire/questionnaire_long.csv")
    parser.add_argument("--eye_aoi_trial_long", default="outputs/03_eye_tracking/eye_aoi_trial_long.csv")
    parser.add_argument("--eeg_trial_long", default="outputs/04_eeg/eeg_trial_long.csv")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--scene_manifest", default="outputs/01_sample_qc/scene_manifest_standardized.csv")
    parser.add_argument("--outdir", default="outputs/05_multimodal_fusion")
    parser.add_argument("--bin_size_ms", type=int, default=2000)
    parser.add_argument("--duration_tolerance_s", type=float, default=2.0)
    parser.add_argument("--expected_scenes_per_subject", type=int, default=12)
    parser.add_argument("--eye_point_source", default="auto", choices=["auto", "gaze", "fixation"])
    parser.add_argument("--eye_screen_w", type=int, default=None)
    parser.add_argument("--eye_screen_h", type=int, default=None)
    parser.add_argument("--eye_validity_accepted", default=None, help="Comma-separated accepted validity values for time-bin eye metrics; omitted means audit only.")
    parser.add_argument("--eye_timestamp_gap_ms", type=float, default=5000.0)
    args = parser.parse_args()
    eye_validity_accepted = tuple(v.strip() for v in args.eye_validity_accepted.split(",") if v.strip()) if args.eye_validity_accepted else None
    for name, path in run_fusion_pipeline(
        questionnaire_long=args.questionnaire_long,
        eye_aoi_trial_long=args.eye_aoi_trial_long,
        eeg_trial_long=args.eeg_trial_long,
        participants_csv=args.participants,
        scene_manifest_csv=args.scene_manifest,
        outdir=args.outdir,
        bin_size_ms=args.bin_size_ms,
        duration_tolerance_s=args.duration_tolerance_s,
        expected_scenes_per_subject=args.expected_scenes_per_subject,
        eye_point_source=args.eye_point_source,
        eye_screen_w=args.eye_screen_w,
        eye_screen_h=args.eye_screen_h,
        eye_validity_accepted=eye_validity_accepted,
        eye_timestamp_gap_ms=args.eye_timestamp_gap_ms,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
