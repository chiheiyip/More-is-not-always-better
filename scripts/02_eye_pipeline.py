#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.eye_tracking.pipeline import run_eye_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute eye-tracking AOI metrics and AOI validation summaries.")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--scene_manifest", default="outputs/01_sample_qc/scene_manifest_standardized.csv")
    parser.add_argument("--outdir", default="outputs/03_eye_tracking")
    parser.add_argument("--point_source", default="auto", choices=["auto", "gaze", "fixation"])
    parser.add_argument("--screen_w", type=int, default=None)
    parser.add_argument("--screen_h", type=int, default=None)
    parser.add_argument("--validity_accepted", default=None, help="Comma-separated accepted validity values; omitted means audit only.")
    parser.add_argument("--timestamp_gap_ms", type=float, default=5000.0)
    args = parser.parse_args()
    validity_accepted = tuple(v.strip() for v in args.validity_accepted.split(",") if v.strip()) if args.validity_accepted else None
    for name, path in run_eye_pipeline(
        args.participants,
        args.scene_manifest,
        args.outdir,
        point_source=args.point_source,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        validity_accepted=validity_accepted,
        timestamp_gap_ms=args.timestamp_gap_ms,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
