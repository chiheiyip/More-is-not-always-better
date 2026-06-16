#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.eye_batch import run_eye_aoi_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute batch eye-tracking AOI metrics.")
    parser.add_argument("--participants", default="manifests/participants.csv")
    parser.add_argument("--scene_manifest", default="manifests/scene_manifest.csv")
    parser.add_argument("--outdir", default="outputs/eye")
    parser.add_argument("--columns_map", default=None)
    parser.add_argument("--dwell_mode", default="fixation", choices=["row", "fixation"])
    parser.add_argument("--point_source", default="auto", choices=["auto", "gaze", "fixation"])
    parser.add_argument("--screen_w", type=int, default=None)
    parser.add_argument("--screen_h", type=int, default=None)
    parser.add_argument("--require_validity", action="store_true")
    parser.add_argument("--validity_accepted", default=None, help="Comma-separated accepted validity values; used only with --require_validity.")
    args = parser.parse_args()
    validity_accepted = tuple(v.strip() for v in args.validity_accepted.split(",") if v.strip()) if args.validity_accepted else None

    out = run_eye_aoi_batch(
        participants_csv=args.participants,
        scene_manifest_csv=args.scene_manifest,
        outdir=args.outdir,
        columns_map=args.columns_map,
        dwell_mode=args.dwell_mode,
        point_source=args.point_source,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        require_validity=args.require_validity,
        validity_accepted=validity_accepted,
    )
    for name, path in out.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
