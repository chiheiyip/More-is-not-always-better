#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.fusion.clock_sync import run_clock_synchronized_fusion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align eye rows to preprocessed EEG samples by absolute clock and build synchronized 2-second features."
    )
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--eeg-sample-manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bin-size-ms", type=int, default=2000)
    parser.add_argument("--match-tolerance-ms", type=int, default=2)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--no-pointwise", action="store_true")
    parser.add_argument("--no-timebins", action="store_true")
    args = parser.parse_args()
    outputs = run_clock_synchronized_fusion(
        scene_manifest_csv=args.scene_manifest,
        eeg_sample_manifest_csv=args.eeg_sample_manifest,
        outdir=args.outdir,
        bin_size_ms=args.bin_size_ms,
        match_tolerance_ms=args.match_tolerance_ms,
        timezone_name=args.timezone,
        export_pointwise=not args.no_pointwise,
        build_timebins=not args.no_timebins,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
