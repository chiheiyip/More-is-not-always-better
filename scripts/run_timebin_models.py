#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.stats.timebin import run_timebin_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit co-primary participant-clustered GEE models to synchronized 2-second eye/EEG features."
    )
    parser.add_argument("--synchronized-timebins", required=True)
    parser.add_argument("--clock-scene-qc", required=True)
    parser.add_argument("--outdir", default="outputs/06_models")
    parser.add_argument("--scene-model-results", default=None)
    args = parser.parse_args()
    outputs = run_timebin_models(
        synchronized_timebin_csv=args.synchronized_timebins,
        clock_scene_qc_csv=args.clock_scene_qc,
        outdir=args.outdir,
        scene_model_results_csv=args.scene_model_results,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
