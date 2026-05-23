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
    args = parser.parse_args()
    for name, path in run_eye_pipeline(args.participants, args.scene_manifest, args.outdir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
