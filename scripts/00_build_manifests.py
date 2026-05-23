#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.intake.pipeline import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize participants and scene manifests, and build sample/design QC outputs.")
    parser.add_argument("--participants", default="manifests/participants.csv")
    parser.add_argument("--scene_manifest", default="manifests/scene_manifest.csv")
    parser.add_argument("--outdir", default="outputs/01_sample_qc")
    args = parser.parse_args()
    for name, path in build_manifests(args.participants, args.scene_manifest, args.outdir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
