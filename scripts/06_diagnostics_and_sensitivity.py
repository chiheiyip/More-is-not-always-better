#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.diagnostics.pipeline import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-driven diagnostics and sensitivity checks.")
    parser.add_argument("--master", default="outputs/05_multimodal_fusion/analysis_master_long.csv")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--outdir", default="outputs/06_robustness")
    args = parser.parse_args()
    for name, path in run_diagnostics(args.master, args.participants, args.outdir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
