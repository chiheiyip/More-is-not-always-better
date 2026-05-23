#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.stats.models import run_statistical_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-aligned statistical model families.")
    parser.add_argument("--master", default="outputs/05_multimodal_fusion/analysis_master_long.csv")
    parser.add_argument("--model_config", default="configs/model_families.json")
    parser.add_argument("--outdir", default="outputs/06_models")
    args = parser.parse_args()
    for name, path in run_statistical_models(args.master, args.model_config, args.outdir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
