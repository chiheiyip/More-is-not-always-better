#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.figures import run_figure_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nature-style manuscript figures, source data, and figure QA.")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--figure_contracts", default="configs/figure_contracts.json")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    for name, path in run_figure_pipeline(args.outputs_root, args.figure_contracts, args.outdir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
