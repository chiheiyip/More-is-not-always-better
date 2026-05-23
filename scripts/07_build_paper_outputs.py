#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.reporting.pipeline import build_paper_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables and reviewer-response evidence indexes.")
    parser.add_argument("--model_results", default="outputs/06_models/model_results.csv")
    parser.add_argument("--diagnostics_dir", default="outputs/06_robustness")
    parser.add_argument("--reviewer_map", default="configs/reviewer_response_map.json")
    parser.add_argument("--data_availability_config", default="configs/data_availability.json")
    parser.add_argument("--figure_contracts_config", default="configs/figure_contracts.json")
    parser.add_argument("--outdir", default="outputs/07_paper_tables")
    args = parser.parse_args()
    for name, path in build_paper_outputs(
        model_results_csv=args.model_results,
        diagnostics_dir=args.diagnostics_dir,
        reviewer_map=args.reviewer_map,
        data_availability_config=args.data_availability_config,
        figure_contracts_config=args.figure_contracts_config,
        outdir=args.outdir,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
