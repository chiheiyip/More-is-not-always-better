#!/usr/bin/env python
"""Run the grain-safe analysis on a freshly rebuilt raw-data result package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.stats.optimized import run_optimized_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deprecated pre-canonical optimized analysis for compatibility only.")
    parser.add_argument("--base-results", required=True, help="Freshly rebuilt current-code result directory.")
    parser.add_argument("--outdir", required=True, help="Independent directory for optimized outputs.")
    parser.add_argument("--legacy-optimized", action="store_true", help="Required acknowledgement that this is not the canonical analysis.")
    args = parser.parse_args()
    if not args.legacy_optimized:
        parser.error("This compatibility entry is disabled by default; pass --legacy-optimized explicitly.")
    base = Path(args.base_results)
    outputs = run_optimized_analysis(
        pre_qc_master_csv=base / "05_multimodal_fusion" / "analysis_master_long_pre_qc.csv",
        qc_csv=base / "05_multimodal_fusion" / "analysis_qc_exclusions.csv",
        outdir=args.outdir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
