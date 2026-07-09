#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.reporting.experiment_results import build_experiment_result_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher-facing experiment result package from an existing outputs root.")
    parser.add_argument("--outputs_root", default="outputs", help="Existing analysis outputs root, e.g. E:\\26\\补\\数据分析结果.")
    parser.add_argument("--outdir", default=None, help="Defaults to <outputs_root>/07_paper_tables.")
    parser.add_argument("--audit_dir", default=None, help="Defaults to <outputs_root>/11_audit.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    outdir = Path(args.outdir) if args.outdir else outputs_root / "07_paper_tables"
    audit_dir = Path(args.audit_dir) if args.audit_dir else outputs_root / "11_audit"
    for name, path in build_experiment_result_package(outputs_root=outputs_root, outdir=outdir, audit_dir=audit_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
