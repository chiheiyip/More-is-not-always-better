#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.questionnaire.pipeline import run_questionnaire_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform and summarize questionnaire data.")
    parser.add_argument("--participants", default="outputs/01_sample_qc/participants_standardized.csv")
    parser.add_argument("--scene_manifest", default="outputs/01_sample_qc/scene_manifest_standardized.csv")
    parser.add_argument("--questionnaire_wide", default=None)
    parser.add_argument("--questionnaire_long", default=None)
    parser.add_argument("--outdir", default="outputs/02_questionnaire")
    parser.add_argument("--with-significance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--afford4-min-items", type=int, default=3)
    parser.add_argument("--wwr-levels", default="15,45,75", help="Comma-separated WWR levels for linear/quadratic trend contrasts.")
    parser.add_argument("--skip-reliability", action="store_true")
    args = parser.parse_args()
    wwr_levels = tuple(float(v.strip()) for v in args.wwr_levels.split(",") if v.strip())
    if len(wwr_levels) != 3:
        raise SystemExit("--wwr-levels must contain exactly three comma-separated levels")
    for name, path in run_questionnaire_pipeline(
        participants_csv=args.participants,
        scene_manifest_csv=args.scene_manifest,
        questionnaire_wide=args.questionnaire_wide,
        questionnaire_long=args.questionnaire_long,
        outdir=args.outdir,
        with_significance=args.with_significance,
        afford4_min_items=args.afford4_min_items,
        wwr_levels=wwr_levels,
        skip_reliability=args.skip_reliability,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
