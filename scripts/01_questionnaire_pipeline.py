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
    args = parser.parse_args()
    for name, path in run_questionnaire_pipeline(
        participants_csv=args.participants,
        scene_manifest_csv=args.scene_manifest,
        questionnaire_wide=args.questionnaire_wide,
        questionnaire_long=args.questionnaire_long,
        outdir=args.outdir,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
