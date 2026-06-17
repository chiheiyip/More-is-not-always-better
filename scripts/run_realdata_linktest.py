#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.discovery import (
    build_participants_from_roots,
    build_scene_manifest_from_eye_root,
    load_questionnaire_long_from_wjx,
)
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline
from paper_analysis.intake.pipeline import build_manifests
from paper_analysis.questionnaire.pipeline import run_questionnaire_pipeline


DEFAULT_QUESTIONNAIRE = r"E:\26\补\VR+EEG实验问卷-补-原始数据-2026-06-14.xlsx"
DEFAULT_EYE_ROOT = r"E:\26\补\眼动数据"
DEFAULT_EEG_ROOT = r"E:\26\补\脑电数据"
DEFAULT_OUTDIR = "outputs/_scratch_realdata_linktest"


def main() -> None:
    parser = argparse.ArgumentParser(description="Small read-only raw-data link test with isolated scratch outputs.")
    parser.add_argument("--questionnaire_xlsx", default=DEFAULT_QUESTIONNAIRE)
    parser.add_argument("--eye_root", default=DEFAULT_EYE_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--participants", default="", help="Comma-separated participant names to include.")
    parser.add_argument("--max-participants", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Remove the scratch outdir after a successful run.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    selected = [p.strip() for p in args.participants.split(",") if p.strip()]
    if args.dry_run:
        print("Dry run only. Would write isolated scratch outputs to:", outdir)
        print("Raw input directories are read-only and will not be modified.")
        return

    try:
        outdir.mkdir(parents=True, exist_ok=True)
        participants_csv = outdir / "participants_raw.csv"
        scene_csv = outdir / "scene_manifest_raw.csv"
        q_long_csv = outdir / "questionnaire_long_raw.csv"

        participants = build_participants_from_roots(
            eye_root=args.eye_root,
            eeg_root=args.eeg_root,
            questionnaire_xlsx=args.questionnaire_xlsx,
        )
        if selected:
            participants = participants.loc[participants["participant_id"].isin(selected)].copy()
        else:
            matched = participants.loc[
                participants["has_eeg_raw"].fillna(False)
                & participants["has_eye_raw"].fillna(False)
                & participants["has_questionnaire"].fillna(False),
                "participant_id",
            ].head(args.max_participants)
            participants = participants.loc[participants["participant_id"].isin(set(matched))].copy()
        if participants.empty:
            raise SystemExit("No participants available for link test after filtering.")
        participants["exclude"] = False
        participants.to_csv(participants_csv, index=False, encoding="utf-8-sig")

        scene = build_scene_manifest_from_eye_root(args.eye_root, participants_csv)
        scene.to_csv(scene_csv, index=False, encoding="utf-8-sig")

        q_long = load_questionnaire_long_from_wjx(args.questionnaire_xlsx, participants=participants["participant_id"].tolist())
        q_long.to_csv(q_long_csv, index=False, encoding="utf-8-sig")

        intake = build_manifests(participants_csv, scene_csv, outdir / "01_sample_qc")
        q = run_questionnaire_pipeline(
            participants_csv=intake["participants_standardized"],
            scene_manifest_csv=intake["scene_manifest_standardized"],
            questionnaire_long=q_long_csv,
            outdir=outdir / "02_questionnaire",
            with_significance=False,
            skip_reliability=True,
        )
        eye = run_eye_pipeline(
            participants_csv=intake["participants_standardized"],
            scene_manifest_csv=intake["scene_manifest_standardized"],
            outdir=outdir / "03_eye_tracking",
        )

        print(f"participants: {len(participants)}")
        print(f"scene rows: {len(scene)}")
        print(f"questionnaire rows: {len(pd.read_csv(q['questionnaire_long']))}")
        print(f"eye metrics rows: {len(pd.read_csv(eye['eye_aoi_trial_long']))}")
        print(f"scratch outdir: {outdir.resolve()}")
    finally:
        if args.cleanup and outdir.exists():
            shutil.rmtree(outdir)
            print(f"cleaned scratch outdir: {outdir.resolve()}")


if __name__ == "__main__":
    main()
