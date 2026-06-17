#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
from paper_analysis.eeg.contract import validate_eeg_scene_summary
from paper_analysis.eeg.pipeline import run_eeg_pipeline
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline
from paper_analysis.fusion.pipeline import run_fusion_pipeline
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
    parser.add_argument("--eeg_scene_csv", default=None, help="Optional scene-level EEG CSV; when provided, run EEG standardization and fusion.")
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
        summary_json = outdir / "linktest_summary.json"
        summary_csv = outdir / "linktest_summary.csv"

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

        eeg_outputs = None
        fusion_outputs = None
        eeg_contract = None
        if args.eeg_scene_csv:
            eeg_contract = validate_eeg_scene_summary(args.eeg_scene_csv)
            if eeg_contract["status"] == "error":
                raise SystemExit("EEG scene CSV failed contract validation: " + "; ".join(eeg_contract["errors"]))
            eeg_outputs = run_eeg_pipeline(
                participants_csv=intake["participants_standardized"],
                scene_manifest_csv=intake["scene_manifest_standardized"],
                eeg_scene_csv=args.eeg_scene_csv,
                outdir=outdir / "04_eeg",
            )
            fusion_outputs = run_fusion_pipeline(
                questionnaire_long=q["questionnaire_long"],
                eye_aoi_trial_long=eye["eye_aoi_trial_long"],
                eeg_trial_long=eeg_outputs["eeg_trial_long"],
                participants_csv=intake["participants_standardized"],
                scene_manifest_csv=intake["scene_manifest_standardized"],
                outdir=outdir / "05_multimodal_fusion",
                expected_scenes_per_subject=12,
            )

        summary = _build_summary(participants, scene, q, eye, eeg_outputs, fusion_outputs, eeg_contract)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

        print(f"participants: {len(participants)}")
        print(f"scene rows: {len(scene)}")
        print(f"questionnaire rows: {len(pd.read_csv(q['questionnaire_long']))}")
        print(f"eye metrics rows: {len(pd.read_csv(eye['eye_aoi_trial_long']))}")
        print(f"eeg scene csv provided: {bool(args.eeg_scene_csv)}")
        print(f"fusion run: {fusion_outputs is not None}")
        print(f"summary: {summary_json.resolve()}")
        print(f"scratch outdir: {outdir.resolve()}")
    finally:
        if args.cleanup and outdir.exists():
            shutil.rmtree(outdir)
            print(f"cleaned scratch outdir: {outdir.resolve()}")


def _build_summary(participants: pd.DataFrame, scene: pd.DataFrame, questionnaire: dict, eye: dict, eeg: dict | None, fusion: dict | None, eeg_contract: dict | None) -> dict:
    q_rows = len(pd.read_csv(questionnaire["questionnaire_long"]))
    eye_rows = len(pd.read_csv(eye["eye_aoi_trial_long"]))
    summary = {
        "participants": int(len(participants)),
        "participant_ids": ",".join(participants["participant_id"].astype(str).tolist()),
        "scene_rows": int(len(scene)),
        "questionnaire_rows": int(q_rows),
        "eye_metric_rows": int(eye_rows),
        "eeg_raw_present_all": bool(participants.get("has_eeg_raw", pd.Series(False, index=participants.index)).fillna(False).all()),
        "eeg_scene_csv_available": eeg is not None,
        "eeg_contract_status": eeg_contract.get("status") if eeg_contract else "not_provided",
        "fusion_run": fusion is not None,
    }
    if eeg is not None:
        summary["eeg_trial_rows"] = int(len(pd.read_csv(eeg["eeg_trial_long"])))
    if fusion is not None:
        qc = pd.read_csv(fusion["analysis_qc_exclusions"])
        summary["fusion_pre_qc_rows"] = int(len(pd.read_csv(fusion["analysis_master_long_pre_qc"])))
        summary["fusion_kept_rows"] = int(len(pd.read_csv(fusion["analysis_master_long"])))
        summary["fusion_excluded_trials"] = int(qc["excluded_from_analysis"].astype(str).str.lower().isin({"true", "1"}).sum()) if "excluded_from_analysis" in qc.columns else 0
    return summary


if __name__ == "__main__":
    main()
