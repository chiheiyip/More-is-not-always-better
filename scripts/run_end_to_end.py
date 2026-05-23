#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.discovery import (
    build_participants_from_roots,
    build_scene_manifest_from_eye_root,
    summarize_roots,
)


DEFAULT_EYE_ROOT = "E:\\2.7\u773c\u52a8\u6570\u636e\\\u6620\u5c04"
DEFAULT_EEG_ROOT = "E:\\eeg\u539f\u59cb\u6587\u4ef6"
DEFAULT_QUESTIONNAIRE = "E:\\VR+EEG\u5b9e\u9a8c\u95ee\u5377-\u6587\u672c\u7248-2026-02-17.xlsx"


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end EEG + eye fusion runner.")
    parser.add_argument("--eye_root", default=DEFAULT_EYE_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--participants", default="manifests/generated/participants.csv")
    parser.add_argument("--scene_manifest", default="manifests/generated/scene_manifest.csv")
    parser.add_argument("--eeg_outdir", default="outputs/eeg")
    parser.add_argument("--eye_outdir", default="outputs/eye")
    parser.add_argument("--fusion_outdir", default="outputs/fusion")
    parser.add_argument("--matlab_command", default="matlab")
    parser.add_argument("--skip_eeg", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--eye_alias_csv", default=None)
    parser.add_argument("--questionnaire_xlsx", default=DEFAULT_QUESTIONNAIRE)
    parser.add_argument("--no_questionnaire", action="store_true")
    args = parser.parse_args()

    questionnaire_xlsx = None if args.no_questionnaire else args.questionnaire_xlsx
    summary = summarize_roots(
        args.eye_root,
        args.eeg_root,
        eye_alias_csv=args.eye_alias_csv,
        questionnaire_xlsx=questionnaire_xlsx,
    )
    print("Raw data summary:")
    for key, value in summary.items():
        if key != "scene_folders":
            print(f"  {key}: {value}")

    matlab_call = (
        "addpath('matlab'); "
        f"run_eeg_bandpower_from_set('{_matlab_path(args.eeg_root)}', "
        f"'{_matlab_path(args.eeg_outdir)}'); exit"
    )
    build_cmd = [
        sys.executable,
        "scripts/build_manifests.py",
        "--eye_root",
        args.eye_root,
        "--eeg_root",
        args.eeg_root,
        "--participants_out",
        args.participants,
        "--scene_manifest_out",
        args.scene_manifest,
    ]
    if args.eye_alias_csv:
        build_cmd.extend(["--eye_alias_csv", args.eye_alias_csv])
    if args.no_questionnaire:
        build_cmd.append("--no_questionnaire")
    elif args.questionnaire_xlsx:
        build_cmd.extend(["--questionnaire_xlsx", args.questionnaire_xlsx])

    commands = [
        build_cmd,
        [
            sys.executable,
            "scripts/run_eye_aoi_batch.py",
            "--participants",
            args.participants,
            "--scene_manifest",
            args.scene_manifest,
            "--outdir",
            args.eye_outdir,
        ],
    ]
    if not args.skip_eeg:
        commands.append([args.matlab_command, "-batch", matlab_call])
    commands.append([
        sys.executable,
        "scripts/run_fusion.py",
        "--participants",
        args.participants,
        "--scene_manifest",
        args.scene_manifest,
        "--eeg_scene_csv",
        str(Path(args.eeg_outdir) / "summary" / "all_subjects_scene_level.csv"),
        "--eye_aoi_class_csv",
        str(Path(args.eye_outdir) / "batch_aoi_metrics_by_class.csv"),
        "--outdir",
        args.fusion_outdir,
    ])
    commands.append([
        sys.executable,
        "scripts/run_alignment_qc.py",
        "--participants",
        args.participants,
        "--scene_manifest",
        args.scene_manifest,
        "--eeg_scene_csv",
        str(Path(args.eeg_outdir) / "summary" / "all_subjects_scene_level.csv"),
        "--outdir",
        args.fusion_outdir,
    ])

    if args.dry_run:
        print("Dry run command plan:")
        for cmd in commands:
            print("  " + " ".join(cmd))
        return

    build_participants_from_roots(
        args.eye_root,
        args.eeg_root,
        out_csv=args.participants,
        eye_alias_csv=args.eye_alias_csv,
        questionnaire_xlsx=questionnaire_xlsx,
    )
    build_scene_manifest_from_eye_root(args.eye_root, args.participants, out_csv=args.scene_manifest, eye_alias_csv=args.eye_alias_csv)
    for cmd in commands[1:]:
        subprocess.run(cmd, check=True)


def _matlab_path(path: str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


if __name__ == "__main__":
    main()
