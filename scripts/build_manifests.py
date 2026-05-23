#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Build participants and scene manifests from raw EEG/eye roots.")
    parser.add_argument("--eye_root", default=DEFAULT_EYE_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--participants_out", default="manifests/generated/participants.csv")
    parser.add_argument("--scene_manifest_out", default="manifests/generated/scene_manifest.csv")
    parser.add_argument("--participants_in", default=None, help="Use an existing participants CSV for Order/group fields.")
    parser.add_argument("--questionnaire_xlsx", default=DEFAULT_QUESTIONNAIRE)
    parser.add_argument("--no_questionnaire", action="store_true")
    parser.add_argument("--eye_alias_csv", default=None)
    parser.add_argument("--include_adaptation", action="store_true")
    parser.add_argument("--default_order", type=int, default=1, choices=[1, 2])
    parser.add_argument("--eye_offset_ms", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
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
        if key == "scene_folders":
            print(f"  {key}: {len(value)} folders")
        else:
            print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run only: no manifest files written.")
        return

    participants_csv = args.participants_in or args.participants_out
    if args.participants_in is None:
        participants = build_participants_from_roots(
            eye_root=args.eye_root,
            eeg_root=args.eeg_root,
            out_csv=args.participants_out,
            include_adaptation=args.include_adaptation,
            eye_alias_csv=args.eye_alias_csv,
            questionnaire_xlsx=questionnaire_xlsx,
        )
        print(f"participants: {args.participants_out} ({len(participants)} rows)")

    scene_manifest = build_scene_manifest_from_eye_root(
        eye_root=args.eye_root,
        participants_csv=participants_csv,
        out_csv=args.scene_manifest_out,
        include_adaptation=args.include_adaptation,
        default_order=args.default_order,
        eye_offset_ms=args.eye_offset_ms,
        eye_alias_csv=args.eye_alias_csv,
    )
    print(f"scene_manifest: {args.scene_manifest_out} ({len(scene_manifest)} rows)")


if __name__ == "__main__":
    main()
