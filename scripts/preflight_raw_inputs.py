#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.discovery import summarize_roots


DEFAULT_QUESTIONNAIRE = r"E:\26\补\VR+EEG实验问卷-补-原始数据-2026-06-14.xlsx"
DEFAULT_EYE_ROOT = r"E:\26\补\眼动数据"
DEFAULT_EEG_ROOT = r"E:\26\补\脑电数据"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only preflight for questionnaire, eye-tracking, and EEG raw entries.")
    parser.add_argument("--questionnaire_xlsx", default=DEFAULT_QUESTIONNAIRE)
    parser.add_argument("--eye_root", default=DEFAULT_EYE_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--eye_alias_csv", default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    summary = summarize_roots(
        eye_root=args.eye_root,
        eeg_root=args.eeg_root,
        eye_alias_csv=args.eye_alias_csv,
        questionnaire_xlsx=args.questionnaire_xlsx,
    )
    if args.json:
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
        return

    print("Raw data preflight (read-only)")
    print(f"  questionnaire_xlsx: {args.questionnaire_xlsx}")
    print(f"  eye_root: {args.eye_root}")
    print(f"  eeg_root: {args.eeg_root}")
    for key in [
        "questionnaire_subject_count",
        "eye_csv_count",
        "eye_subject_count_after_alias",
        "eye_suffix_note_alias_rows",
        "eye_scene_folder_count",
        "eye_aoi_json_count",
        "eye_missing_aoi_json_rows",
        "eeg_set_count",
        "eeg_fdt_count",
        "matched_subject_count",
        "trimodal_subject_count",
    ]:
        print(f"  {key}: {summary.get(key)}")
    print(f"  questionnaire_order_counts: {summary.get('questionnaire_order_counts')}")
    for key in ["eye_only_subjects", "eeg_only_subjects", "questionnaire_missing_for_eye_subjects"]:
        values = summary.get(key, [])
        preview = values[:10]
        suffix = " ..." if len(values) > 10 else ""
        print(f"  {key}: {preview}{suffix}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


if __name__ == "__main__":
    main()
