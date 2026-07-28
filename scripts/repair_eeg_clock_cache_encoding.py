#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from more_is_not_always_better.discovery import _repair_legacy_mojibake
from paper_analysis.teacher.state import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a UTF-8, Chinese-identifier-safe copy of the EEG clock "
            "cache without changing the source cache."
        )
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    for source in sorted(source_root.glob("*.csv")):
        frame = pd.read_csv(source, encoding="utf-8-sig")
        repaired_cells = 0
        text_columns = [
            column for column in frame.columns
            if pd.api.types.is_object_dtype(frame[column].dtype)
            or pd.api.types.is_string_dtype(frame[column].dtype)
        ]
        for column in text_columns:
            original = frame[column].copy()
            frame[column] = frame[column].map(
                lambda value: (
                    value if pd.isna(value)
                    else _repair_legacy_mojibake(value)
                )
            )
            repaired_cells += int(
                original.fillna("").astype(str).ne(
                    frame[column].fillna("").astype(str)
                ).sum()
            )
        target = output_root / source.name
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        audit_rows.append({
            "Source": str(source),
            "SourceSHA256": file_sha256(source),
            "Output": str(target),
            "OutputSHA256": file_sha256(target),
            "Rows": len(frame),
            "RepairedCells": repaired_cells,
        })
    clock_path = output_root / "eeg_recording_clock.csv"
    if not clock_path.is_file():
        raise SystemExit("eeg_recording_clock.csv was not found in the input cache")
    clock = pd.read_csv(clock_path, encoding="utf-8-sig")
    if clock["participant_id"].duplicated().any():
        raise SystemExit("Repaired clock cache has duplicate participant_id rows")
    (output_root / "encoding_repair_audit.json").write_text(
        json.dumps(
            {
                "policy": (
                    "source cache preserved; only reversible legacy "
                    "latin1-to-GBK mojibake repair was applied"
                ),
                "files": audit_rows,
                "clock_participants": int(clock["participant_id"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Repaired {len(audit_rows)} cache files; "
        f"clock participants={clock['participant_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
