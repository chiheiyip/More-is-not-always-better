#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.eeg.contract import validate_eeg_scene_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an EEG scene-level summary CSV against the Python pipeline contract.")
    parser.add_argument("eeg_scene_csv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate_eeg_scene_summary(args.eeg_scene_csv)
    if args.json:
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
        return
    print(f"EEG scene summary: {result['path']}")
    print(f"  status: {result['status']}")
    print(f"  rows: {result['rows']}")
    print(f"  duplicate_key_rows: {result['duplicate_key_rows']}")
    for err in result["errors"]:
        print(f"  ERROR: {err}")
    for warn in result["warnings"]:
        print(f"  WARNING: {warn}")
    if result["status"] == "error":
        raise SystemExit(1)


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
