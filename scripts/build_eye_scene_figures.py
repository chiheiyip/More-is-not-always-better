#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_analysis.teacher.eye_figures import build_eye_scene_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the registered six-condition eye-tracking manuscript figures."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--mapping-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manual-registration-file", type=Path)
    parser.add_argument("--expected-participants", type=int)
    parser.add_argument("--expected-trials", type=int)
    parser.add_argument("--expected-fixations", type=int)
    args = parser.parse_args()
    outputs = build_eye_scene_figures(
        run_root=args.run_root,
        mapping_file=args.mapping_file,
        output_dir=args.output_dir,
        manual_registration_file=args.manual_registration_file,
        expected_participants=args.expected_participants,
        expected_trials=args.expected_trials,
        expected_fixations=args.expected_fixations,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
