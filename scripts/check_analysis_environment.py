"""Read-only runtime/AOI checks; explicit one-time freeze against formal evidence."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_analysis.teacher.runtime_lock import validate_analysis
from paper_analysis.teacher.aoi_lock import freeze_aoi, verify_aoi
from paper_analysis.teacher.state import StageBlockedError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--freeze-aoi", type=Path)
    parser.add_argument("--reference-area", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument('--scope', choices=['eye','eeg','joint','all'], default='all')
    args = parser.parse_args(argv)
    from run_teacher_analysis import _resolve_paths
    config = _resolve_paths(json.loads(args.config.read_text(encoding="utf-8-sig")), args.config.resolve())
    try:
        snapshot = validate_analysis(config, require_matlab=args.scope in {'eeg','all'})
        if args.freeze_aoi:
            if not args.reference_area or not args.reference_run:
                parser.error("--freeze-aoi requires --reference-area and --reference-run")
            snapshot["aoi"] = freeze_aoi(config, args.freeze_aoi, args.reference_area, args.reference_run)
        elif args.scope in {'eye','joint','all'}:
            snapshot["aoi"] = verify_aoi(config, required=True)
        snapshot['scope_readiness']={name:'passed' for name in
            (['eye','eeg','joint'] if args.scope=='all' else [args.scope])}
        snapshot['purpose']='environment readiness only; not a result reproducibility claim'
    except StageBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
