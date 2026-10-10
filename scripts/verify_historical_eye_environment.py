from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_analysis.teacher.historical_eye import prepare, run, compare
from paper_analysis.teacher.state import StageBlockedError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recalculate eye data in the locked 0805 environment, compare after freezing outputs")
    parser.add_argument("--phase", choices=["prepare", "run", "compare"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--historical-package", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    from run_teacher_analysis import _resolve_paths
    try:
        if args.phase == "prepare":
            if not args.config:
                parser.error("prepare requires --config")
            config = _resolve_paths(json.loads(args.config.read_text(encoding="utf-8-sig")), args.config.resolve())
            result = str(prepare(config, args.historical_package, args.outdir, REPO))
        elif args.phase == "run":
            run(args.outdir / "config.local.json", args.outdir, REPO)
            result = "fresh eye calculation complete"
        else:
            result = compare(args.historical_package, args.outdir)
    except StageBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
