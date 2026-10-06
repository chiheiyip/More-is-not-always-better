"""Run a committed, independent eye/EEG audit without ICA special audit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_analysis.teacher.incremental_audit import load_config, preflight, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", choices=["all", "eye-process", "eye-models", "eye-boundary-models", "eye-seal-models", "eye-boundary-escalation", "eye-compare", "eeg-source", "eeg-current-models", "eeg-compare", "eeg-evidence", "compare"], default="all")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    plan = preflight(config, REPO)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    output = args.outdir or Path(plan["outdir"])
    run(config, args.config.resolve(), REPO, output, args.phase)
    print(json.dumps({"phase": args.phase, "outdir": str(output), "status": "completed"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
