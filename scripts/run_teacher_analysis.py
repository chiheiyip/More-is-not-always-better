from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_analysis.teacher.eeg import run_eeg_order, run_eeg_primary
from paper_analysis.teacher.eye import (
    run_eye_stage1,
    run_eye_stage2,
    run_eye_stage3,
    run_eye_stage3_plan,
)
from paper_analysis.teacher.state import StageBlockedError


STAGE_FOLDERS = {
    "eye-stage1": "01_eye_stage1",
    "eye-stage2": "02_eye_stage2",
    "eye-stage3-plan": "03_eye_stage3_plan",
    "eye-stage3-run": "04_eye_stage3",
    "eeg-order": "05_eeg_order",
    "eeg-primary": "06_eeg_primary",
}


def _resolve_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.parent
    result = json.loads(json.dumps(config))
    keys = {
        "outputs_root", "participant_information", "trial_order_mapping",
        "scene_aoi_mapping", "raw_root", "base_images_root", "stage1_dir",
        "aoi_root",
        "stage2_dir", "stage3_plan_dir", "trial_file",
        "preprocessing_audit_file", "order_stage_dir",
        "s3_trial_file",
    }

    def walk(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, key) for v in value]
        if key in keys and isinstance(value, str) and value.strip():
            path = Path(value)
            return str(path if path.is_absolute() else (base / path).resolve())
        return value

    return walk(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teacher-priority eye-tracking and EEG analysis workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in STAGE_FOLDERS:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--outputs-root", type=Path)
        child.add_argument("--run-id")
        child.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _resolve_paths(
        json.loads(config_path.read_text(encoding="utf-8")), config_path
    )
    outputs_root = (
        args.outputs_root.resolve()
        if args.outputs_root
        else Path(config.get("outputs_root", REPO_ROOT / "outputs"))
    )
    run_id = (
        args.run_id
        or config.get("run_id")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = outputs_root / "teacher_runs" / str(run_id)
    outdir = run_root / STAGE_FOLDERS[args.command]
    config.setdefault("eye", {})
    config["eye"]["stage1_dir"] = (
        config["eye"].get("stage1_dir") or str(run_root / STAGE_FOLDERS["eye-stage1"])
    )
    config["eye"]["stage2_dir"] = (
        config["eye"].get("stage2_dir") or str(run_root / STAGE_FOLDERS["eye-stage2"])
    )
    config["eye"]["stage3_plan_dir"] = (
        config["eye"].get("stage3_plan_dir")
        or str(run_root / STAGE_FOLDERS["eye-stage3-plan"])
    )
    config.setdefault("eeg", {})
    config["eeg"]["order_stage_dir"] = (
        config["eeg"].get("order_stage_dir")
        or str(run_root / STAGE_FOLDERS["eeg-order"])
    )
    plan = {
        "command": args.command,
        "config": str(config_path),
        "outputs_root": str(outputs_root),
        "run_id": run_id,
        "outdir": str(outdir),
        "formal_r_inference_required": args.command != "eye-stage1"
        and args.command != "eye-stage3-plan",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    functions = {
        "eye-stage1": run_eye_stage1,
        "eye-stage2": run_eye_stage2,
        "eye-stage3-plan": run_eye_stage3_plan,
        "eye-stage3-run": run_eye_stage3,
        "eeg-order": run_eeg_order,
        "eeg-primary": run_eeg_primary,
    }
    try:
        functions[args.command](
            config,
            config_path=config_path,
            outdir=outdir,
            repo_root=REPO_ROOT,
        )
    except StageBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        print(json.dumps(plan, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({**plan, "status": "complete"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
