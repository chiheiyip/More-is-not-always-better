#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.eeg.contract import validate_eeg_scene_summary


DEFAULT_EEG_ROOT = r"E:\26\补\脑电数据"
DEFAULT_EEGLAB_ROOT = r"D:\Program Files\MATLAB\eeglab"
DEFAULT_OUTDIR = "outputs/eeg_realdata"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MATLAB/EEGLAB EEG exporter from raw .set/.fdt files, then validate the scene-level CSV.")
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT, help="Folder containing .set/.fdt files, or one .set file for a small test.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--eeglab_root", default=DEFAULT_EEGLAB_ROOT)
    parser.add_argument("--matlab_command", default="matlab")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    eeglab_root = Path(args.eeglab_root)
    outdir = Path(args.outdir)
    if not eeg_root.exists():
        raise SystemExit(f"EEG input not found: {eeg_root}")
    if not eeglab_root.exists():
        raise SystemExit(f"EEGLAB root not found: {eeglab_root}")

    matlab_expr = _matlab_expression(eeg_root, outdir, eeglab_root)
    cmd = [args.matlab_command, "-batch", matlab_expr]
    if args.dry_run:
        print(" ".join(cmd))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    summary_csv = outdir / "summary" / "all_subjects_scene_level.csv"
    result = validate_eeg_scene_summary(summary_csv)
    print(f"eeg_scene_csv: {summary_csv}")
    print(f"validation_status: {result['status']}")
    for err in result["errors"]:
        print(f"ERROR: {err}")
    for warn in result["warnings"]:
        print(f"WARNING: {warn}")
    if result["status"] == "error":
        raise SystemExit(1)


def _matlab_expression(eeg_root: Path, outdir: Path, eeglab_root: Path) -> str:
    eeglab = _matlab_path(eeglab_root)
    eeg = _matlab_path(eeg_root)
    out = _matlab_path(outdir)
    return "; ".join([
        f"addpath('{eeglab}')",
        "eeglab('nogui')",
        "addpath('matlab')",
        f"run_eeg_bandpower_from_set('{eeg}', '{out}')",
    ])


def _matlab_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


if __name__ == "__main__":
    main()
