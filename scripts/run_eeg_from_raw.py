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
    parser.add_argument("--eeg-clock-cache-root", default="", help="E-drive EEG clock cache created by build_eeg_clock_cache.py.")
    parser.add_argument("--export-eeg-samples", action="store_true", help="Export one preprocessed 500-Hz CSV per participant-scene.")
    parser.add_argument("--participants", default="", help="Comma-separated participant IDs for a bounded export.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    eeglab_root = Path(args.eeglab_root)
    outdir = Path(args.outdir)
    if not eeg_root.exists():
        raise SystemExit(f"EEG input not found: {eeg_root}")
    if not eeglab_root.exists():
        raise SystemExit(f"EEGLAB root not found: {eeglab_root}")

    if args.export_eeg_samples and not args.eeg_clock_cache_root:
        raise SystemExit("--eeg-clock-cache-root is required with --export-eeg-samples")
    if args.export_eeg_samples:
        recording_cache = Path(args.eeg_clock_cache_root) / "eeg_recording_clock.csv"
        if not recording_cache.exists():
            raise SystemExit(
                f"EEG clock cache not found: {recording_cache}. "
                "Run scripts/build_eeg_clock_cache.py first; routine export will not reread EASY."
            )
    matlab_expr = _matlab_expression(
        eeg_root,
        outdir,
        eeglab_root,
        clock_cache_root=Path(args.eeg_clock_cache_root) if args.eeg_clock_cache_root else None,
        export_samples=args.export_eeg_samples,
        participants=[v.strip() for v in args.participants.split(",") if v.strip()],
    )
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


def _matlab_expression(
    eeg_root: Path,
    outdir: Path,
    eeglab_root: Path,
    clock_cache_root: Path | None = None,
    export_samples: bool = False,
    participants: list[str] | None = None,
) -> str:
    eeglab = _matlab_utf8(_matlab_path(eeglab_root))
    eeg = _matlab_utf8(_matlab_path(eeg_root))
    out = _matlab_utf8(_matlab_path(outdir))
    args = [eeg, out]
    if export_samples:
        cache = _matlab_utf8(_matlab_path(clock_cache_root or Path()))
        args.extend(["'ClockCacheRoot'", cache, "'ExportSamples'", "true"])
    if participants:
        participant_expr = "[" + ",".join(
            f"string({_matlab_utf8(v)})" for v in participants
        ) + "]"
        args.extend(["'ParticipantFilter'", participant_expr])
    return "; ".join([
        f"addpath({eeglab})",
        "eeglab('nogui')",
        "addpath('matlab')",
        f"run_eeg_bandpower_from_set({', '.join(args)})",
    ])


def _matlab_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _matlab_utf8(value: str) -> str:
    numbers = " ".join(str(byte) for byte in value.encode("utf-8"))
    return f"native2unicode(uint8([{numbers}]),'UTF-8')"


if __name__ == "__main__":
    main()
