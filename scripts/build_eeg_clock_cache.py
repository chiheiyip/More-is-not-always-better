#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.eeg.clock_cache import build_eeg_clock_cache


DEFAULT_ACQUISITION_ROOT = r"D:\AAA所有应用\暂存\1.31"
DEFAULT_EEG_ROOT = r"E:\26\补\脑电数据"
DEFAULT_CACHE_ROOT = r"E:\26\补\脑电数据\eeg_clock_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable EEG absolute-clock and trigger cache from one-time EASY/INFO access.")
    parser.add_argument("--acquisition-root", default=DEFAULT_ACQUISITION_ROOT)
    parser.add_argument("--eeg-root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--participants", default="", help="Comma-separated participant IDs for a bounded cache build.")
    parser.add_argument("--matlab-command", default="matlab")
    parser.add_argument("--set-metadata-csv", default="", help="Use pre-exported set metadata; mainly for tests.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    acquisition_root = Path(args.acquisition_root)
    eeg_root = Path(args.eeg_root)
    cache_root = Path(args.cache_root)
    if not acquisition_root.exists():
        raise SystemExit(f"EEG acquisition root not found: {acquisition_root}")
    if not eeg_root.exists():
        raise SystemExit(f"EEG .set/.fdt root not found: {eeg_root}")

    if args.dry_run:
        print(f"acquisition_root: {acquisition_root}")
        print(f"eeg_root: {eeg_root}")
        print(f"cache_root: {cache_root}")
        print("raw_inputs_read_only: true")
        return

    if args.set_metadata_csv:
        metadata_csv = Path(args.set_metadata_csv)
        outputs = _build(args, metadata_csv)
    else:
        cache_root.mkdir(parents=True, exist_ok=True)
        metadata_csv = cache_root / "eeg_set_metadata.csv"
        _export_set_metadata(args.matlab_command, eeg_root, metadata_csv)
        outputs = _build(args, metadata_csv)
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _build(args: argparse.Namespace, metadata_csv: Path) -> dict[str, Path]:
    participants = [v.strip() for v in args.participants.split(",") if v.strip()]
    return build_eeg_clock_cache(
        set_metadata_csv=metadata_csv,
        acquisition_root=args.acquisition_root,
        cache_root=args.cache_root,
        timezone_name=args.timezone,
        participants=participants,
        allow_partial=args.allow_partial,
    )


def _export_set_metadata(matlab_command: str, eeg_root: Path, output_csv: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    matlab_dir = repo_root / "matlab" / "eeg_bandpower_pipeline"
    expr = "; ".join([
        f"addpath({_matlab_utf8(_matlab_path(matlab_dir))})",
        f"export_eeg_set_metadata({_matlab_utf8(_matlab_path(eeg_root))},{_matlab_utf8(_matlab_path(output_csv))})",
    ])
    subprocess.run([matlab_command, "-batch", expr], check=True, cwd=repo_root)
    if not output_csv.exists():
        raise RuntimeError(f"MATLAB did not create set metadata: {output_csv}")


def _matlab_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _matlab_utf8(value: str) -> str:
    numbers = " ".join(str(byte) for byte in value.encode("utf-8"))
    return f"native2unicode(uint8([{numbers}]),'UTF-8')"


if __name__ == "__main__":
    main()
