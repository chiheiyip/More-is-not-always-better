#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.eeg.contract import validate_eeg_scene_summary
from paper_analysis.eeg.onset import load_eeg_analysis_config


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
    parser.add_argument("--eeg-analysis-config", default="configs/eeg_analysis.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    eeglab_root = Path(args.eeglab_root)
    outdir = Path(args.outdir)
    analysis_config = load_eeg_analysis_config(args.eeg_analysis_config)
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
        config_path=Path(args.eeg_analysis_config).resolve(),
    )
    cmd = [args.matlab_command, "-batch", matlab_expr]
    if args.dry_run:
        print(
            "EEG onset trim: primary="
            f"{analysis_config['primary_onset_trim_s']:g}s variants="
            + ",".join(f"{value:g}s" for value in analysis_config["onset_trim_variants_s"])
        )
        print(" ".join(cmd))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    summary_csv = outdir / "summary" / "all_subjects_scene_level.csv"
    result = validate_eeg_scene_summary(
        summary_csv,
        expected_onset_trim_s=analysis_config["primary_onset_trim_s"],
        require_onset_metadata=True,
    )
    sensitivity_csv = outdir / "summary" / "all_subjects_scene_level_onset_sensitivity.csv"
    sensitivity_result = validate_eeg_scene_summary(
        sensitivity_csv, require_onset_metadata=True, sensitivity=True
    )
    print(f"eeg_scene_csv: {summary_csv}")
    print(f"validation_status: {result['status']}")
    for err in result["errors"]:
        print(f"ERROR: {err}")
    for warn in result["warnings"]:
        print(f"WARNING: {warn}")
    for err in sensitivity_result["errors"]:
        print(f"ERROR: onset sensitivity: {err}")
    if result["status"] == "error" or sensitivity_result["status"] == "error":
        raise SystemExit(1)
    if args.export_eeg_samples:
        _write_sample_export_audit(
            eeg_root=eeg_root,
            cache_root=Path(args.eeg_clock_cache_root),
            outdir=outdir,
        )


def _matlab_expression(
    eeg_root: Path,
    outdir: Path,
    eeglab_root: Path,
    clock_cache_root: Path | None = None,
    export_samples: bool = False,
    participants: list[str] | None = None,
    config_path: Path | None = None,
) -> str:
    eeglab = _matlab_utf8(_matlab_path(eeglab_root))
    eeg = _matlab_utf8(_matlab_path(eeg_root))
    out = _matlab_utf8(_matlab_path(outdir))
    args = [eeg, out]
    if config_path is not None:
        args.extend(["'ConfigPath'", _matlab_utf8(_matlab_path(config_path))])
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


def _write_sample_export_audit(
    *,
    eeg_root: Path,
    cache_root: Path,
    outdir: Path,
) -> None:
    manifest_path = outdir / "summary" / "eeg_sample_file_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit("Sample export was requested but its manifest is missing")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    cache = pd.read_csv(
        cache_root / "eeg_recording_clock.csv", encoding="utf-8-sig"
    )
    set_files = [eeg_root] if eeg_root.is_file() else sorted(eeg_root.glob("*.set"))
    participants = [path.stem for path in set_files]
    rows = []
    for participant in participants:
        sub = manifest.loc[
            manifest["participant_id"].astype(str).eq(participant)
        ]
        cache_rows = int(
            cache["participant_id"].astype(str).eq(participant).sum()
        )
        scene_count = int(sub["scene_id"].nunique()) if not sub.empty else 0
        if cache_rows == 0:
            status = "excluded"
            reason = "no_validated_clock_cache_row"
        elif scene_count != 12:
            status = "failed"
            reason = f"expected_12_sample_files_found_{scene_count}"
        else:
            status = "included"
            reason = ""
        rows.append({
            "Participant": participant,
            "ValidatedClockCacheRows": cache_rows,
            "ExportedSceneFiles": scene_count,
            "Status": status,
            "ExclusionOrFailureReason": reason,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(
        outdir / "summary" / "eeg_sample_export_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failed = audit["Status"].eq("failed")
    if failed.any():
        raise SystemExit(
            "EEG sample export incomplete for cached participants: "
            + ", ".join(audit.loc[failed, "Participant"].astype(str))
        )
    print(
        "sample_export_audit: "
        f"included={audit['Status'].eq('included').sum()} "
        f"excluded_no_clock={audit['Status'].eq('excluded').sum()}"
    )


if __name__ == "__main__":
    main()
