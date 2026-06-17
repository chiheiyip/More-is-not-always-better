from __future__ import annotations

from pathlib import Path

import pandas as pd


ID_COLUMN_OPTIONS = ("participant_id", "subject_id")
REQUIRED_COLUMNS = ("scene_id",)
DURATION_COLUMN_OPTIONS = ("view_dur_s", "duration_s", "dur_s", "view_duration_s", "eeg_view_dur_s")
TIMING_COLUMNS = ("view_start_s", "view_end_s")
CORE_METRIC_OPTIONS = ("O_theta", "F_theta", "O_alpha")
QC_RECOMMENDED_COLUMNS = (
    "hf_ratio_20_40Hz",
    "rms_mean_uV",
    "peak_to_peak_uV",
    "nan_fraction",
    "flat_fraction",
    "segment_valid_duration",
)


def validate_eeg_scene_summary(path: str | Path) -> dict:
    eeg_path = Path(path)
    if not eeg_path.exists():
        return {
            "path": str(eeg_path),
            "status": "error",
            "errors": [f"EEG scene summary not found: {eeg_path}"],
            "warnings": [],
            "rows": 0,
            "columns": [],
        }
    df = pd.read_csv(eeg_path, encoding="utf-8-sig")
    return validate_eeg_scene_summary_frame(df, path=eeg_path)


def validate_eeg_scene_summary_frame(df: pd.DataFrame, path: str | Path | None = None) -> dict:
    columns = set(df.columns)
    errors: list[str] = []
    warnings: list[str] = []
    if not any(col in columns for col in ID_COLUMN_OPTIONS):
        errors.append("missing participant identifier: provide participant_id or subject_id")
    for col in REQUIRED_COLUMNS:
        if col not in columns:
            errors.append(f"missing required column: {col}")
    if not any(col in columns for col in DURATION_COLUMN_OPTIONS):
        warnings.append("missing duration column: recommended one of " + ",".join(DURATION_COLUMN_OPTIONS))
    missing_timing = [col for col in TIMING_COLUMNS if col not in columns]
    if missing_timing:
        warnings.append("missing precise timing columns: " + ",".join(missing_timing))
    if not any(col in columns for col in CORE_METRIC_OPTIONS):
        errors.append("missing core EEG metric: provide at least one of " + ",".join(CORE_METRIC_OPTIONS))
    missing_qc = [col for col in QC_RECOMMENDED_COLUMNS if col not in columns]
    if missing_qc:
        warnings.append("missing recommended QC columns: " + ",".join(missing_qc))
    duplicate_count = 0
    id_col = "participant_id" if "participant_id" in columns else "subject_id" if "subject_id" in columns else None
    if id_col and "scene_id" in columns:
        duplicate_count = int(df.duplicated([id_col, "scene_id"]).sum())
        if duplicate_count:
            errors.append(f"duplicate {id_col}+scene_id rows: {duplicate_count}")
    status = "error" if errors else "warning" if warnings else "pass"
    return {
        "path": str(path) if path is not None else "<dataframe>",
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "columns": sorted(df.columns.tolist()),
        "duplicate_key_rows": duplicate_count,
    }
