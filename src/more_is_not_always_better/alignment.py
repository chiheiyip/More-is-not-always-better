from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .columns import load_columns_map, rename_df_columns_inplace
from .io import active_participants, load_participants, read_csv, resolve_path, write_csv


KEYS = ["participant_id", "scene_id"]


def run_alignment_qc(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    eeg_scene_csv: str | Path,
    outdir: str | Path = "outputs/fusion",
    columns_map: Optional[str | Path] = None,
    duration_outlier_s: float = 5.0,
    residual_outlier_ms: float = 3000.0,
) -> dict[str, Path]:
    participants = active_participants(load_participants(participants_csv))
    eye_timing = build_eye_timing_table(scene_manifest_csv, columns_map=columns_map)
    eeg_timing = load_eeg_timing(eeg_scene_csv, participants)
    scene_qc, landmarks, sync_map = estimate_time_alignment(
        eye_timing,
        eeg_timing,
        duration_outlier_s=duration_outlier_s,
        residual_outlier_ms=residual_outlier_ms,
    )
    outdir = Path(outdir)
    return {
        "scene_qc": write_csv(scene_qc, outdir / "alignment_scene_qc.csv"),
        "landmarks": write_csv(landmarks, outdir / "alignment_landmarks.csv"),
        "sync_map": write_csv(sync_map, outdir / "time_sync_map.csv"),
    }


def build_eye_timing_table(
    scene_manifest_csv: str | Path,
    columns_map: Optional[str | Path] = None,
) -> pd.DataFrame:
    manifest_path = Path(scene_manifest_csv)
    manifest = read_csv(manifest_path)
    cmap = load_columns_map(columns_map)
    rows: list[dict] = []
    for _, row in manifest.iterrows():
        eye_csv = resolve_path(row.get("eye_csv_path"), manifest_path.parent)
        base = {
            "participant_id": str(row["participant_id"]).strip(),
            "scene_id": int(row["scene_id"]),
            "eye_csv_path": str(eye_csv) if eye_csv else "",
            "missing_eye_file": not eye_csv or not eye_csv.exists(),
        }
        for col in ["participant_order", "scene_name", "Cond", "WWR", "Complexity", "source_folder"]:
            if col in row.index:
                base[col] = row[col]
        if base["missing_eye_file"]:
            rows.append(base)
            continue
        df = read_csv(eye_csv)
        rename_df_columns_inplace(df, cmap)
        if "Recording Time Stamp[ms]" not in df.columns:
            rows.append({**base, "missing_eye_timestamp": True})
            continue
        t = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce").dropna()
        if t.empty:
            rows.append({**base, "missing_eye_timestamp": True})
            continue
        rows.append({
            **base,
            "missing_eye_timestamp": False,
            "eye_start_ms": float(t.min()),
            "eye_end_ms": float(t.max()),
            "eye_duration_s": float((t.max() - t.min()) / 1000.0),
            "eye_sample_count": int(len(t)),
        })
    return pd.DataFrame(rows)


def load_eeg_timing(eeg_scene_csv: str | Path, participants: pd.DataFrame) -> pd.DataFrame:
    eeg = read_csv(eeg_scene_csv)
    if "participant_id" not in eeg.columns:
        if "subject_id" not in eeg.columns:
            raise ValueError("EEG timing table must contain participant_id or subject_id")
        mapping = participants[["participant_id", "eeg_subject_id"]].rename(columns={"eeg_subject_id": "subject_id"})
        eeg = eeg.merge(mapping, on="subject_id", how="left")
    required = {"participant_id", "scene_id", "view_start_s", "view_end_s"}
    missing = required - set(eeg.columns)
    if missing:
        raise ValueError(f"EEG timing table missing columns for precise alignment: {sorted(missing)}")
    out = eeg.copy()
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["scene_id"] = pd.to_numeric(out["scene_id"], errors="coerce").astype("Int64")
    out["view_start_s"] = pd.to_numeric(out["view_start_s"], errors="coerce")
    out["view_end_s"] = pd.to_numeric(out["view_end_s"], errors="coerce")
    if "view_dur_s" not in out.columns:
        out["view_dur_s"] = out["view_end_s"] - out["view_start_s"]
    else:
        out["view_dur_s"] = pd.to_numeric(out["view_dur_s"], errors="coerce")
    return out[list(dict.fromkeys(KEYS + ["view_start_s", "view_end_s", "view_dur_s"]))]


def estimate_time_alignment(
    eye_timing: pd.DataFrame,
    eeg_timing: pd.DataFrame,
    duration_outlier_s: float = 5.0,
    residual_outlier_ms: float = 3000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scene_qc = eye_timing.merge(eeg_timing, on=KEYS, how="left")
    scene_qc["duration_delta_s"] = scene_qc["eye_duration_s"] - scene_qc["view_dur_s"]
    scene_qc["duration_outlier"] = scene_qc["duration_delta_s"].abs() > duration_outlier_s

    landmark_rows: list[dict] = []
    map_rows: list[dict] = []
    for participant_id, sub in scene_qc.dropna(subset=["eye_start_ms", "eye_end_ms", "view_start_s", "view_end_s"]).groupby("participant_id"):
        trial_ok = ~sub["duration_outlier"].fillna(True)
        fit_sub = sub.loc[trial_ok].copy()
        if len(fit_sub) < 4:
            fit_sub = sub.copy()

        x = np.r_[fit_sub["eye_start_ms"].to_numpy(), fit_sub["eye_end_ms"].to_numpy()]
        y = np.r_[fit_sub["view_start_s"].to_numpy() * 1000.0, fit_sub["view_end_s"].to_numpy() * 1000.0]
        slope, offset = _fit_affine(x, y)
        residual = (offset + slope * x) - y
        keep = np.abs(residual) <= residual_outlier_ms
        if keep.sum() >= 8:
            slope, offset = _fit_affine(x[keep], y[keep])

        for _, row in sub.iterrows():
            for kind, eye_col, eeg_col in [
                ("start", "eye_start_ms", "view_start_s"),
                ("end", "eye_end_ms", "view_end_s"),
            ]:
                eye_ms = row[eye_col]
                eeg_ms = row[eeg_col] * 1000.0
                pred = offset + slope * eye_ms
                landmark_rows.append({
                    "participant_id": participant_id,
                    "scene_id": int(row["scene_id"]),
                    "landmark": kind,
                    "eye_time_ms": eye_ms,
                    "eeg_time_ms": eeg_ms,
                    "predicted_eeg_time_ms": pred,
                    "residual_ms": pred - eeg_ms,
                    "duration_outlier": bool(row.get("duration_outlier", False)),
                })

        participant_landmarks = pd.DataFrame([r for r in landmark_rows if r["participant_id"] == participant_id])
        usable = participant_landmarks.loc[~participant_landmarks["duration_outlier"]]
        if usable.empty:
            usable = participant_landmarks
        map_rows.append({
            "participant_id": participant_id,
            "time_sync_slope": slope,
            "time_sync_offset_ms": offset,
            "landmark_count": int(len(participant_landmarks)),
            "usable_landmark_count": int(len(usable)),
            "median_abs_residual_ms": float(usable["residual_ms"].abs().median()),
            "p95_abs_residual_ms": float(usable["residual_ms"].abs().quantile(0.95)),
            "max_abs_residual_ms": float(usable["residual_ms"].abs().max()),
            "duration_outlier_count": int(sub["duration_outlier"].sum()),
        })

    landmarks = pd.DataFrame(landmark_rows)
    sync_map = pd.DataFrame(map_rows)
    return scene_qc, landmarks, sync_map


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    slope, offset = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(offset)
