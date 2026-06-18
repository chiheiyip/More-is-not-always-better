from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .aoi import (
    compute_timebin_metrics,
    compute_whole_scene_timebin_metrics,
    eye_file_stats,
    load_aoi_json,
)
from .columns import load_columns_map, rename_df_columns_inplace
from .filters import filter_by_screen_and_validity
from .io import (
    active_participants,
    assert_required_columns,
    assert_unique,
    load_participants,
    normalize_key_columns,
    read_csv,
    resolve_path,
    write_csv,
)


KEYS = ["participant_id", "scene_id"]


def run_fusion(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    eeg_scene_csv: str | Path,
    eye_aoi_class_csv: str | Path,
    outdir: str | Path = "outputs/fusion",
    columns_map: Optional[str | Path] = None,
    bin_size_ms: int = 2000,
    duration_tolerance_s: float = 10.0,
    expected_scenes_per_subject: int = 12,
    dwell_mode: str = "fixation",
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    require_validity: bool = False,
) -> dict[str, Path]:
    participants = active_participants(load_participants(participants_csv))
    manifest_path = Path(scene_manifest_csv)
    manifest_base = manifest_path.parent
    scene_manifest = load_scene_manifest(manifest_path, participants)
    eeg_scene = load_eeg_scene_table(eeg_scene_csv, participants)
    eye_class = load_eye_class_table(eye_aoi_class_csv)

    aligned_scene = build_aligned_scene_table(scene_manifest, participants, eeg_scene, eye_class)
    sync_qc = build_sync_qc(
        scene_manifest=scene_manifest,
        manifest_base=manifest_base,
        participants=participants,
        eeg_scene=eeg_scene,
        columns_map=columns_map,
        bin_size_ms=bin_size_ms,
        duration_tolerance_s=duration_tolerance_s,
        expected_scenes_per_subject=expected_scenes_per_subject,
        screen_w=screen_w,
        screen_h=screen_h,
        require_validity=require_validity,
    )
    aligned_timebin = build_aligned_timebin_table(
        scene_manifest=scene_manifest,
        manifest_base=manifest_base,
        participants=participants,
        eeg_scene=eeg_scene,
        columns_map=columns_map,
        bin_size_ms=bin_size_ms,
        dwell_mode=dwell_mode,
        screen_w=screen_w,
        screen_h=screen_h,
        require_validity=require_validity,
    )

    outdir = Path(outdir)
    return {
        "aligned_scene": write_csv(aligned_scene, outdir / "aligned_scene_table.csv"),
        "aligned_timebin": write_csv(aligned_timebin, outdir / "aligned_timebin_table.csv"),
        "sync_qc": write_csv(sync_qc, outdir / "sync_qc.csv"),
    }


def load_scene_manifest(path: str | Path, participants: pd.DataFrame) -> pd.DataFrame:
    manifest = read_csv(path)
    required = {"participant_id", "scene_id", "eye_csv_path"}
    assert_required_columns(manifest, required, "scene_manifest.csv")
    if "aoi_json_path" not in manifest.columns:
        manifest["aoi_json_path"] = ""
    manifest = manifest.copy()
    manifest["participant_id"] = manifest["participant_id"].astype(str).str.strip()
    manifest["scene_id"] = pd.to_numeric(manifest["scene_id"], errors="coerce").astype("Int64")
    if "eye_offset_ms" not in manifest.columns:
        manifest["eye_offset_ms"] = 0.0
    manifest["eye_offset_ms"] = pd.to_numeric(manifest["eye_offset_ms"], errors="coerce").fillna(0.0)
    assert_unique(manifest, KEYS, "scene_manifest.csv")
    active_ids = set(participants["participant_id"])
    return manifest.loc[manifest["participant_id"].isin(active_ids)].copy()


def load_eeg_scene_table(path: str | Path, participants: pd.DataFrame) -> pd.DataFrame:
    eeg = normalize_key_columns(read_csv(path))
    if "participant_id" not in eeg.columns:
        if "subject_id" not in eeg.columns:
            raise ValueError("EEG scene table must contain participant_id or subject_id")
        map_df = participants[["participant_id", "eeg_subject_id"]].rename(columns={"eeg_subject_id": "subject_id"})
        eeg = eeg.merge(map_df, on="subject_id", how="left")
    eeg = normalize_key_columns(eeg)
    if eeg["participant_id"].isna().any():
        missing = eeg.loc[eeg["participant_id"].isna()].head(10).to_dict("records")
        raise ValueError(f"EEG subject_id values not found in participants.csv. Sample: {missing}")
    assert_required_columns(eeg, KEYS, "EEG scene table")
    assert_unique(eeg, KEYS, "EEG scene table")
    return eeg


def load_eye_class_table(path: str | Path) -> pd.DataFrame:
    eye = normalize_key_columns(read_csv(path))
    required = {"participant_id", "scene_id", "class_name"}
    assert_required_columns(eye, required, "eye AOI class table")
    assert_unique(eye, ["participant_id", "scene_id", "class_name"], "eye AOI class table")
    return eye


def build_aligned_scene_table(
    scene_manifest: pd.DataFrame,
    participants: pd.DataFrame,
    eeg_scene: pd.DataFrame,
    eye_class: pd.DataFrame,
) -> pd.DataFrame:
    scene_base = scene_manifest.merge(participants, on="participant_id", how="left", suffixes=("", "_participant"))
    eye_plus = eye_class.merge(scene_base, on=KEYS, how="left", suffixes=("_eye", ""))
    out = eye_plus.merge(eeg_scene, on=KEYS, how="left", suffixes=("", "_eeg"))
    out["missing_eeg_scene"] = out[_first_existing(out, ["subject_id", "view_dur_s", "dur_s", "duration_s"])].isna() if not out.empty else pd.Series(dtype=bool)
    sort_cols = [c for c in ["participant_id", "scene_id", "class_name"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def build_sync_qc(
    scene_manifest: pd.DataFrame,
    manifest_base: str | Path,
    participants: pd.DataFrame,
    eeg_scene: pd.DataFrame,
    columns_map: Optional[str | Path],
    bin_size_ms: int,
    duration_tolerance_s: float,
    expected_scenes_per_subject: int,
    screen_w: Optional[int],
    screen_h: Optional[int],
    require_validity: bool,
) -> pd.DataFrame:
    cmap = load_columns_map(columns_map)
    eeg_duration = _eeg_duration_table(eeg_scene)
    scene_counts = scene_manifest.groupby("participant_id")["scene_id"].nunique().rename("manifest_scene_count")
    rows: list[dict] = []

    for _, row in scene_manifest.iterrows():
        participant_id = str(row["participant_id"])
        scene_id = int(row["scene_id"])
        eye_csv = resolve_path(row["eye_csv_path"], manifest_base)
        eye_offset_ms = _number_or_default(row.get("eye_offset_ms"), 0.0)
        base = {
            "participant_id": participant_id,
            "scene_id": scene_id,
            "eye_csv_path": str(eye_csv) if eye_csv else "",
            "eye_offset_ms": eye_offset_ms,
            "missing_eye_file": not eye_csv or not eye_csv.exists(),
            "missing_eeg_scene": not ((eeg_duration["participant_id"] == participant_id) & (eeg_duration["scene_id"] == scene_id)).any(),
            "manifest_scene_count": int(scene_counts.get(participant_id, 0)),
            "expected_scenes_per_subject": int(expected_scenes_per_subject),
        }
        base["scene_count_mismatch"] = base["manifest_scene_count"] != int(expected_scenes_per_subject)

        if base["missing_eye_file"]:
            eye_stats = {}
        else:
            df = read_csv(eye_csv)
            rename_df_columns_inplace(df, cmap)
            df = filter_by_screen_and_validity(df, screen_w, screen_h, require_validity)
            eye_stats = eye_file_stats(df, eye_offset_ms=eye_offset_ms, bin_size_ms=bin_size_ms)

        eeg_row = eeg_duration.loc[(eeg_duration["participant_id"] == participant_id) & (eeg_duration["scene_id"] == scene_id)]
        eeg_view_dur_s = float(eeg_row["eeg_view_dur_s"].iloc[0]) if not eeg_row.empty else np.nan
        eye_duration_s = eye_stats.get("eye_duration_s", np.nan)
        delta = eye_duration_s - eeg_view_dur_s if pd.notna(eye_duration_s) and pd.notna(eeg_view_dur_s) else np.nan
        rows.append({
            **base,
            **eye_stats,
            "eeg_view_dur_s": eeg_view_dur_s,
            "duration_delta_s": delta,
            "duration_mismatch": bool(abs(delta) > duration_tolerance_s) if pd.notna(delta) else True,
        })
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def build_aligned_timebin_table(
    scene_manifest: pd.DataFrame,
    manifest_base: str | Path,
    participants: pd.DataFrame,
    eeg_scene: pd.DataFrame,
    columns_map: Optional[str | Path],
    bin_size_ms: int,
    dwell_mode: str,
    screen_w: Optional[int],
    screen_h: Optional[int],
    require_validity: bool,
) -> pd.DataFrame:
    cmap = load_columns_map(columns_map)
    rows: list[pd.DataFrame] = []
    eeg_prefixed = _prefix_non_key_columns(eeg_scene, "eeg")
    scene_plus = scene_manifest.merge(participants, on="participant_id", how="left", suffixes=("", "_participant"))

    for _, row in scene_plus.iterrows():
        eye_csv = resolve_path(row["eye_csv_path"], manifest_base)
        aoi_json = resolve_path(row.get("aoi_json_path"), manifest_base)
        if not eye_csv or not eye_csv.exists():
            continue
        df = read_csv(eye_csv)
        rename_df_columns_inplace(df, cmap)
        df = filter_by_screen_and_validity(df, screen_w, screen_h, require_validity)
        if aoi_json and aoi_json.exists():
            aois = load_aoi_json(aoi_json)
            timebin = compute_timebin_metrics(
                df,
                aois,
                bin_size_ms=bin_size_ms,
                eye_offset_ms=_number_or_default(row.get("eye_offset_ms"), 0.0),
                dwell_mode=dwell_mode,
            )
        else:
            timebin = compute_whole_scene_timebin_metrics(
                df,
                bin_size_ms=bin_size_ms,
                eye_offset_ms=_number_or_default(row.get("eye_offset_ms"), 0.0),
                dwell_mode=dwell_mode,
            )
        if timebin.empty:
            continue
        timebin.insert(0, "scene_id", int(row["scene_id"]))
        timebin.insert(0, "participant_id", str(row["participant_id"]))
        for col in ["scene_name", "block", "position", "WWR", "Cond", "Complexity", "SportFreq", "Experience", "Order"]:
            if col in row.index and col not in timebin.columns:
                timebin[col] = row[col]
        rows.append(timebin)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out = out.merge(eeg_prefixed, on=KEYS, how="left")
    return out.sort_values(["participant_id", "scene_id", "bin_index", "class_name"]).reset_index(drop=True)


def _eeg_duration_table(eeg_scene: pd.DataFrame) -> pd.DataFrame:
    out = eeg_scene[KEYS].copy()
    out["eeg_view_dur_s"] = _infer_eeg_duration_s(eeg_scene)
    return out


def _infer_eeg_duration_s(eeg_scene: pd.DataFrame) -> pd.Series:
    for col in ["eeg_view_dur_s", "view_dur_s", "view_duration_s", "dur_s", "duration_s"]:
        if col in eeg_scene.columns:
            return pd.to_numeric(eeg_scene[col], errors="coerce")
    for start_col, end_col in [("view_start_s", "view_end_s"), ("start_s", "end_s")]:
        if {start_col, end_col}.issubset(eeg_scene.columns):
            start = pd.to_numeric(eeg_scene[start_col], errors="coerce")
            end = pd.to_numeric(eeg_scene[end_col], errors="coerce")
            return end - start
    return pd.Series(np.nan, index=eeg_scene.index)


def _prefix_non_key_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {
        col: f"{prefix}_{col}"
        for col in df.columns
        if col not in KEYS and not col.startswith(f"{prefix}_")
    }
    return df.rename(columns=rename)


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) else ""


def _number_or_default(value: object, default: float) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default
