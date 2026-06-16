from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .aoi import compute_metrics, compute_whole_scene_metrics, eye_file_stats, load_aoi_json
from .columns import load_columns_map, rename_df_columns_inplace
from .filters import filter_by_screen_and_validity
from .io import (
    active_participants,
    assert_required_columns,
    assert_unique,
    load_participants,
    read_csv,
    resolve_path,
    write_csv,
)


def run_eye_aoi_batch(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path = "outputs/eye",
    columns_map: Optional[str | Path] = None,
    dwell_mode: str = "fixation",
    point_source: str = "auto",
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    require_validity: bool = False,
    validity_accepted: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    participants = active_participants(load_participants(participants_csv))
    manifest_path = Path(scene_manifest_csv)
    manifest_base = manifest_path.parent
    scene_manifest = read_csv(manifest_path)
    required = {"participant_id", "scene_id", "eye_csv_path"}
    assert_required_columns(scene_manifest, required, "scene_manifest.csv")
    if "aoi_json_path" not in scene_manifest.columns:
        scene_manifest["aoi_json_path"] = ""
    scene_manifest = scene_manifest.copy()
    scene_manifest["participant_id"] = scene_manifest["participant_id"].astype(str).str.strip()
    scene_manifest["scene_id"] = pd.to_numeric(scene_manifest["scene_id"], errors="coerce").astype("Int64")
    assert_unique(scene_manifest, ["participant_id", "scene_id"], "scene_manifest.csv")
    scene_manifest = scene_manifest.merge(participants, on="participant_id", how="inner", suffixes=("", "_participant"))

    cmap = load_columns_map(columns_map)
    poly_rows: list[pd.DataFrame] = []
    class_rows: list[pd.DataFrame] = []
    qc_rows: list[dict] = []

    for _, row in scene_manifest.iterrows():
        participant_id = str(row["participant_id"])
        scene_id = int(row["scene_id"])
        eye_csv = resolve_path(row["eye_csv_path"], manifest_base)
        aoi_json = resolve_path(row["aoi_json_path"], manifest_base)
        eye_offset_ms = _number_or_default(row.get("eye_offset_ms"), 0.0)

        qc_base = {
            "participant_id": participant_id,
            "scene_id": scene_id,
            "eye_csv_path": str(eye_csv) if eye_csv else "",
            "aoi_json_path": str(aoi_json) if aoi_json else "",
            "missing_eye_file": not eye_csv or not eye_csv.exists(),
            "missing_aoi_file": not aoi_json or not aoi_json.exists(),
        }
        if qc_base["missing_eye_file"]:
            qc_rows.append(qc_base)
            continue

        df = read_csv(eye_csv)
        rename_df_columns_inplace(df, cmap)
        x_col, y_col = _point_columns_for_filter(df, point_source)
        df = filter_by_screen_and_validity(
            df,
            screen_w,
            screen_h,
            require_validity,
            x_col=x_col,
            y_col=y_col,
            validity_accepted=validity_accepted,
        )
        if qc_base["missing_aoi_file"]:
            poly_df = pd.DataFrame()
            class_df = compute_whole_scene_metrics(df, dwell_mode=dwell_mode, point_source=point_source)
        else:
            aois = load_aoi_json(aoi_json)
            poly_df, class_df = compute_metrics(df, aois, dwell_mode=dwell_mode, point_source=point_source)

        for output_df in (poly_df, class_df):
            if output_df.empty:
                continue
            output_df.insert(0, "scene_id", scene_id)
            output_df.insert(0, "participant_id", participant_id)
            _attach_optional_manifest_columns(output_df, row)

        poly_rows.append(poly_df)
        class_rows.append(class_df)
        stats = eye_file_stats(df, eye_offset_ms=eye_offset_ms)
        qc_rows.append({**qc_base, **stats, "eye_offset_ms": eye_offset_ms})

    outdir = Path(outdir)
    out = {
        "polygon": write_csv(
            pd.concat(poly_rows, ignore_index=True) if poly_rows else pd.DataFrame(),
            outdir / "batch_aoi_metrics_by_polygon.csv",
        ),
        "class": write_csv(
            pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame(),
            outdir / "batch_aoi_metrics_by_class.csv",
        ),
        "qc": write_csv(pd.DataFrame(qc_rows), outdir / "eye_scene_qc.csv"),
    }
    return out


def _attach_optional_manifest_columns(df: pd.DataFrame, row: pd.Series) -> None:
    for col in ["scene_name", "block", "position", "WWR", "Cond", "Complexity", "SportFreq", "Experience", "Order"]:
        if col in row.index and col not in df.columns:
            df[col] = row[col]


def _number_or_default(value: object, default: float) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def _point_columns_for_filter(df: pd.DataFrame, point_source: str) -> tuple[str, str]:
    point_source = str(point_source or "auto").strip().lower()
    fixation = ("Fixation Point X[px]", "Fixation Point Y[px]")
    gaze = ("Gaze Point X[px]", "Gaze Point Y[px]")
    if point_source in {"auto", "fixation"} and set(fixation).issubset(df.columns):
        return fixation
    return gaze
