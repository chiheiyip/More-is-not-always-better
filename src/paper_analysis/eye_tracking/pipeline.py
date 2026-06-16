from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_analysis.eye_tracking.aoi import aoi_validation, compute_aoi_metrics, compute_whole_scene_metrics, load_aoi_json
from paper_analysis.utils.io import read_table, resolve_path, require_columns, write_table


def run_eye_pipeline(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path = "outputs/03_eye_tracking",
    point_source: str = "auto",
    screen_w: int | None = None,
    screen_h: int | None = None,
    validity_accepted: tuple[str, ...] | None = None,
    timestamp_gap_ms: float = 5000.0,
) -> dict[str, Path]:
    participants = read_table(participants_csv)
    scene = read_table(scene_manifest_csv)
    require_columns(scene, ["participant_id", "scene_id", "eye_csv_path"], "scene_manifest")
    exclude = participants["exclude"] if "exclude" in participants.columns else pd.Series(False, index=participants.index)
    active_ids = set(participants.loc[~exclude.astype(str).str.lower().isin({"true", "1", "yes", "y"}), "participant_id"].astype(str))
    scene = scene.loc[scene["participant_id"].astype(str).isin(active_ids)].copy()
    manifest_base = Path(scene_manifest_csv).parent

    class_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    overlap_rows: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    for _, row in scene.iterrows():
        eye_csv = resolve_path(row.get("eye_csv_path"), manifest_base)
        aoi_json = resolve_path(row.get("aoi_json_path"), manifest_base)
        base = _trial_base(row)
        if not eye_csv or not eye_csv.exists():
            qc_rows.append({**base, "missing_eye_file": True, "missing_aoi_file": not aoi_json or not aoi_json.exists()})
            continue
        df = read_table(eye_csv)
        if aoi_json and aoi_json.exists():
            aois = load_aoi_json(aoi_json)
            _, metrics = compute_aoi_metrics(
                df,
                aois,
                point_source=point_source,
                screen_w=screen_w,
                screen_h=screen_h,
                validity_accepted=validity_accepted,
                timestamp_gap_ms=timestamp_gap_ms,
            )
            validation = aoi_validation(aois, metrics)
        else:
            metrics = compute_whole_scene_metrics(
                df,
                point_source=point_source,
                screen_w=screen_w,
                screen_h=screen_h,
                validity_accepted=validity_accepted,
                timestamp_gap_ms=timestamp_gap_ms,
            )
            validation = pd.DataFrame([{"class_name": "whole_scene", "polygon_id": 0, "polygon_area_px2": None, "visited_rate": 1.0}])
        for frame in (metrics, validation):
            for key, value in base.items():
                frame[key] = value
        overlap = pd.DataFrame(metrics.attrs.get("aoi_overlap", []))
        if not overlap.empty:
            for key, value in base.items():
                overlap[key] = value
            overlap_rows.append(overlap)
        class_rows.append(metrics)
        validation_rows.append(validation)
        qc_rows.append({
            **base,
            **metrics.attrs.get("eye_qc", {}),
            "missing_eye_file": False,
            "missing_aoi_file": not aoi_json or not aoi_json.exists(),
            "eye_sample_count": len(df),
        })

    outdir = Path(outdir)
    eye_long = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    overlap = pd.concat(overlap_rows, ignore_index=True) if overlap_rows else pd.DataFrame()
    return {
        "eye_aoi_trial_long": write_table(eye_long, outdir / "eye_aoi_trial_long.csv"),
        "aoi_validation_summary": write_table(validation, outdir / "aoi_validation_summary.csv"),
        "aoi_overlap_summary": write_table(overlap, outdir / "aoi_overlap_summary.csv"),
        "eye_qc": write_table(pd.DataFrame(qc_rows), outdir / "eye_qc.csv"),
    }


def _trial_base(row: pd.Series) -> dict:
    return {
        "participant_id": str(row["participant_id"]).strip(),
        "scene_id": int(row["scene_id"]),
        "WWR": row.get("WWR"),
        "Complexity": row.get("Complexity"),
        "Cond": row.get("Cond"),
        "block": row.get("block"),
        "position": row.get("position"),
        "round": row.get("round"),
        "condition_id": row.get("condition_id"),
    }
