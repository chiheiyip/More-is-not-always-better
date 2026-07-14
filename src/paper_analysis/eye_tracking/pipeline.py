from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.eye_tracking.aoi import aoi_validation, compute_aoi_metrics, compute_whole_scene_metrics, load_aoi_document, polygon_area
from paper_analysis.eye_tracking.dynamics import compute_dynamic_eye_metrics
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
    eye_qc_config: str | Path | dict | None = None,
) -> dict[str, Path]:
    policy = _load_eye_policy(eye_qc_config)
    point_source = str(policy.get("point_source", point_source))
    screen_w = screen_w if screen_w is not None else policy.get("screen_w")
    screen_h = screen_h if screen_h is not None else policy.get("screen_h")
    if validity_accepted is None and policy.get("validity_accepted") is not None:
        validity_accepted = tuple(str(value) for value in policy["validity_accepted"])
    timestamp_gap_ms = float(policy.get("timestamp_gap_ms", timestamp_gap_ms))
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
    fixation_rows: list[pd.DataFrame] = []
    transition_rows: list[pd.DataFrame] = []
    transition_matrix_rows: list[pd.DataFrame] = []
    dynamic_rows: list[dict] = []
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
            aois, aoi_metadata = load_aoi_document(aoi_json)
            df, coordinate_contract = _apply_explicit_coordinate_contract(df, row, aoi_metadata)
            aoi_metadata.update(_aoi_structure_metadata(aois, aoi_metadata))
            aoi_metadata.update(coordinate_contract)
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
            aois, aoi_metadata = [], {}
            metrics = compute_whole_scene_metrics(
                df,
                point_source=point_source,
                screen_w=screen_w,
                screen_h=screen_h,
                validity_accepted=validity_accepted,
                timestamp_gap_ms=timestamp_gap_ms,
            )
            validation = pd.DataFrame([{"class_name": "whole_scene", "polygon_id": 0, "polygon_area_px2": None, "visited_rate": 1.0}])
        metric_qc = dict(metrics.attrs.get("eye_qc", {}))
        metric_overlap = list(metrics.attrs.get("aoi_overlap", []))
        dynamic = compute_dynamic_eye_metrics(
            df,
            aois,
            aoi_metadata=aoi_metadata,
            timestamp_gap_ms=timestamp_gap_ms,
        )
        if not dynamic.revisit_by_aoi.empty and "class_name" in metrics.columns:
            metrics = metrics.merge(dynamic.revisit_by_aoi, on="class_name", how="left")
        for key, value in aoi_metadata.items():
            validation[key] = value
        for frame in (metrics, validation):
            for key, value in base.items():
                frame[key] = value
        overlap = pd.DataFrame(metric_overlap)
        if not overlap.empty:
            for key, value in base.items():
                overlap[key] = value
            overlap_rows.append(overlap)
        for frame, target in [
            (dynamic.fixations, fixation_rows),
            (dynamic.transitions, transition_rows),
            (dynamic.transition_matrix, transition_matrix_rows),
        ]:
            if not frame.empty:
                frame = frame.copy()
                for key, value in base.items():
                    frame[key] = value
                target.append(frame)
        dynamic_rows.append({**base, **dynamic.trial_metrics})
        class_rows.append(metrics)
        validation_rows.append(validation)
        qc_rows.append({
            **base,
            **metric_qc,
            **dynamic.qc,
            **{key: aoi_metadata.get(key) for key in [
                "aoi_total_area_px2", "aoi_canvas_fraction", "aoi_class_count",
                "coordinate_contract_status", "coordinate_scale_x", "coordinate_scale_y",
            ]},
            "missing_eye_file": False,
            "missing_aoi_file": not aoi_json or not aoi_json.exists(),
            "eye_sample_count": len(df),
        })

    outdir = Path(outdir)
    eye_long = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    overlap = pd.concat(overlap_rows, ignore_index=True) if overlap_rows else pd.DataFrame()
    fixations = pd.concat(fixation_rows, ignore_index=True) if fixation_rows else pd.DataFrame()
    transitions = pd.concat(transition_rows, ignore_index=True) if transition_rows else pd.DataFrame()
    transition_matrix = pd.concat(transition_matrix_rows, ignore_index=True) if transition_matrix_rows else pd.DataFrame()
    dynamic_metrics = pd.DataFrame(dynamic_rows)
    qc = pd.DataFrame(qc_rows)
    qc_sensitivity, metric_availability = _eye_qc_sensitivity(qc, eye_long, dynamic_metrics)
    return {
        "eye_aoi_trial_long": write_table(eye_long, outdir / "eye_aoi_trial_long.csv"),
        "aoi_validation_summary": write_table(validation, outdir / "aoi_validation_summary.csv"),
        "aoi_overlap_summary": write_table(overlap, outdir / "aoi_overlap_summary.csv"),
        "eye_qc": write_table(qc, outdir / "eye_qc.csv"),
        "eye_fixation_sequence_long": write_table(fixations, outdir / "eye_fixation_sequence_long.csv"),
        "eye_transition_long": write_table(transitions, outdir / "eye_transition_long.csv"),
        "eye_transition_matrix": write_table(transition_matrix, outdir / "eye_transition_matrix.csv"),
        "eye_trial_dynamic_metrics": write_table(dynamic_metrics, outdir / "eye_trial_dynamic_metrics.csv"),
        "eye_qc_sensitivity": write_table(qc_sensitivity, outdir / "eye_qc_sensitivity.csv"),
        "eye_metric_availability": write_table(metric_availability, outdir / "eye_metric_availability.csv"),
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


def _eye_qc_sensitivity(
    qc: pd.DataFrame,
    eye_long: pd.DataFrame,
    dynamic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    ratio = pd.to_numeric(qc.get("analysis_valid_ratio"), errors="coerce") if not qc.empty else pd.Series(dtype=float)
    dimensions = [c for c in ["WWR", "Complexity", "block", "position"] if c in qc.columns]
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        keep = ratio.ge(threshold)
        selected = qc.loc[keep]
        rows.append({
            "threshold": threshold,
            "factor": "overall",
            "level": "all",
            "n_trials": int(len(selected)),
            "n_subjects": int(selected["participant_id"].astype(str).nunique()) if "participant_id" in selected else 0,
            "retention_rate": float(keep.mean()) if len(keep) else np.nan,
        })
        for factor in dimensions:
            for level, sub in qc.groupby(factor, dropna=False):
                sub_keep = ratio.loc[sub.index].ge(threshold)
                rows.append({
                    "threshold": threshold,
                    "factor": factor,
                    "level": level,
                    "n_trials": int(sub_keep.sum()),
                    "n_subjects": int(sub.loc[sub_keep, "participant_id"].astype(str).nunique()),
                    "retention_rate": float(sub_keep.mean()) if len(sub_keep) else np.nan,
                })
    metric_rows: list[dict] = []
    for grain, frame, metrics in [
        ("aoi", eye_long, ["visited", "FC", "FCR", "TFD_ms", "TTFF_ms", "attention_share", "median_revisit_latency_ms"]),
        ("scene_trial", dynamic, [
            "aoi_transition_count", "aoi_transition_rate_per_min", "transition_entropy_normalized",
            "angular_scanpath_deg_per_s", "saccade_rate_per_min", "blink_rate_per_min",
            "pupil_post_early_delta_mm", "first_window_fixation_latency_ms",
        ]),
    ]:
        if frame.empty:
            continue
        for metric in metrics:
            if metric not in frame:
                continue
            valid = frame[metric].notna()
            metric_rows.append({
                "grain": grain,
                "metric": metric,
                "n_rows": int(valid.sum()),
                "n_trials": int(frame.loc[valid, ["participant_id", "scene_id"]].drop_duplicates().shape[0]) if {"participant_id", "scene_id"}.issubset(frame.columns) else 0,
                "n_subjects": int(frame.loc[valid, "participant_id"].astype(str).nunique()) if "participant_id" in frame else 0,
                "missing_rate": float(1.0 - valid.mean()) if len(valid) else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def _load_eye_policy(config: str | Path | dict | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _apply_explicit_coordinate_contract(df: pd.DataFrame, row: pd.Series, metadata: dict) -> tuple[pd.DataFrame, dict]:
    """Scale coordinates only when both source and AOI canvases are explicit."""
    source_w = _first_number(row, "eye_coordinate_width_px", "source_canvas_width_px")
    source_h = _first_number(row, "eye_coordinate_height_px", "source_canvas_height_px")
    target_w = _finite_number(metadata.get("aoi_image_width_px"))
    target_h = _finite_number(metadata.get("aoi_image_height_px"))
    contract = {"coordinate_scale_x": np.nan, "coordinate_scale_y": np.nan}
    if source_w is None or source_h is None:
        contract["coordinate_contract_status"] = "unverified_source_canvas_not_in_manifest"
        return df, contract
    if target_w is None or target_h is None:
        contract["coordinate_contract_status"] = "incomplete_aoi_canvas_metadata"
        return df, contract
    scale_x, scale_y = target_w / source_w, target_h / source_h
    out = df.copy()
    for x_col, y_col in [("Gaze Point X[px]", "Gaze Point Y[px]"), ("Fixation Point X[px]", "Fixation Point Y[px]")]:
        if x_col in out:
            out[x_col] = pd.to_numeric(out[x_col], errors="coerce") * scale_x
        if y_col in out:
            out[y_col] = pd.to_numeric(out[y_col], errors="coerce") * scale_y
    contract.update({
        "coordinate_scale_x": scale_x,
        "coordinate_scale_y": scale_y,
        "coordinate_contract_status": "identity_verified" if np.isclose(scale_x, 1) and np.isclose(scale_y, 1) else "scaled_from_explicit_manifest_canvas",
    })
    return out, contract


def _aoi_structure_metadata(aois: list, metadata: dict) -> dict:
    total = float(sum(polygon_area(aoi.points) for aoi in aois))
    width = _finite_number(metadata.get("aoi_image_width_px"))
    height = _finite_number(metadata.get("aoi_image_height_px"))
    canvas = width * height if width and height else np.nan
    return {
        "aoi_total_area_px2": total,
        "aoi_canvas_fraction": total / canvas if np.isfinite(canvas) and canvas > 0 else np.nan,
        "aoi_class_count": len({aoi.class_name for aoi in aois}),
    }


def _first_number(row: pd.Series, *columns: str) -> float | None:
    for column in columns:
        if column in row.index:
            value = _finite_number(row.get(column))
            if value is not None and value > 0:
                return value
    return None


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
