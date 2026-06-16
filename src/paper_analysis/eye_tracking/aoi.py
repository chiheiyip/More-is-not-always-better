from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.eye_tracking.qc import EyeQCPolicy, valid_eye_mask


@dataclass(frozen=True)
class PolygonAOI:
    class_name: str
    polygon_id: int
    points: list[tuple[float, float]]


def load_aoi_json(path: str | Path) -> list[PolygonAOI]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    aois: list[PolygonAOI] = []
    raw = data.get("aoi_classes") or data.get("classes")
    if isinstance(raw, dict):
        for class_name, polygons in raw.items():
            for idx, item in enumerate(polygons, start=1):
                points = _extract_points(item)
                if points:
                    aois.append(PolygonAOI(str(class_name), idx, points))
    elif isinstance(data.get("aois"), list):
        counts: dict[str, int] = {}
        for item in data["aois"]:
            class_name = str(item.get("class_name") or item.get("name") or "AOI")
            points = _extract_points(item)
            if points:
                counts[class_name] = counts.get(class_name, 0) + 1
                aois.append(PolygonAOI(class_name, counts[class_name], points))
    if not aois:
        raise ValueError(f"Unsupported or empty AOI JSON: {path}")
    return aois


def compute_aoi_metrics(
    df: pd.DataFrame,
    aois: list[PolygonAOI],
    point_source: str = "auto",
    screen_w: int | None = None,
    screen_h: int | None = None,
    validity_accepted: tuple[str, ...] | None = None,
    timestamp_gap_ms: float = 5000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_col, y_col, point_source_used = _resolve_point_columns(df, point_source)
    _require_eye_columns(df, x_col=x_col, y_col=y_col)
    policy = EyeQCPolicy(screen_w=screen_w, screen_h=screen_h, validity_accepted=validity_accepted)
    analysis_mask, qc_stats = valid_eye_mask(df, x_col, y_col, policy)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    finite = analysis_mask.to_numpy(dtype=bool)
    time_ctx = _time_context(df, timestamp_gap_ms=timestamp_gap_ms)
    trial_totals = _trial_totals(df, analysis_mask, time_ctx)
    class_masks: dict[str, list[np.ndarray]] = {}
    poly_rows = []
    for aoi in aois:
        mask = point_in_poly(x, y, aoi.points) & finite
        class_masks.setdefault(aoi.class_name, []).append(mask)
        poly_rows.append(_metric_row(df, mask, aoi.class_name, {"polygon_id": aoi.polygon_id}, time_ctx, trial_totals))

    class_rows = []
    for class_name, masks in class_masks.items():
        union = np.logical_or.reduce(masks) if masks else np.zeros(len(df), dtype=bool)
        class_rows.append(_metric_row(df, union, class_name, {"polygon_count": len(masks)}, time_ctx, trial_totals))
    poly_df = pd.DataFrame(poly_rows)
    class_df = pd.DataFrame(class_rows)
    qc = {
        **qc_stats,
        **_time_qc_stats(time_ctx),
        "point_source_requested": point_source,
        "point_source_used": point_source_used,
        "x_col_used": x_col,
        "y_col_used": y_col,
    }
    overlap = _class_overlap_rows(class_masks)
    poly_df.attrs["eye_qc"] = qc
    class_df.attrs["eye_qc"] = qc
    poly_df.attrs["aoi_overlap"] = overlap
    class_df.attrs["aoi_overlap"] = overlap
    return poly_df, class_df


def compute_whole_scene_metrics(
    df: pd.DataFrame,
    point_source: str = "auto",
    screen_w: int | None = None,
    screen_h: int | None = None,
    validity_accepted: tuple[str, ...] | None = None,
    timestamp_gap_ms: float = 5000.0,
) -> pd.DataFrame:
    x_col, y_col, point_source_used = _resolve_point_columns(df, point_source)
    _require_eye_columns(df, x_col=x_col, y_col=y_col)
    policy = EyeQCPolicy(screen_w=screen_w, screen_h=screen_h, validity_accepted=validity_accepted)
    analysis_mask, qc_stats = valid_eye_mask(df, x_col, y_col, policy)
    mask = analysis_mask.to_numpy(dtype=bool)
    time_ctx = _time_context(df, timestamp_gap_ms=timestamp_gap_ms)
    trial_totals = _trial_totals(df, analysis_mask, time_ctx)
    row = _metric_row(df, mask, "whole_scene", {"polygon_count": 0, "aoi_available": False}, time_ctx, trial_totals)
    row["visited"] = True if len(df) else False
    out = pd.DataFrame([row])
    out.attrs["eye_qc"] = {
        **qc_stats,
        **_time_qc_stats(time_ctx),
        "point_source_requested": point_source,
        "point_source_used": point_source_used,
        "x_col_used": x_col,
        "y_col_used": y_col,
    }
    out.attrs["aoi_overlap"] = []
    return out


def compute_timebin_aoi_metrics(
    df: pd.DataFrame,
    aois: list[PolygonAOI],
    bin_size_ms: int = 2000,
    eye_offset_ms: float = 0.0,
    point_source: str = "auto",
    screen_w: int | None = None,
    screen_h: int | None = None,
    validity_accepted: tuple[str, ...] | None = None,
    timestamp_gap_ms: float = 5000.0,
) -> pd.DataFrame:
    x_col, y_col, _ = _resolve_point_columns(df, point_source)
    _require_eye_columns(df, x_col=x_col, y_col=y_col)
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")
    out = _add_timebin_columns(df, bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
    rows: list[dict] = []
    for bin_index, sub in out.dropna(subset=["bin_index"]).groupby("bin_index", sort=True):
        _, metrics = compute_aoi_metrics(
            sub,
            aois,
            point_source=point_source,
            screen_w=screen_w,
            screen_h=screen_h,
            validity_accepted=validity_accepted,
            timestamp_gap_ms=timestamp_gap_ms,
        )
        bin_start_ms = int(bin_index) * bin_size_ms
        bin_end_ms = bin_start_ms + bin_size_ms
        if metrics.empty:
            rows.append(_empty_bin_row(int(bin_index), bin_start_ms, bin_end_ms))
            continue
        metrics = metrics.copy()
        metrics.insert(0, "bin_end_ms", bin_end_ms)
        metrics.insert(0, "bin_start_ms", bin_start_ms)
        metrics.insert(0, "bin_index", int(bin_index))
        rows.extend(metrics.to_dict("records"))
    return pd.DataFrame(rows)


def compute_whole_scene_timebin_metrics(
    df: pd.DataFrame,
    bin_size_ms: int = 2000,
    eye_offset_ms: float = 0.0,
    point_source: str = "auto",
    screen_w: int | None = None,
    screen_h: int | None = None,
    validity_accepted: tuple[str, ...] | None = None,
    timestamp_gap_ms: float = 5000.0,
) -> pd.DataFrame:
    x_col, y_col, _ = _resolve_point_columns(df, point_source)
    _require_eye_columns(df, x_col=x_col, y_col=y_col)
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")
    out = _add_timebin_columns(df, bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
    rows: list[dict] = []
    for bin_index, sub in out.dropna(subset=["bin_index"]).groupby("bin_index", sort=True):
        metrics = compute_whole_scene_metrics(
            sub,
            point_source=point_source,
            screen_w=screen_w,
            screen_h=screen_h,
            validity_accepted=validity_accepted,
            timestamp_gap_ms=timestamp_gap_ms,
        )
        bin_start_ms = int(bin_index) * bin_size_ms
        bin_end_ms = bin_start_ms + bin_size_ms
        metrics.insert(0, "bin_end_ms", bin_end_ms)
        metrics.insert(0, "bin_start_ms", bin_start_ms)
        metrics.insert(0, "bin_index", int(bin_index))
        rows.extend(metrics.to_dict("records"))
    return pd.DataFrame(rows)


def eye_file_stats(df: pd.DataFrame, bin_size_ms: int = 2000, eye_offset_ms: float = 0.0) -> dict:
    if "Recording Time Stamp[ms]" not in df.columns:
        return {
            "eye_sample_count": int(len(df)),
            "eye_duration_s": np.nan,
            "eye_first_timestamp_ms": np.nan,
            "eye_last_timestamp_ms": np.nan,
            "timebin_count": 0,
        }
    t = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce").dropna()
    if t.empty:
        return {
            "eye_sample_count": int(len(df)),
            "eye_duration_s": np.nan,
            "eye_first_timestamp_ms": np.nan,
            "eye_last_timestamp_ms": np.nan,
            "timebin_count": 0,
        }
    duration_ms = float(t.max() - t.min())
    aligned_max = duration_ms + float(eye_offset_ms or 0.0)
    return {
        "eye_sample_count": int(len(df)),
        "eye_duration_s": duration_ms / 1000.0,
        "eye_first_timestamp_ms": float(t.min()),
        "eye_last_timestamp_ms": float(t.max()),
        "timebin_count": int(np.ceil(max(aligned_max, 0.0) / bin_size_ms)) if bin_size_ms > 0 else 0,
    }


def aoi_validation(aois: list[PolygonAOI], class_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aoi in aois:
        xs = [p[0] for p in aoi.points]
        ys = [p[1] for p in aoi.points]
        rows.append({
            "class_name": aoi.class_name,
            "polygon_id": aoi.polygon_id,
            "polygon_area_px2": polygon_area(aoi.points),
            "bbox_width_px": max(xs) - min(xs),
            "bbox_height_px": max(ys) - min(ys),
        })
    out = pd.DataFrame(rows)
    if not class_metrics.empty:
        visited = class_metrics.groupby("class_name")["visited"].mean().rename("visited_rate").reset_index()
        out = out.merge(visited, on="class_name", how="left")
    return out


def _add_timebin_columns(df: pd.DataFrame, bin_size_ms: int, eye_offset_ms: float) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_numeric(out["Recording Time Stamp[ms]"], errors="coerce")
    if not t.notna().any():
        out["eye_aligned_ms"] = np.nan
        out["bin_index"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        return out
    out["eye_aligned_ms"] = t - t.min() + float(eye_offset_ms or 0.0)
    out["bin_index"] = np.floor(out["eye_aligned_ms"] / bin_size_ms).astype("Int64")
    return out


def _empty_bin_row(bin_index: int, bin_start_ms: int, bin_end_ms: int) -> dict:
    return {
        "bin_index": bin_index,
        "bin_start_ms": bin_start_ms,
        "bin_end_ms": bin_end_ms,
        "class_name": "",
        "polygon_count": 0,
        "samples": 0,
        "TFD_ms": np.nan,
        "fixation_count": 0,
        "TTFF_ms": np.nan,
        "visited": False,
        "attention_share": np.nan,
        "FCR": np.nan,
        "FC": 0,
        "FC_share": np.nan,
        "FC_rate": np.nan,
        "FFD_ms": np.nan,
        "MFD_ms": np.nan,
        "RFF": 0,
        "MPD": np.nan,
    }


def point_in_poly(
    x: np.ndarray,
    y: np.ndarray,
    poly: list[tuple[float, float]],
    boundary_eps: float = 1e-6,
) -> np.ndarray:
    n = len(poly)
    if n < 3:
        return np.zeros_like(x, dtype=bool)
    px = np.array([p[0] for p in poly], dtype=float)
    py = np.array([p[1] for p in poly], dtype=float)
    bbox = (
        (x >= np.nanmin(px) - boundary_eps)
        & (x <= np.nanmax(px) + boundary_eps)
        & (y >= np.nanmin(py) - boundary_eps)
        & (y <= np.nanmax(py) + boundary_eps)
    )
    inside = np.zeros_like(x, dtype=bool)
    if not np.any(bbox):
        return inside

    xx = x[bbox].astype(float)
    yy = y[bbox].astype(float)
    on_edge = np.zeros_like(xx, dtype=bool)
    j = n - 1
    for i in range(n):
        x1, y1 = px[j], py[j]
        x2, y2 = px[i], py[i]
        minx, maxx = min(x1, x2), max(x1, x2)
        miny, maxy = min(y1, y2), max(y1, y2)
        seg_bbox = (
            (xx >= minx - boundary_eps)
            & (xx <= maxx + boundary_eps)
            & (yy >= miny - boundary_eps)
            & (yy <= maxy + boundary_eps)
        )
        cross = (xx - x1) * (y2 - y1) - (yy - y1) * (x2 - x1)
        on_edge |= seg_bbox & (np.abs(cross) <= boundary_eps)
        j = i

    ray_inside = np.zeros_like(xx, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersects = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-12) + xi
        )
        ray_inside ^= intersects
        j = i
    inside[bbox] = ray_inside | on_edge
    return inside


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _metric_row(
    df: pd.DataFrame,
    mask: np.ndarray,
    class_name: str,
    extra: dict,
    time_ctx: dict,
    trial_totals: dict,
) -> dict:
    sub = df.loc[mask]
    row = {"class_name": class_name, **extra}
    row["samples"] = int(mask.sum())
    row["TFD_ms"] = _dwell_time(sub)
    row["TFD"] = row["TFD_ms"]
    row["fixation_count"] = _fixation_count(sub)
    row["FC"] = row["fixation_count"]
    row["TTFF_ms"] = _ttff(sub, time_ctx)
    row["TTFF"] = row["TTFF_ms"]
    row["FFD_ms"] = _first_fixation_duration(sub)
    row["FFD"] = row["FFD_ms"]
    row["MFD_ms"] = _mean_fixation_duration(sub)
    row["MFD"] = row["MFD_ms"]
    row["RFF"] = _refixation_frequency(df, mask)
    row["MPD"] = _mean_pupil_diameter(sub)
    row["visited"] = bool(row["FC"] > 0 or row["samples"] > 0)
    total_tfd = trial_totals["TFD_total_trial"]
    total_fc = trial_totals["FC_total_trial"]
    trial_s = trial_totals["trial_duration_s"]
    row["attention_share"] = row["TFD_ms"] / total_tfd if total_tfd and pd.notna(row["TFD_ms"]) else np.nan
    row["share"] = row["attention_share"]
    row["share_pct"] = 100.0 * row["share"] if pd.notna(row["share"]) else np.nan
    row["FC_share"] = row["FC"] / total_fc if total_fc and total_fc > 0 else np.nan
    row["FC_prop"] = row["FC_share"]
    row["FC_rate"] = row["FC"] / trial_s if trial_s and trial_s > 0 else np.nan
    row["FCR"] = row["FC_rate"]
    row["TFD_total_trial"] = total_tfd
    row["FC_total_trial"] = total_fc
    row["trial_duration_s"] = trial_s
    row["ttff_source"] = time_ctx.get("source")
    row["time_segment_count"] = time_ctx.get("segment_count")
    row["timestamp_reset_count"] = time_ctx.get("reset_count")
    row["timestamp_gap_count"] = time_ctx.get("gap_count")
    return row


def _require_eye_columns(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    required = {"Recording Time Stamp[ms]", x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eye table missing columns: {sorted(missing)}")


def _resolve_point_columns(df: pd.DataFrame, point_source: str = "auto") -> tuple[str, str, str]:
    point_source = str(point_source or "auto").strip().lower()
    if point_source not in {"auto", "gaze", "fixation"}:
        raise ValueError("point_source must be one of: auto, gaze, fixation")
    fixation = ("Fixation Point X[px]", "Fixation Point Y[px]")
    gaze = ("Gaze Point X[px]", "Gaze Point Y[px]")
    if point_source in {"auto", "fixation"} and set(fixation).issubset(df.columns):
        return fixation[0], fixation[1], "fixation"
    if point_source in {"auto", "gaze"} and set(gaze).issubset(df.columns):
        return gaze[0], gaze[1], "gaze"
    if point_source == "fixation":
        raise ValueError("Fixation point columns are unavailable")
    raise ValueError("Gaze point columns are unavailable")


def _extract_points(item: object) -> list[tuple[float, float]]:
    raw = item.get("points") or item.get("polygon") or item.get("vertices") if isinstance(item, dict) else item
    points = []
    if not isinstance(raw, list):
        return points
    for point in raw:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        points.append((float(x), float(y)))
    return points


def _dwell_time(sub: pd.DataFrame) -> float:
    if sub.empty or "Fixation Duration[ms]" not in sub.columns:
        return np.nan
    durations = pd.to_numeric(sub["Fixation Duration[ms]"], errors="coerce")
    if "Fixation Index" in sub.columns:
        tmp = pd.DataFrame({"idx": pd.to_numeric(sub["Fixation Index"], errors="coerce"), "dur": durations}).dropna()
        return float(tmp.groupby("idx")["dur"].max().sum()) if not tmp.empty else np.nan
    return float(durations.dropna().sum())


def _fixation_count(sub: pd.DataFrame) -> int:
    if sub.empty:
        return 0
    if "Fixation Index" in sub.columns:
        return int(pd.to_numeric(sub["Fixation Index"], errors="coerce").dropna().nunique())
    return int(len(sub))


def _ttff(sub: pd.DataFrame, time_ctx: dict) -> float:
    if sub.empty:
        return np.nan
    t = pd.to_numeric(sub["Recording Time Stamp[ms]"], errors="coerce")
    if not t.notna().any():
        return np.nan
    idx = t.idxmin()
    source = time_ctx.get("time")
    segments = time_ctx.get("segment_id")
    baselines = time_ctx.get("baseline_by_segment", {})
    if source is None or segments is None or idx not in source.index:
        return np.nan
    sid = int(segments.loc[idx])
    value = source.loc[idx]
    baseline = baselines.get(sid, np.nan)
    return float(value - baseline) if pd.notna(value) and pd.notna(baseline) else np.nan


def _trial_total_fixation_duration(df: pd.DataFrame) -> float:
    return _dwell_time(df)


def _trial_duration_s(df: pd.DataFrame) -> float:
    t = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce").dropna()
    return float((t.max() - t.min()) / 1000.0) if len(t) > 1 else np.nan


def _trial_totals(df: pd.DataFrame, analysis_mask: pd.Series, time_ctx: dict) -> dict:
    sub = df.loc[analysis_mask]
    source = time_ctx.get("time")
    duration_s = _trial_duration_s(sub)
    if source is not None and analysis_mask.any():
        vals = source.loc[analysis_mask].dropna()
        if len(vals) > 1:
            duration_s = float((vals.max() - vals.min()) / 1000.0)
    return {
        "TFD_total_trial": _dwell_time(sub),
        "FC_total_trial": _fixation_count(sub),
        "trial_duration_s": duration_s,
    }


def _time_context(df: pd.DataFrame, timestamp_gap_ms: float = 5000.0) -> dict:
    source_name = "recording_timestamp"
    if "Video Time[ms]" in df.columns:
        time = pd.to_numeric(df["Video Time[ms]"], errors="coerce")
        source_name = "video_time_ms"
    elif "Video Time[HH:mm:ss.ms]" in df.columns:
        time = _parse_hhmmss_ms_series(df["Video Time[HH:mm:ss.ms]"])
        source_name = "video_time_hhmmss"
    else:
        time = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce")
    diff = time.diff()
    resets = (diff < 0).fillna(False)
    gaps = (diff > float(timestamp_gap_ms)).fillna(False)
    if len(resets):
        resets.iloc[0] = False
        gaps.iloc[0] = False
    segment_id = (resets | gaps).cumsum().astype(int)
    baseline_by_segment: dict[int, float] = {}
    for sid, sub in pd.DataFrame({"segment_id": segment_id, "time": time}).groupby("segment_id"):
        baseline_by_segment[int(sid)] = float(sub["time"].dropna().min()) if sub["time"].notna().any() else np.nan
    return {
        "source": source_name,
        "time": time,
        "segment_id": segment_id,
        "segment_count": int(segment_id.nunique()) if len(segment_id) else 0,
        "reset_count": int(resets.sum()),
        "gap_count": int(gaps.sum()),
        "gap_threshold_ms": float(timestamp_gap_ms),
        "baseline_by_segment": baseline_by_segment,
    }


def _time_qc_stats(time_ctx: dict) -> dict:
    return {
        "time_source": time_ctx.get("source"),
        "time_segment_count": time_ctx.get("segment_count"),
        "timestamp_reset_count": time_ctx.get("reset_count"),
        "timestamp_gap_count": time_ctx.get("gap_count"),
        "timestamp_gap_threshold_ms": time_ctx.get("gap_threshold_ms"),
    }


def _parse_hhmmss_ms_series(series: pd.Series) -> pd.Series:
    delta = pd.to_timedelta(series.astype(str).str.strip(), errors="coerce")
    return pd.Series(delta.dt.total_seconds() * 1000.0, index=series.index, dtype=float)


def _fixation_table(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame(columns=["fixation_id", "first_ts", "duration_ms"])
    ts = pd.to_numeric(sub.get("Recording Time Stamp[ms]"), errors="coerce")
    duration = pd.to_numeric(sub.get("Fixation Duration[ms]"), errors="coerce") if "Fixation Duration[ms]" in sub.columns else pd.Series(np.nan, index=sub.index)
    if "Fixation Index" in sub.columns:
        idx = pd.to_numeric(sub["Fixation Index"], errors="coerce")
    else:
        idx = pd.Series(np.arange(len(sub)), index=sub.index, dtype=float)
    tmp = pd.DataFrame({"fixation_id": idx, "first_ts": ts, "duration_ms": duration}).dropna(subset=["fixation_id"])
    if tmp.empty:
        return pd.DataFrame(columns=["fixation_id", "first_ts", "duration_ms"])
    return tmp.groupby("fixation_id", as_index=False).agg(first_ts=("first_ts", "min"), duration_ms=("duration_ms", "max")).sort_values(["first_ts", "fixation_id"]).reset_index(drop=True)


def _first_fixation_duration(sub: pd.DataFrame) -> float:
    table = _fixation_table(sub)
    if table.empty or not table["duration_ms"].notna().any():
        return np.nan
    first = table.loc[table["first_ts"].idxmin()] if table["first_ts"].notna().any() else table.iloc[0]
    return float(first["duration_ms"]) if pd.notna(first["duration_ms"]) else np.nan


def _mean_fixation_duration(sub: pd.DataFrame) -> float:
    table = _fixation_table(sub)
    if table.empty or not table["duration_ms"].notna().any():
        return np.nan
    return float(table["duration_ms"].dropna().mean())


def _refixation_frequency(df: pd.DataFrame, mask: np.ndarray) -> int:
    if len(df) == 0 or "Fixation Index" not in df.columns:
        return 0
    tmp = pd.DataFrame({
        "fixation_id": pd.to_numeric(df["Fixation Index"], errors="coerce"),
        "first_ts": pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce"),
        "in_aoi": np.asarray(mask, dtype=bool),
    }).dropna(subset=["fixation_id"])
    if tmp.empty:
        return 0
    seq = tmp.groupby("fixation_id", as_index=False).agg(first_ts=("first_ts", "min"), in_aoi=("in_aoi", "max")).sort_values(["first_ts", "fixation_id"])
    entries = 0
    prev = False
    for cur in seq["in_aoi"].astype(bool).tolist():
        if cur and not prev:
            entries += 1
        prev = cur
    return int(max(entries - 1, 0))


def _mean_pupil_diameter(sub: pd.DataFrame) -> float:
    if sub.empty:
        return np.nan
    candidates = [
        ("Pupil Diameter Left[mm]", "Pupil Diameter Right[mm]"),
        ("Pupil Diameter Left[px]", "Pupil Diameter Right[px]"),
    ]
    for left_col, right_col in candidates:
        cols = [c for c in [left_col, right_col] if c in sub.columns]
        if not cols:
            continue
        values = sub[cols].apply(pd.to_numeric, errors="coerce")
        row_mean = values.mean(axis=1, skipna=True)
        return float(row_mean.dropna().mean()) if row_mean.notna().any() else np.nan
    return np.nan


def _class_overlap_rows(class_masks: dict[str, list[np.ndarray]]) -> list[dict]:
    rows: list[dict] = []
    classes = list(class_masks)
    unions = {
        name: np.logical_or.reduce(masks) if masks else np.array([], dtype=bool)
        for name, masks in class_masks.items()
    }
    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1:]:
            mask_a = unions[class_a]
            mask_b = unions[class_b]
            if len(mask_a) == 0 or len(mask_b) == 0:
                continue
            overlap = np.logical_and(mask_a, mask_b)
            count = int(overlap.sum())
            if count == 0:
                continue
            samples_a = int(mask_a.sum())
            samples_b = int(mask_b.sum())
            rows.append({
                "class_a": class_a,
                "class_b": class_b,
                "overlap_samples": count,
                "samples_a": samples_a,
                "samples_b": samples_b,
                "overlap_ratio_a": count / samples_a if samples_a else np.nan,
                "overlap_ratio_b": count / samples_b if samples_b else np.nan,
            })
    return rows
