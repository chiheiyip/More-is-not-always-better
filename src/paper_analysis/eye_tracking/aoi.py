from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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


def compute_aoi_metrics(df: pd.DataFrame, aois: list[PolygonAOI]) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_eye_columns(df)
    x = pd.to_numeric(df["Gaze Point X[px]"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["Gaze Point Y[px]"], errors="coerce").to_numpy()
    t0 = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce").min()
    finite = np.isfinite(x) & np.isfinite(y)
    class_masks: dict[str, list[np.ndarray]] = {}
    poly_rows = []
    for aoi in aois:
        mask = point_in_poly(x, y, aoi.points) & finite
        class_masks.setdefault(aoi.class_name, []).append(mask)
        poly_rows.append(_metric_row(df.loc[mask], aoi.class_name, {"polygon_id": aoi.polygon_id, "samples": int(mask.sum())}, t0=t0))

    class_rows = []
    total_fix = _trial_total_fixation_duration(df)
    trial_s = _trial_duration_s(df)
    for class_name, masks in class_masks.items():
        union = np.logical_or.reduce(masks) if masks else np.zeros(len(df), dtype=bool)
        row = _metric_row(df.loc[union], class_name, {"polygon_count": len(masks), "samples": int(union.sum())}, t0=t0)
        row["visited"] = bool(row["fixation_count"] > 0 or row["samples"] > 0)
        row["attention_share"] = row["TFD_ms"] / total_fix if total_fix and pd.notna(row["TFD_ms"]) else np.nan
        row["FCR"] = row["fixation_count"] / trial_s if trial_s and trial_s > 0 else np.nan
        class_rows.append(row)
    return pd.DataFrame(poly_rows), pd.DataFrame(class_rows)


def compute_whole_scene_metrics(df: pd.DataFrame) -> pd.DataFrame:
    _require_eye_columns(df)
    t0 = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce").min()
    row = _metric_row(df, "whole_scene", {"polygon_count": 0, "samples": int(len(df)), "aoi_available": False}, t0=t0)
    row["visited"] = True
    row["attention_share"] = 1.0
    row["FCR"] = row["fixation_count"] / _trial_duration_s(df) if _trial_duration_s(df) else np.nan
    return pd.DataFrame([row])


def compute_timebin_aoi_metrics(
    df: pd.DataFrame,
    aois: list[PolygonAOI],
    bin_size_ms: int = 2000,
    eye_offset_ms: float = 0.0,
) -> pd.DataFrame:
    _require_eye_columns(df)
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")
    out = _add_timebin_columns(df, bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
    rows: list[dict] = []
    for bin_index, sub in out.dropna(subset=["bin_index"]).groupby("bin_index", sort=True):
        _, metrics = compute_aoi_metrics(sub, aois)
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
) -> pd.DataFrame:
    _require_eye_columns(df)
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")
    out = _add_timebin_columns(df, bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
    rows: list[dict] = []
    for bin_index, sub in out.dropna(subset=["bin_index"]).groupby("bin_index", sort=True):
        metrics = compute_whole_scene_metrics(sub)
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
    }


def point_in_poly(x: np.ndarray, y: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    n = len(poly)
    inside = np.zeros_like(x, dtype=bool)
    if n < 3:
        return inside
    px = np.array([p[0] for p in poly], dtype=float)
    py = np.array([p[1] for p in poly], dtype=float)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersects = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= intersects
        j = i
    return inside


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _metric_row(sub: pd.DataFrame, class_name: str, extra: dict, t0: float) -> dict:
    row = {"class_name": class_name, **extra}
    row["TFD_ms"] = _dwell_time(sub)
    row["fixation_count"] = _fixation_count(sub)
    row["TTFF_ms"] = _ttff(sub, t0)
    return row


def _require_eye_columns(df: pd.DataFrame) -> None:
    required = {"Gaze Point X[px]", "Gaze Point Y[px]", "Recording Time Stamp[ms]"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eye table missing columns: {sorted(missing)}")


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


def _ttff(sub: pd.DataFrame, t0: float) -> float:
    if sub.empty:
        return np.nan
    t = pd.to_numeric(sub["Recording Time Stamp[ms]"], errors="coerce")
    return float(t.min() - t0) if t.notna().any() and pd.notna(t0) else np.nan


def _trial_total_fixation_duration(df: pd.DataFrame) -> float:
    return _dwell_time(df)


def _trial_duration_s(df: pd.DataFrame) -> float:
    t = pd.to_numeric(df["Recording Time Stamp[ms]"], errors="coerce").dropna()
    return float((t.max() - t.min()) / 1000.0) if len(t) > 1 else np.nan
