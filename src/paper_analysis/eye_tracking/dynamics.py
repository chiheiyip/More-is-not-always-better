from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable

import numpy as np
import pandas as pd

from paper_analysis.eye_tracking.aoi import PolygonAOI, point_in_poly, polygon_area


TIME_COL = "Recording Time Stamp[ms]"
FIX_X = "Fixation Point X[px]"
FIX_Y = "Fixation Point Y[px]"


@dataclass(frozen=True)
class DynamicEyeResult:
    fixations: pd.DataFrame
    transitions: pd.DataFrame
    transition_matrix: pd.DataFrame
    trial_metrics: dict
    revisit_by_aoi: pd.DataFrame
    qc: dict


def compute_dynamic_eye_metrics(
    df: pd.DataFrame,
    aois: list[PolygonAOI],
    aoi_metadata: dict | None = None,
    timestamp_gap_ms: float = 5000.0,
    pupil_early_window_ms: float = 2000.0,
    blink_padding_ms: float = 100.0,
    pupil_max_interp_gap_ms: float = 200.0,
) -> DynamicEyeResult:
    metadata = dict(aoi_metadata or {})
    fixations = build_fixation_sequence(df, aois, metadata, timestamp_gap_ms)
    transitions = build_transition_events(fixations)
    matrix = build_transition_matrix(transitions)
    valid_duration_s = _valid_duration_s(df, timestamp_gap_ms)
    available_classes = sorted({aoi.class_name for aoi in aois})
    revisit = _revisit_summary(fixations)
    trial_metrics = {
        **_transition_metrics(fixations, transitions, available_classes, valid_duration_s),
        **_scanpath_metrics(fixations, metadata, valid_duration_s),
        **_saccade_metrics(df, aois, valid_duration_s),
        **_blink_metrics(df, valid_duration_s),
        **_pupil_metrics(
            df,
            early_window_ms=pupil_early_window_ms,
            blink_padding_ms=blink_padding_ms,
            max_interp_gap_ms=pupil_max_interp_gap_ms,
        ),
        "valid_eye_duration_s": valid_duration_s,
    }
    qc = _dynamic_qc(df, fixations, metadata, timestamp_gap_ms, trial_metrics)
    return DynamicEyeResult(fixations, transitions, matrix, trial_metrics, revisit, qc)


def build_fixation_sequence(
    df: pd.DataFrame,
    aois: list[PolygonAOI],
    metadata: dict | None = None,
    timestamp_gap_ms: float = 5000.0,
) -> pd.DataFrame:
    required = {"Fixation Index", TIME_COL, FIX_X, FIX_Y}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=_fixation_columns())
    work = df.copy()
    work["_fixation_id"] = pd.to_numeric(work["Fixation Index"], errors="coerce")
    work["_row_order"] = np.arange(len(work))
    work["_time_ms"] = pd.to_numeric(work[TIME_COL], errors="coerce")
    work["_x"] = pd.to_numeric(work[FIX_X], errors="coerce")
    work["_y"] = pd.to_numeric(work[FIX_Y], errors="coerce")
    work = work.dropna(subset=["_fixation_id", "_time_ms"])
    if work.empty:
        return pd.DataFrame(columns=_fixation_columns())
    work = work.sort_values("_row_order", kind="stable")
    diff = work["_time_ms"].diff()
    work["_segment_id"] = ((diff < 0) | (diff > float(timestamp_gap_ms))).fillna(False).cumsum().astype(int)
    width = _finite_number((metadata or {}).get("aoi_image_width_px"))
    height = _finite_number((metadata or {}).get("aoi_image_height_px"))
    class_areas = _class_areas(aois)
    rows: list[dict] = []
    for (segment_id, fixation_id), sub in work.groupby(["_segment_id", "_fixation_id"], sort=False):
        x = float(sub["_x"].mean()) if sub["_x"].notna().any() else np.nan
        y = float(sub["_y"].mean()) if sub["_y"].notna().any() else np.nan
        candidates = _point_aoi_candidates(x, y, aois)
        chosen = min(candidates, key=lambda name: (class_areas.get(name, np.inf), name)) if candidates else "outside"
        direction = _mean_gaze_direction(sub)
        start_ms = float(sub["_time_ms"].min())
        duration = pd.to_numeric(sub.get("Fixation Duration[ms]"), errors="coerce") if "Fixation Duration[ms]" in sub else pd.Series(dtype=float)
        duration_ms = float(duration.max()) if duration.notna().any() else max(float(sub["_time_ms"].max()) - start_ms, 0.0)
        end_ms = max(float(sub["_time_ms"].max()), start_ms + duration_ms)
        in_canvas = bool(np.isfinite(x) and np.isfinite(y) and 0 <= x <= width and 0 <= y <= height) if width is not None and height is not None else np.nan
        rows.append({
            "fixation_index": int(fixation_id),
            "segment_id": int(segment_id),
            "fixation_start_ms": start_ms,
            "fixation_end_ms": end_ms,
            "fixation_duration_ms": duration_ms,
            "fixation_x_px": x,
            "fixation_y_px": y,
            "gaze_direction_x": direction[0],
            "gaze_direction_y": direction[1],
            "gaze_direction_z": direction[2],
            "aoi_state": chosen,
            "aoi_candidates": ";".join(sorted(candidates)),
            "aoi_ambiguous": len(candidates) > 1,
            "coordinate_in_aoi_canvas": in_canvas,
        })
    return pd.DataFrame(rows, columns=_fixation_columns()).sort_values(["segment_id", "fixation_start_ms", "fixation_index"]).reset_index(drop=True)


def build_transition_events(fixations: pd.DataFrame) -> pd.DataFrame:
    if fixations.empty:
        return pd.DataFrame(columns=_transition_columns())
    rows: list[dict] = []
    full_index = 0
    named_index = 0
    for segment_id, sub in fixations.groupby("segment_id", sort=True):
        seq = _collapse_states(sub)
        for prev, cur in zip(seq[:-1], seq[1:]):
            full_index += 1
            rows.append(_transition_row(prev, cur, segment_id, "state", full_index, "outside" in {prev["aoi_state"], cur["aoi_state"]}))
        # Preserve positions in the full compressed state sequence so an
        # A->outside->B visit is distinguishable from a direct A->B visit.
        for pos, item in enumerate(seq):
            item["_full_seq_pos"] = pos
        named = [row for row in seq if row["aoi_state"] != "outside"]
        named = _collapse_state_records(named)
        for prev, cur in zip(named[:-1], named[1:]):
            named_index += 1
            between = seq[int(prev["_full_seq_pos"]) + 1:int(cur["_full_seq_pos"])]
            rows.append(_transition_row(prev, cur, segment_id, "named_aoi", named_index, any(item["aoi_state"] == "outside" for item in between)))
    return pd.DataFrame(rows, columns=_transition_columns())


def build_transition_matrix(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame(columns=["transition_scope", "from_aoi", "to_aoi", "transition_count", "transition_probability"])
    grouped = transitions.groupby(["transition_scope", "from_aoi", "to_aoi"], dropna=False).size().rename("transition_count").reset_index()
    totals = grouped.groupby(["transition_scope", "from_aoi"])["transition_count"].transform("sum")
    grouped["transition_probability"] = grouped["transition_count"] / totals.where(totals > 0, np.nan)
    return grouped


def _transition_metrics(
    fixations: pd.DataFrame,
    transitions: pd.DataFrame,
    available_classes: list[str],
    duration_s: float,
) -> dict:
    state = transitions.loc[transitions.get("transition_scope", pd.Series(dtype=str)).eq("state")] if not transitions.empty else transitions
    named = transitions.loc[transitions.get("transition_scope", pd.Series(dtype=str)).eq("named_aoi")] if not transitions.empty else transitions
    count = int(len(named))
    counts = named.groupby(["from_aoi", "to_aoi"]).size() if not named.empty else pd.Series(dtype=int)
    if counts.empty:
        entropy = np.nan
        entropy_norm = np.nan
    else:
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * probabilities.map(log2)).sum())
        possible = len(available_classes) * max(len(available_classes) - 1, 0)
        entropy_norm = float(entropy / log2(possible)) if possible > 1 else np.nan
    window_entries = int((named.get("to_aoi", pd.Series(dtype=str)).eq("window") & ~named.get("from_aoi", pd.Series(dtype=str)).eq("window")).sum()) if not named.empty else 0
    trial_start = float(fixations["fixation_start_ms"].min()) if not fixations.empty else np.nan
    window_fix = fixations.loc[fixations["aoi_state"].eq("window")] if not fixations.empty else fixations
    first_window = float(window_fix["fixation_start_ms"].min() - trial_start) if not window_fix.empty and np.isfinite(trial_start) else np.nan
    return {
        "state_transition_count": int(len(state)),
        "aoi_transition_count": count,
        "aoi_transition_rate_per_min": count / (duration_s / 60.0) if duration_s and duration_s > 0 else np.nan,
        "transition_entropy_bits": entropy,
        "transition_entropy_normalized": entropy_norm,
        "window_entry_count": window_entries,
        "window_directed_transition_rate_per_min": window_entries / (duration_s / 60.0) if duration_s and duration_s > 0 else np.nan,
        "first_window_fixation_latency_ms": first_window,
    }


def _scanpath_metrics(fixations: pd.DataFrame, metadata: dict, duration_s: float) -> dict:
    pixel_length = 0.0
    angular_length = 0.0
    pixel_pairs = 0
    angular_pairs = 0
    for _, sub in fixations.groupby("segment_id", sort=True) if not fixations.empty else []:
        sub = sub.sort_values("fixation_start_ms")
        xy = sub[["fixation_x_px", "fixation_y_px"]].to_numpy(dtype=float)
        valid_pairs = np.isfinite(xy[:-1]).all(axis=1) & np.isfinite(xy[1:]).all(axis=1) if len(xy) > 1 else np.array([], dtype=bool)
        if valid_pairs.any():
            pixel_length += float(np.linalg.norm(xy[1:] - xy[:-1], axis=1)[valid_pairs].sum())
            pixel_pairs += int(valid_pairs.sum())
        directions = sub[["gaze_direction_x", "gaze_direction_y", "gaze_direction_z"]].to_numpy(dtype=float)
        if len(directions) > 1:
            valid = np.isfinite(directions[:-1]).all(axis=1) & np.isfinite(directions[1:]).all(axis=1)
            if valid.any():
                dots = np.sum(directions[:-1][valid] * directions[1:][valid], axis=1)
                angular_length += float(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))).sum())
                angular_pairs += int(valid.sum())
    width = _finite_number(metadata.get("aoi_image_width_px"))
    height = _finite_number(metadata.get("aoi_image_height_px"))
    diagonal = float(np.hypot(width, height)) if width and height else np.nan
    return {
        "scanpath_length_px": pixel_length if pixel_pairs else np.nan,
        "scanpath_length_screen_diagonal": pixel_length / diagonal if pixel_pairs and diagonal > 0 else np.nan,
        "scanpath_length_px_per_s": pixel_length / duration_s if pixel_pairs and duration_s > 0 else np.nan,
        "angular_scanpath_deg": angular_length if angular_pairs else np.nan,
        "angular_scanpath_deg_per_s": angular_length / duration_s if angular_pairs and duration_s > 0 else np.nan,
    }


def _saccade_metrics(df: pd.DataFrame, aois: list[PolygonAOI], duration_s: float) -> dict:
    if "Saccade Index" not in df.columns:
        return _empty_saccade_metrics()
    ids = pd.to_numeric(df["Saccade Index"], errors="coerce")
    work = df.loc[ids.notna()].copy()
    work["_saccade_id"] = ids.loc[ids.notna()]
    if work.empty:
        return _empty_saccade_metrics(zero_count=True, duration_s=duration_s)
    events = []
    for _, sub in work.groupby("_saccade_id", sort=False):
        events.append({
            "amplitude": _median_numeric(sub, "Saccade Amplitude[px]"),
            "velocity_avg": _median_numeric(sub, "Saccade Velocity Average[px/ms]"),
            "velocity_peak": _median_numeric(sub, "Saccade Velocity Peak[px/ms]"),
            "cross_aoi": _event_crosses_aoi(sub, aois),
        })
    event_df = pd.DataFrame(events)
    count = len(event_df)
    return {
        "saccade_count": count,
        "saccade_rate_per_min": count / (duration_s / 60.0) if duration_s > 0 else np.nan,
        "saccade_amplitude_median_px": float(event_df["amplitude"].median()),
        "saccade_velocity_average_median_px_per_ms": float(event_df["velocity_avg"].median()),
        "saccade_velocity_peak_median_px_per_ms": float(event_df["velocity_peak"].median()),
        "cross_aoi_saccade_ratio": float(event_df["cross_aoi"].mean()) if event_df["cross_aoi"].notna().any() else np.nan,
    }


def _blink_metrics(df: pd.DataFrame, duration_s: float) -> dict:
    if "Blink Index" not in df.columns:
        return {"blink_count": np.nan, "blink_rate_per_min": np.nan, "blink_duration_median_ms": np.nan}
    ids = pd.to_numeric(df["Blink Index"], errors="coerce")
    work = df.loc[ids.notna()].copy()
    work["_blink_id"] = ids.loc[ids.notna()]
    if work.empty:
        return {"blink_count": 0, "blink_rate_per_min": 0.0 if duration_s > 0 else np.nan, "blink_duration_median_ms": np.nan}
    durations = work.groupby("_blink_id")["Blink Duration[ms]"].max() if "Blink Duration[ms]" in work else pd.Series(dtype=float)
    count = int(work["_blink_id"].nunique())
    return {
        "blink_count": count,
        "blink_rate_per_min": count / (duration_s / 60.0) if duration_s > 0 else np.nan,
        "blink_duration_median_ms": float(pd.to_numeric(durations, errors="coerce").median()) if not durations.empty else np.nan,
    }


def _pupil_metrics(df: pd.DataFrame, early_window_ms: float, blink_padding_ms: float, max_interp_gap_ms: float) -> dict:
    t = pd.to_numeric(df.get(TIME_COL), errors="coerce") if TIME_COL in df else pd.Series(np.nan, index=df.index)
    eye_cols = [c for c in ["Pupil Diameter Left[mm]", "Pupil Diameter Right[mm]"] if c in df]
    if not eye_cols or not t.notna().any():
        return _empty_pupil_metrics()
    blink_mask = pd.Series(False, index=df.index)
    if "Blink Index" in df:
        blink = pd.to_numeric(df["Blink Index"], errors="coerce").notna()
        for event_t in t.loc[blink].dropna().to_numpy():
            blink_mask |= t.between(event_t - blink_padding_ms, event_t + blink_padding_ms)
    processed = []
    raw_valid = []
    for col in eye_cols:
        values = pd.to_numeric(df[col], errors="coerce").mask(blink_mask)
        raw_valid.append(values.notna())
        finite = values.dropna()
        if not finite.empty:
            median = float(finite.median())
            mad = float((finite - median).abs().median())
            if mad > 0:
                values = values.mask((values - median).abs() > 3.5 * mad)
        processed.append(_interpolate_short_gaps(t, values, max_interp_gap_ms))
    pupil = pd.concat(processed, axis=1).mean(axis=1, skipna=True)
    coverage = float(pupil.notna().mean()) if len(pupil) else np.nan
    binocular = float(pd.concat(raw_valid, axis=1).all(axis=1).mean()) if len(raw_valid) == 2 else 0.0
    t0 = float(t.min())
    early = pupil.loc[t.le(t0 + early_window_ms)]
    post = pupil.loc[t.gt(t0 + early_window_ms)]
    early_ref = float(early.median()) if early.notna().any() else np.nan
    post_median = float(post.median()) if post.notna().any() else np.nan
    delta = post_median - early_ref if np.isfinite(early_ref) and np.isfinite(post_median) else np.nan
    valid = pupil.notna() & t.notna()
    slope = float(np.polyfit((t.loc[valid] - t0) / 1000.0, pupil.loc[valid], 1)[0]) if valid.sum() >= 2 else np.nan
    return {
        "pupil_early_reference_mm": early_ref,
        "pupil_post_early_delta_mm": delta,
        "pupil_post_early_change_pct": 100.0 * delta / early_ref if np.isfinite(delta) and early_ref != 0 else np.nan,
        "pupil_slope_mm_per_s": slope,
        "pupil_valid_coverage_ratio": coverage,
        "pupil_binocular_coverage_ratio": binocular,
        "pupil_interpretation_tier": "exploratory_scene_early_reference_luminance_confounded",
    }


def _dynamic_qc(df: pd.DataFrame, fixations: pd.DataFrame, metadata: dict, timestamp_gap_ms: float, metrics: dict) -> dict:
    t = pd.to_numeric(df.get(TIME_COL), errors="coerce") if TIME_COL in df else pd.Series(np.nan, index=df.index)
    diff = t.diff().dropna()
    positive = diff.loc[(diff > 0) & (diff <= timestamp_gap_ms)]
    tracking = pd.to_numeric(df.get("Tracking Ratio[%]"), errors="coerce") if "Tracking Ratio[%]" in df else pd.Series(dtype=float)
    return {
        "raw_tracking_ratio_pct": float(tracking.median()) if tracking.notna().any() else np.nan,
        "valid_fixation_count": int(len(fixations)),
        "effective_sampling_rate_hz": 1000.0 / float(positive.median()) if not positive.empty and positive.median() > 0 else np.nan,
        "sampling_interval_iqr_ms": float(positive.quantile(.75) - positive.quantile(.25)) if not positive.empty else np.nan,
        "max_timestamp_gap_ms": float(diff.max()) if not diff.empty else np.nan,
        "fixation_in_aoi_canvas_ratio": float(fixations["coordinate_in_aoi_canvas"].mean()) if not fixations.empty else np.nan,
        "aoi_available_class_count": int(metadata.get("aoi_class_count", len({aoi for aoi in fixations.get("aoi_state", pd.Series(dtype=str)).unique() if aoi != "outside"}))),
        "aoi_ambiguous_fixation_ratio": float(fixations["aoi_ambiguous"].mean()) if not fixations.empty else np.nan,
        "coordinate_contract_status": metadata.get("coordinate_contract_status", "not_applicable_no_aoi"),
        **{k: metadata.get(k) for k in ["aoi_image_width_px", "aoi_image_height_px", "aoi_tool_name", "aoi_tool_version", "aoi_exported_at", "aoi_sha256"]},
        "dynamic_fixation_metrics_available": not fixations.empty,
        "dynamic_saccade_metrics_available": np.isfinite(metrics.get("saccade_count", np.nan)),
        "dynamic_pupil_metrics_available": np.isfinite(metrics.get("pupil_early_reference_mm", np.nan)),
    }


def _revisit_summary(fixations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if fixations.empty:
        return pd.DataFrame(columns=["class_name", "median_revisit_latency_ms", "revisit_count"])
    for class_name in sorted(set(fixations["aoi_state"]) - {"outside"}):
        latencies = []
        for _, sub in fixations.groupby("segment_id"):
            seq = _collapse_states(sub)
            prior_end = None
            left = False
            for item in seq:
                if item["aoi_state"] == class_name:
                    if prior_end is not None and left:
                        latencies.append(max(float(item["fixation_start_ms"]) - prior_end, 0.0))
                    prior_end = float(item["fixation_end_ms"])
                    left = False
                elif prior_end is not None:
                    left = True
        rows.append({"class_name": class_name, "median_revisit_latency_ms": float(np.median(latencies)) if latencies else np.nan, "revisit_count": len(latencies)})
    return pd.DataFrame(rows)


def _collapse_states(sub: pd.DataFrame) -> list[dict]:
    records = sub.sort_values("fixation_start_ms").to_dict("records")
    return _collapse_state_records(records)


def _collapse_state_records(records: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for record in records:
        item = dict(record)
        if out and out[-1]["aoi_state"] == item["aoi_state"]:
            out[-1]["fixation_end_ms"] = max(float(out[-1]["fixation_end_ms"]), float(item["fixation_end_ms"]))
            continue
        item["_seq_pos"] = len(out)
        out.append(item)
    return out


def _transition_row(prev: dict, cur: dict, segment_id: int, scope: str, index: int, passed_outside: bool) -> dict:
    return {
        "transition_scope": scope,
        "transition_index": index,
        "segment_id": int(segment_id),
        "from_aoi": prev["aoi_state"],
        "to_aoi": cur["aoi_state"],
        "from_fixation_index": prev.get("fixation_index"),
        "to_fixation_index": cur.get("fixation_index"),
        "transition_time_ms": cur.get("fixation_start_ms"),
        "transition_gap_ms": max(float(cur.get("fixation_start_ms", np.nan)) - float(prev.get("fixation_end_ms", np.nan)), 0.0),
        "passed_outside": bool(passed_outside),
    }


def _class_areas(aois: list[PolygonAOI]) -> dict[str, float]:
    areas: dict[str, float] = {}
    for aoi in aois:
        areas[aoi.class_name] = areas.get(aoi.class_name, 0.0) + polygon_area(aoi.points)
    return areas


def _point_aoi_candidates(x: float, y: float, aois: list[PolygonAOI]) -> list[str]:
    if not np.isfinite(x) or not np.isfinite(y):
        return []
    candidates = {aoi.class_name for aoi in aois if bool(point_in_poly(np.array([x]), np.array([y]), aoi.points)[0])}
    return sorted(candidates)


def _mean_gaze_direction(sub: pd.DataFrame) -> tuple[float, float, float]:
    vectors = []
    for eye in ["Left", "Right"]:
        cols = [f"Gaze Direction {eye} {axis}" for axis in "XYZ"]
        if not set(cols).issubset(sub.columns):
            continue
        vec = sub[cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(dtype=float)
        if np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
            vectors.append(vec / np.linalg.norm(vec))
    if not vectors:
        return (np.nan, np.nan, np.nan)
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    return tuple((mean / norm).tolist()) if norm > 0 else (np.nan, np.nan, np.nan)


def _valid_duration_s(df: pd.DataFrame, gap_ms: float) -> float:
    if TIME_COL not in df:
        return np.nan
    t = pd.to_numeric(df[TIME_COL], errors="coerce").dropna()
    if len(t) < 2:
        return np.nan
    diff = t.diff().dropna()
    return float(diff.loc[(diff > 0) & (diff <= gap_ms)].sum() / 1000.0)


def _event_crosses_aoi(sub: pd.DataFrame, aois: list[PolygonAOI]) -> bool | float:
    if not {"Gaze Point X[px]", "Gaze Point Y[px]"}.issubset(sub.columns) or sub.empty:
        return np.nan
    x = pd.to_numeric(sub["Gaze Point X[px]"], errors="coerce")
    y = pd.to_numeric(sub["Gaze Point Y[px]"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return np.nan
    first = _point_aoi_candidates(float(x.loc[valid].iloc[0]), float(y.loc[valid].iloc[0]), aois)
    last = _point_aoi_candidates(float(x.loc[valid].iloc[-1]), float(y.loc[valid].iloc[-1]), aois)
    return bool(first and last and first[0] != last[0])


def _interpolate_short_gaps(t: pd.Series, values: pd.Series, max_gap_ms: float) -> pd.Series:
    out = values.copy()
    candidate = values.interpolate(limit_direction="both")
    valid_idx = np.flatnonzero(values.notna().to_numpy())
    if len(valid_idx) < 2:
        return out
    missing_idx = np.flatnonzero(values.isna().to_numpy())
    times = t.to_numpy(dtype=float)
    for idx in missing_idx:
        before = valid_idx[valid_idx < idx]
        after = valid_idx[valid_idx > idx]
        if len(before) and len(after) and np.isfinite(times[before[-1]]) and np.isfinite(times[after[0]]) and times[after[0]] - times[before[-1]] <= max_gap_ms:
            out.iloc[idx] = candidate.iloc[idx]
    return out


def _median_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.median()) if values.notna().any() else np.nan


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _empty_saccade_metrics(zero_count: bool = False, duration_s: float = np.nan) -> dict:
    count = 0 if zero_count else np.nan
    return {
        "saccade_count": count,
        "saccade_rate_per_min": 0.0 if zero_count and duration_s > 0 else np.nan,
        "saccade_amplitude_median_px": np.nan,
        "saccade_velocity_average_median_px_per_ms": np.nan,
        "saccade_velocity_peak_median_px_per_ms": np.nan,
        "cross_aoi_saccade_ratio": np.nan,
    }


def _empty_pupil_metrics() -> dict:
    return {
        "pupil_early_reference_mm": np.nan,
        "pupil_post_early_delta_mm": np.nan,
        "pupil_post_early_change_pct": np.nan,
        "pupil_slope_mm_per_s": np.nan,
        "pupil_valid_coverage_ratio": np.nan,
        "pupil_binocular_coverage_ratio": np.nan,
        "pupil_interpretation_tier": "exploratory_scene_early_reference_luminance_confounded",
    }


def _fixation_columns() -> list[str]:
    return [
        "fixation_index", "segment_id", "fixation_start_ms", "fixation_end_ms", "fixation_duration_ms",
        "fixation_x_px", "fixation_y_px", "gaze_direction_x", "gaze_direction_y", "gaze_direction_z",
        "aoi_state", "aoi_candidates", "aoi_ambiguous", "coordinate_in_aoi_canvas",
    ]


def _transition_columns() -> list[str]:
    return [
        "transition_scope", "transition_index", "segment_id", "from_aoi", "to_aoi",
        "from_fixation_index", "to_fixation_index", "transition_time_ms", "transition_gap_ms", "passed_outside",
    ]
