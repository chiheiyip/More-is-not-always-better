from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.signal import welch

from paper_analysis.eye_tracking.aoi import compute_aoi_metrics, compute_whole_scene_metrics, load_aoi_json
from paper_analysis.eye_tracking.qc import EyeQCPolicy, valid_eye_mask
from paper_analysis.utils.io import read_table, write_table


COMPACT_EYE_COLUMNS = [
    "Recording Time Stamp[ms]", "Time of Day[HH:mm:ss.ms]", "Video Time[HH:mm:ss.ms]",
    "Validity Left", "Validity Right", "Gaze Point X[px]", "Gaze Point Y[px]",
    "Fixation Index", "Fixation Duration[ms]", "Fixation Point X[px]", "Fixation Point Y[px]",
    "Saccade Index", "Saccade Duration[ms]", "Saccade Amplitude[px]",
    "Blink Index", "Blink Duration[ms]", "Pupil Diameter Left[mm]", "Pupil Diameter Right[mm]",
]
ROIS = {"F": ("F3", "F4"), "P": ("P3", "Pz", "P4"), "O": ("O1", "Oz", "O2")}
BANDS = {"theta": (4.0, 7.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0)}


def run_clock_synchronized_fusion(
    scene_manifest_csv: str | Path,
    eeg_sample_manifest_csv: str | Path,
    outdir: str | Path,
    bin_size_ms: int = 2000,
    match_tolerance_ms: int = 2,
    timezone_name: str = "Asia/Shanghai",
    export_pointwise: bool = True,
    build_timebins: bool = True,
    eye_point_source: str = "auto",
    eye_screen_w: int | None = None,
    eye_screen_h: int | None = None,
    eye_validity_accepted: tuple[str, ...] | None = None,
    eye_timestamp_gap_ms: float = 5000.0,
    onset_trim_s: float = 0.0,
    onset_trim_variants_s: tuple[float, ...] | list[float] | None = None,
) -> dict[str, Path]:
    scene = read_table(scene_manifest_csv)
    eeg_manifest = read_table(eeg_sample_manifest_csv)
    required_scene = {"participant_id", "scene_id", "eye_csv_path"}
    required_eeg = {"participant_id", "scene_id", "eeg_sample_csv_path"}
    if not required_scene.issubset(scene.columns):
        raise ValueError(f"Scene manifest missing {sorted(required_scene - set(scene.columns))}")
    if not required_eeg.issubset(eeg_manifest.columns):
        raise ValueError(f"EEG sample manifest missing {sorted(required_eeg - set(eeg_manifest.columns))}")
    merged = scene.merge(eeg_manifest, on=["participant_id", "scene_id"], how="inner", suffixes=("", "_eeg"))
    if merged.empty:
        raise ValueError("No participant_id + scene_id overlap between eye and EEG sample manifests")

    outdir = Path(outdir)
    pointwise_root = outdir / "aligned_pointwise"
    qc_rows: list[dict] = []
    timebin_rows: list[dict] = []
    parallel_trims = tuple(dict.fromkeys(
        float(value)
        for value in (
            onset_trim_variants_s
            if onset_trim_variants_s is not None
            else (onset_trim_s,)
        )
    ))
    if not any(np.isclose(value, float(onset_trim_s)) for value in parallel_trims):
        parallel_trims = (*parallel_trims, float(onset_trim_s))
    parallel_trims = tuple(sorted(parallel_trims))
    pointwise_rows: list[dict] = []
    for _, trial in merged.sort_values(["participant_id", "scene_id"]).iterrows():
        participant_id = str(trial["participant_id"])
        scene_id = int(trial["scene_id"])
        eye_path = Path(str(trial["eye_csv_path"]))
        eeg_path = Path(str(trial["eeg_sample_csv_path"]))
        if not eye_path.exists() or not eeg_path.exists():
            qc_rows.append({
                "participant_id": participant_id, "scene_id": scene_id,
                "clock_alignment_pass": False,
                "clock_alignment_reasons": "missing_eye_or_eeg_sample_file",
            })
            continue
        eye = add_eye_epoch_ms(read_table(eye_path), trial, timezone_name=timezone_name)
        eeg = read_table(eeg_path)
        aligned, qc = align_eye_to_eeg(
            eye, eeg, participant_id=participant_id, scene_id=scene_id,
            match_tolerance_ms=match_tolerance_ms,
        )
        qc = _apply_manifest_qc(qc, trial)
        qc_rows.append({**_trial_metadata(trial), **qc})
        if export_pointwise:
            participant_root = pointwise_root / _safe_name(participant_id)
            participant_root.mkdir(parents=True, exist_ok=True)
            pointwise_path = participant_root / f"scene_{scene_id:02d}_eye_eeg_aligned.csv"
            aligned.to_csv(pointwise_path, index=False, encoding="utf-8-sig")
            pointwise_rows.append({
                "participant_id": participant_id,
                "scene_id": scene_id,
                "pointwise_output_path": str(pointwise_path),
            })
        if build_timebins:
            for parallel_trim in parallel_trims:
                timebin_rows.extend(synchronized_timebins(
                    eye, eeg, trial, bin_size_ms=bin_size_ms,
                    eye_point_source=eye_point_source, eye_screen_w=eye_screen_w,
                    eye_screen_h=eye_screen_h, eye_validity_accepted=eye_validity_accepted,
                    eye_timestamp_gap_ms=eye_timestamp_gap_ms,
                    onset_trim_s=float(parallel_trim),
                ))

    scene_qc = pd.DataFrame(qc_rows)
    outputs = {
        "clock_alignment_scene_qc": write_table(scene_qc, outdir / "clock_alignment_scene_qc.csv"),
        "clock_alignment_participant_qc": write_table(summarize_clock_alignment(scene_qc), outdir / "clock_alignment_participant_qc.csv"),
    }
    if build_timebins:
        outputs["aligned_synchronized_timebin"] = write_table(
            pd.DataFrame(timebin_rows), outdir / "aligned_synchronized_timebin_table.csv"
        )
        if onset_trim_variants_s is not None:
            # Compatibility alias: the canonical synchronized table now
            # contains every parallel onset window, so a second 0/5/15-only
            # copy would be both ambiguous and unnecessarily large.
            outputs["aligned_synchronized_timebin_onset_sensitivity"] = outputs[
                "aligned_synchronized_timebin"
            ]
    if export_pointwise:
        outputs["aligned_pointwise_index"] = write_table(
            pd.DataFrame(pointwise_rows), outdir / "aligned_pointwise_index.csv"
        )
    return outputs


def combine_parallel_synchronized_timebins(
    reference_timebin_csv: str | Path,
    variant_timebin_csv: str | Path,
    eeg_onset_trial_csv: str | Path,
    outdir: str | Path,
    expected_onset_trims_s: tuple[float, ...] | list[float] = (0, 5, 10, 15),
) -> dict[str, Path]:
    """Combine previously validated synchronized exports without recomputation.

    The legacy workflow wrote the 10-second reference export separately from
    the 0/5/15-second sensitivity export.  Both files were calculated from the
    same absolute-clock bins.  This function verifies their overlap, restores
    the common scene-relative time axis from the EEG trial duration table, and
    emits one canonical four-window table.
    """
    reference = read_table(reference_timebin_csv)
    variants = read_table(variant_timebin_csv)
    required = {
        "participant_id", "scene_id", "onset_trim_s", "class_name",
        "bin_start_epoch_ms", "bin_end_epoch_ms", "scene_elapsed_s",
        "analysis_elapsed_s",
    }
    for name, frame in (("reference", reference), ("variants", variants)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} synchronized table missing {sorted(missing)}")

    combined = pd.concat([reference, variants], ignore_index=True, sort=False)
    combined["onset_trim_s"] = pd.to_numeric(
        combined["onset_trim_s"], errors="coerce"
    )
    expected = tuple(sorted({float(value) for value in expected_onset_trims_s}))
    observed = tuple(sorted(combined["onset_trim_s"].dropna().unique().tolist()))
    if observed != expected:
        raise ValueError(
            f"Expected parallel onset trims {expected}, observed {observed}"
        )
    unique_keys = [
        "participant_id", "scene_id", "onset_trim_s", "class_name",
        "bin_start_epoch_ms", "bin_end_epoch_ms",
    ]
    duplicated = combined.duplicated(unique_keys, keep=False)
    if duplicated.any():
        sample = combined.loc[duplicated, unique_keys].head(10).to_dict("records")
        raise ValueError(f"Duplicate synchronized parallel bins: {sample}")

    duration = read_table(eeg_onset_trial_csv)
    duration_required = {
        "participant_id", "scene_id", "onset_trim_s", "view_dur_s"
    }
    missing_duration = duration_required - set(duration.columns)
    if missing_duration:
        raise ValueError(
            f"EEG onset trial table missing {sorted(missing_duration)}"
        )
    duration["onset_trim_s"] = pd.to_numeric(
        duration["onset_trim_s"], errors="coerce"
    )
    duration["view_dur_s"] = pd.to_numeric(duration["view_dur_s"], errors="coerce")
    duration = (
        duration.loc[np.isclose(duration["onset_trim_s"], expected[0]),
                     ["participant_id", "scene_id", "view_dur_s"]]
        .drop_duplicates(["participant_id", "scene_id"])
        .rename(columns={"view_dur_s": "scene_duration_s"})
    )
    combined = combined.drop(
        columns=["scene_duration_s", "scene_time_norm"], errors="ignore"
    ).merge(
        duration,
        on=["participant_id", "scene_id"],
        how="left",
        validate="many_to_one",
    )
    if combined["scene_duration_s"].isna().any():
        missing = combined.loc[
            combined["scene_duration_s"].isna(), ["participant_id", "scene_id"]
        ].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Missing exact EEG scene duration for synchronized trials: {missing}")
    combined["scene_time_norm"] = (
        pd.to_numeric(combined["scene_elapsed_s"], errors="coerce")
        / combined["scene_duration_s"]
    )
    combined["analysis_status"] = "parallel"
    combined["hypothesis_family"] = "parallel_synchronized_time_dynamics"

    feature_columns = [
        column for column in combined.columns
        if column.startswith("eeg_") or column in {
            "eeg_sample_count", "eeg_window_coverage", "eye_sample_count",
            "eye_window_coverage", "eye_valid_sample_count",
            "eye_valid_sample_coverage", "polygon_count", "samples", "TFD_ms",
            "TFD", "fixation_count", "FC", "TTFF_ms", "TTFF", "FFD_ms",
            "FFD", "MFD_ms", "MFD", "RFF", "MPD", "visited",
            "attention_share", "share", "share_pct", "FC_share", "FC_prop",
            "FC_rate", "FCR", "TFD_total_trial", "FC_total_trial",
        }
    ]
    combined["_feature_hash"] = pd.util.hash_pandas_object(
        combined[feature_columns], index=False
    ).astype("uint64")
    absolute_keys = [
        "participant_id", "scene_id", "class_name",
        "bin_start_epoch_ms", "bin_end_epoch_ms",
    ]
    audit_rows: list[dict] = []
    for left_trim, right_trim in combinations(expected, 2):
        left = combined.loc[np.isclose(combined["onset_trim_s"], left_trim)]
        right = combined.loc[np.isclose(combined["onset_trim_s"], right_trim)]
        overlap = left[absolute_keys + ["_feature_hash"]].merge(
            right[absolute_keys + ["_feature_hash"]],
            on=absolute_keys,
            how="inner",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        mismatches = int(
            overlap["_feature_hash_left"].ne(overlap["_feature_hash_right"]).sum()
        )
        audit_rows.append({
            "left_onset_trim_s": left_trim,
            "right_onset_trim_s": right_trim,
            "overlap_rows": int(len(overlap)),
            "feature_mismatch_rows": mismatches,
            # A 5-second trim shifts a 2-second bin lattice by one second, so
            # some window pairs legitimately have no identical bin interval.
            # Their placement is checked by the scene-axis residual below.
            "absolute_clock_alignment_pass": bool(mismatches == 0),
        })
    audit = pd.DataFrame(audit_rows)
    if not audit["absolute_clock_alignment_pass"].all():
        raise ValueError(
            "Legacy synchronized exports failed absolute-clock overlap validation"
        )
    alignment_residual = (
        pd.to_numeric(combined["scene_elapsed_s"], errors="coerce")
        - pd.to_numeric(combined["analysis_elapsed_s"], errors="coerce")
        - combined["onset_trim_s"]
    ).abs()
    if alignment_residual.max() > 1e-9:
        raise ValueError(
            "Parallel bins are not positioned on the same scene-relative axis"
        )
    audit["max_scene_axis_residual_s"] = float(alignment_residual.max())
    audit["validation_status"] = "pass"

    combined = combined.drop(columns="_feature_hash").sort_values(
        ["participant_id", "scene_id", "onset_trim_s", "bin_start_epoch_ms", "class_name"]
    )
    outdir = Path(outdir)
    table_path = write_table(
        combined, outdir / "aligned_synchronized_timebin_table.csv"
    )
    return {
        "aligned_synchronized_timebin": table_path,
        "aligned_synchronized_timebin_onset_sensitivity": table_path,
        "parallel_timebin_alignment_audit": write_table(
            audit, outdir / "parallel_timebin_alignment_audit.csv"
        ),
    }


def add_eye_epoch_ms(df: pd.DataFrame, trial: pd.Series | dict, timezone_name: str = "Asia/Shanghai") -> pd.DataFrame:
    out = df.copy()
    time_col = "Time of Day[HH:mm:ss.ms]"
    if time_col not in out.columns:
        out["eye_epoch_ms"] = np.nan
        return out
    base = pd.Timestamp(_eye_date(trial), tz=ZoneInfo(timezone_name))
    delta = pd.to_timedelta(out[time_col].astype(str).str.strip(), errors="coerce")
    tod_ms = delta.dt.total_seconds() * 1000.0
    day_rollover = (tod_ms.diff() < -(12 * 60 * 60 * 1000)).fillna(False).cumsum()
    out["eye_epoch_ms"] = base.value // 1_000_000 + tod_ms + day_rollover * 86_400_000
    return out


def align_eye_to_eeg(
    eye: pd.DataFrame,
    eeg: pd.DataFrame,
    participant_id: str,
    scene_id: int,
    match_tolerance_ms: int = 2,
) -> tuple[pd.DataFrame, dict]:
    if "eeg_epoch_ms" not in eeg.columns:
        raise ValueError("EEG sample table lacks eeg_epoch_ms")
    eye_work = eye.copy()
    eye_work.insert(0, "eye_source_row", np.arange(1, len(eye_work) + 1))
    compact = [c for c in COMPACT_EYE_COLUMNS if c in eye_work.columns]
    eye_work = eye_work[["eye_source_row", "eye_epoch_ms", *compact]].copy()
    eye_work["eye_epoch_ms"] = pd.to_numeric(eye_work["eye_epoch_ms"], errors="coerce").astype(float)
    eye_valid = eye_work.dropna(subset=["eye_epoch_ms"]).sort_values("eye_epoch_ms")

    eeg_cols = [c for c in eeg.columns if c.startswith("preproc_")]
    eeg_work = eeg[["sample_index_recording", "sample_index_scene", "eeg_epoch_ms", "eeg_scene_time_ms", *eeg_cols]].copy()
    eeg_work["eeg_epoch_ms"] = pd.to_numeric(eeg_work["eeg_epoch_ms"], errors="coerce").astype(float)
    eeg_work = eeg_work.dropna(subset=["eeg_epoch_ms"]).sort_values("eeg_epoch_ms")
    eeg_work = eeg_work.rename(columns={"eeg_epoch_ms": "matched_eeg_epoch_ms"})
    aligned_valid = pd.merge_asof(
        eye_valid, eeg_work, left_on="eye_epoch_ms", right_on="matched_eeg_epoch_ms",
        direction="nearest", tolerance=match_tolerance_ms,
    )
    matched_columns = [
        "eye_source_row", "sample_index_recording", "sample_index_scene",
        "matched_eeg_epoch_ms", "eeg_scene_time_ms", *eeg_cols,
    ]
    aligned = eye_work.merge(
        aligned_valid[matched_columns],
        on="eye_source_row",
        how="left",
        validate="one_to_one",
    )
    aligned.insert(0, "scene_id", scene_id)
    aligned.insert(0, "participant_id", participant_id)
    eeg_start = float(eeg_work["matched_eeg_epoch_ms"].min())
    eeg_end = float(eeg_work["matched_eeg_epoch_ms"].max())
    aligned["match_delta_ms"] = aligned["matched_eeg_epoch_ms"] - aligned["eye_epoch_ms"]
    aligned["alignment_status"] = np.select(
        [
            aligned["eye_epoch_ms"].isna(),
            aligned["eye_epoch_ms"] < eeg_start,
            aligned["eye_epoch_ms"] > eeg_end,
            aligned["matched_eeg_epoch_ms"].notna(),
        ],
        ["missing_eye_time", "before_eeg_view", "after_eeg_view", "matched"],
        default="unmatched_within_view",
    )
    aligned = aligned.rename(columns={
        "sample_index_recording": "eeg_sample_index",
        "matched_eeg_epoch_ms": "eeg_epoch_ms",
    })

    eye_times = pd.to_numeric(eye_work["eye_epoch_ms"], errors="coerce").dropna()
    eye_start = float(eye_times.min()) if not eye_times.empty else np.nan
    eye_end = float(eye_times.max()) if not eye_times.empty else np.nan
    eeg_duration = max(eeg_end - eeg_start, 0.0)
    overlap = max(0.0, min(eeg_end, eye_end) - max(eeg_start, eye_start)) if np.isfinite(eye_start) else 0.0
    overlap_ratio = overlap / eeg_duration if eeg_duration else 0.0
    in_view = aligned["eye_epoch_ms"].between(eeg_start, eeg_end, inclusive="both")
    matched = aligned["alignment_status"].eq("matched")
    match_rate = float((matched & in_view).sum() / in_view.sum()) if in_view.sum() else 0.0
    abs_delta = aligned.loc[matched, "match_delta_ms"].abs()
    p95 = float(abs_delta.quantile(0.95)) if not abs_delta.empty else np.nan
    reasons = []
    eye_monotonic = bool(eye_times.is_monotonic_increasing)
    eeg_monotonic = bool(pd.to_numeric(eeg["eeg_epoch_ms"], errors="coerce").dropna().is_monotonic_increasing)
    if not eye_monotonic:
        reasons.append("eye_time_not_monotonic")
    if not eeg_monotonic:
        reasons.append("eeg_time_not_monotonic")
    if overlap_ratio < 0.95:
        reasons.append("eeg_time_coverage_below_95pct")
    if match_rate < 0.95:
        reasons.append("eye_match_rate_below_95pct")
    if not np.isfinite(p95) or p95 > match_tolerance_ms:
        reasons.append("p95_match_delta_above_tolerance")
    qc = {
        "participant_id": participant_id, "scene_id": scene_id,
        "eye_first_epoch_ms": eye_start, "eye_last_epoch_ms": eye_end,
        "eeg_first_epoch_ms": eeg_start, "eeg_last_epoch_ms": eeg_end,
        "eye_rows": int(len(eye_work)), "eye_rows_in_eeg_view": int(in_view.sum()),
        "matched_eye_rows": int(matched.sum()), "eeg_time_coverage_ratio": overlap_ratio,
        "eye_match_rate": match_rate,
        "median_abs_match_delta_ms": float(abs_delta.median()) if not abs_delta.empty else np.nan,
        "p95_abs_match_delta_ms": p95, "clock_offset_ms": 0.0,
        "eye_time_monotonic": eye_monotonic, "eeg_time_monotonic": eeg_monotonic,
        "clock_basis": "same_windows_system_clock",
        "clock_alignment_pass": not reasons, "clock_alignment_reasons": ";".join(reasons),
    }
    return aligned, qc


def synchronized_timebins(
    eye: pd.DataFrame,
    eeg: pd.DataFrame,
    trial: pd.Series,
    bin_size_ms: int = 2000,
    eye_point_source: str = "auto",
    eye_screen_w: int | None = None,
    eye_screen_h: int | None = None,
    eye_validity_accepted: tuple[str, ...] | None = None,
    eye_timestamp_gap_ms: float = 5000.0,
    onset_trim_s: float = 0.0,
) -> list[dict]:
    eeg = eeg.copy()
    eye = eye.copy()
    eeg["eeg_epoch_ms"] = pd.to_numeric(eeg["eeg_epoch_ms"], errors="coerce")
    eye["eye_epoch_ms"] = pd.to_numeric(eye["eye_epoch_ms"], errors="coerce")
    eeg = eeg.dropna(subset=["eeg_epoch_ms"])
    if eeg.empty:
        return []
    srate = _sample_rate(eeg)
    eye_interval_ms = _eye_sample_interval(eye)
    if onset_trim_s < 0:
        raise ValueError("onset_trim_s must be non-negative")
    scene_start = float(eeg["eeg_epoch_ms"].min())
    start = scene_start + float(onset_trim_s) * 1000.0
    stop = int(eeg["eeg_epoch_ms"].max() + round(1000.0 / srate))
    n_bins = int((stop - start) // bin_size_ms)
    scene_duration_ms = max(float(stop) - scene_start, 0.0)
    aoi_path = Path(str(trial.get("aoi_json_path", "")))
    aois = load_aoi_json(aoi_path) if aoi_path.is_file() else None
    rows: list[dict] = []
    for bin_index in range(n_bins):
        bin_start = start + bin_index * bin_size_ms
        bin_end = min(bin_start + bin_size_ms, stop)
        eeg_sub = eeg.loc[eeg["eeg_epoch_ms"].ge(bin_start) & eeg["eeg_epoch_ms"].lt(bin_end)]
        eye_sub = eye.loc[eye["eye_epoch_ms"].ge(bin_start) & eye["eye_epoch_ms"].lt(bin_end)]
        eeg_metrics = _eeg_window_metrics(eeg_sub, srate)
        if not eye_sub.empty and aois is not None:
            _, metrics = compute_aoi_metrics(
                eye_sub, aois, point_source=eye_point_source, screen_w=eye_screen_w,
                screen_h=eye_screen_h, validity_accepted=eye_validity_accepted,
                timestamp_gap_ms=eye_timestamp_gap_ms,
            )
        elif not eye_sub.empty:
            metrics = compute_whole_scene_metrics(
                eye_sub, point_source=eye_point_source, screen_w=eye_screen_w,
                screen_h=eye_screen_h, validity_accepted=eye_validity_accepted,
                timestamp_gap_ms=eye_timestamp_gap_ms,
            )
        else:
            metrics = pd.DataFrame([{"class_name": "whole_scene", "visited": False, "samples": 0}])
        duration_ms = bin_end - bin_start
        expected_eeg = duration_ms * srate / 1000.0
        expected_eye = duration_ms / eye_interval_ms if np.isfinite(eye_interval_ms) else np.nan
        valid_eye_samples = _valid_eye_sample_count(
            eye_sub, eye_point_source, eye_screen_w, eye_screen_h, eye_validity_accepted
        )
        for metric in metrics.to_dict("records"):
            rows.append({
                **_trial_metadata(trial),
                "bin_index": bin_index, "bin_start_epoch_ms": bin_start, "bin_end_epoch_ms": bin_end,
                "bin_start_ms": bin_index * bin_size_ms,
                "bin_end_ms": bin_index * bin_size_ms + duration_ms,
                "scene_elapsed_s": (bin_start - scene_start) / 1000.0,
                "analysis_elapsed_s": bin_index * bin_size_ms / 1000.0,
                "onset_trim_s": float(onset_trim_s),
                "scene_duration_s": scene_duration_ms / 1000.0,
                "scene_time_norm": (
                    (bin_start - scene_start) / scene_duration_ms
                    if scene_duration_ms > 0 else np.nan
                ),
                "time_norm": bin_index / max(n_bins - 1, 1),
                "eeg_sample_count": int(len(eeg_sub)),
                "eeg_window_coverage": float(len(eeg_sub) / expected_eeg) if expected_eeg else 0.0,
                "eye_sample_count": int(len(eye_sub)),
                "eye_window_coverage": float(len(eye_sub) / expected_eye) if np.isfinite(expected_eye) and expected_eye else np.nan,
                "eye_valid_sample_count": valid_eye_samples,
                "eye_valid_sample_coverage": float(valid_eye_samples / expected_eye) if np.isfinite(expected_eye) and expected_eye else np.nan,
                "eeg_temporal_resolution": "window_specific",
                "analysis_resolution": "synchronized_timebin",
                "analysis_status": "parallel",
                "hypothesis_family": "parallel_synchronized_time_dynamics",
                "clock_qc_policy": "same_clock_epoch_nearest_2ms",
                **metric, **eeg_metrics,
            })
    return rows


def summarize_clock_alignment(scene_qc: pd.DataFrame) -> pd.DataFrame:
    if scene_qc.empty:
        return pd.DataFrame()
    rows = []
    for participant_id, sub in scene_qc.groupby("participant_id"):
        passed = sub.get("clock_alignment_pass", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        rows.append({
            "participant_id": participant_id, "scene_count": int(sub["scene_id"].nunique()),
            "passed_scene_count": int(passed.sum()),
            "all_12_scenes_present": int(sub["scene_id"].nunique()) == 12,
            "participant_clock_alignment_pass": bool(int(sub["scene_id"].nunique()) == 12 and passed.all()),
            "median_eye_match_rate": float(pd.to_numeric(sub.get("eye_match_rate"), errors="coerce").median()),
            "median_p95_abs_match_delta_ms": float(pd.to_numeric(sub.get("p95_abs_match_delta_ms"), errors="coerce").median()),
            "clock_basis": "same_windows_system_clock",
        })
    return pd.DataFrame(rows)


def _apply_manifest_qc(qc: dict, trial: pd.Series) -> dict:
    reasons = [value for value in str(qc.get("clock_alignment_reasons", "")).split(";") if value]
    checks = {
        "eeg_source_validated": str(trial.get("clock_cache_status", "")).startswith("pass"),
        "sample_export_pass": bool(_truthy_value(trial.get("sample_export_pass", False))),
        "srate_cache_match": bool(_truthy_value(trial.get("srate_cache_match", False))),
        "sample_time_continuous": bool(_truthy_value(trial.get("time_continuous", False))),
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name + "_failed")
    qc.update(checks)
    qc["clock_alignment_pass"] = not reasons
    qc["clock_alignment_reasons"] = ";".join(dict.fromkeys(reasons))
    return qc


def _eeg_window_metrics(eeg: pd.DataFrame, srate: float) -> dict:
    out: dict[str, float] = {}
    for roi, channels in ROIS.items():
        cols = [f"preproc_{channel}_uV" for channel in channels if f"preproc_{channel}_uV" in eeg.columns]
        signal = eeg[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy(dtype=float) if cols else np.array([])
        signal = signal[np.isfinite(signal)]
        if len(signal) < max(8, int(srate)):
            out.update({f"eeg_{roi}_{band}": np.nan for band in BANDS})
            continue
        nperseg = min(len(signal), max(int(round(2 * srate)), 8))
        freq, density = welch(signal, fs=srate, nperseg=nperseg, noverlap=nperseg // 2)
        for band, (low, high) in BANDS.items():
            mask = (freq >= low) & (freq <= high)
            out[f"eeg_{roi}_{band}"] = (
                _trapezoidal_integral(density[mask], freq[mask])
                if mask.any()
                else np.nan
            )
    return out


def _trapezoidal_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate without relying on NumPy's removed ``np.trapz`` alias."""
    if len(y) < 2 or len(x) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))


def _sample_rate(eeg: pd.DataFrame) -> float:
    times = pd.to_numeric(eeg["eeg_epoch_ms"], errors="coerce").dropna().sort_values()
    interval = times.diff().loc[lambda value: value.gt(0)].median()
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Could not infer EEG sample rate from eeg_epoch_ms")
    return 1000.0 / float(interval)


def _eye_sample_interval(eye: pd.DataFrame) -> float:
    times = pd.to_numeric(eye["eye_epoch_ms"], errors="coerce").dropna()
    interval = times.diff().loc[lambda value: value.gt(0)].median()
    return float(interval) if np.isfinite(interval) and interval > 0 else np.nan


def _valid_eye_sample_count(
    eye: pd.DataFrame,
    point_source: str,
    screen_w: int | None,
    screen_h: int | None,
    validity_accepted: tuple[str, ...] | None,
) -> int:
    if eye.empty:
        return 0
    fixation = ("Fixation Point X[px]", "Fixation Point Y[px]")
    gaze = ("Gaze Point X[px]", "Gaze Point Y[px]")
    if point_source in {"auto", "fixation"} and set(fixation).issubset(eye.columns):
        x_col, y_col = fixation
    elif set(gaze).issubset(eye.columns):
        x_col, y_col = gaze
    else:
        return 0
    policy = EyeQCPolicy(
        screen_w=screen_w, screen_h=screen_h,
        validity_accepted=validity_accepted,
    )
    mask, _ = valid_eye_mask(eye, x_col, y_col, policy)
    return int(mask.sum())


def _eye_date(trial: pd.Series | dict) -> str:
    record_id = str(trial.get("eye_record_id", "") or "")
    match = re.match(r"(\d{6})", record_id)
    if match:
        return pd.to_datetime(match.group(1), format="%y%m%d").strftime("%Y-%m-%d")
    collection_date = trial.get("collection_date")
    if collection_date is not None and str(collection_date).strip():
        return pd.Timestamp(collection_date).strftime("%Y-%m-%d")
    raise ValueError("Eye date unavailable: expected YYMMDD prefix in eye_record_id or collection_date")


def _trial_metadata(trial: pd.Series | dict) -> dict:
    columns = [
        "participant_id", "scene_id", "condition_id", "WWR", "WWR_numeric", "Cond", "Complexity",
        "block", "position", "round", "ExperienceGroup", "Gender", "Age", "RecruitmentBatch",
        "eye_record_id", "eye_csv_path", "aoi_json_path",
    ]
    return {column: trial.get(column) for column in columns if column in trial}


def _safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", value).strip() or "participant"


def _truthy_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
