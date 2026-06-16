from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.core.schema import DESIGN_COLUMNS, KEYS
from paper_analysis.core.tables import (
    build_trial_index,
    canonical_columns_first,
    coalesce_design_columns,
    eeg_duration_table,
    prefix_non_core_columns,
    standardize_eeg_scene,
)
from paper_analysis.eye_tracking.aoi import (
    compute_timebin_aoi_metrics,
    compute_whole_scene_timebin_metrics,
    eye_file_stats,
    load_aoi_json,
)
from paper_analysis.utils.coding import active_rows, standardize_participants
from paper_analysis.utils.io import read_table, resolve_path, write_table


def run_fusion_pipeline(
    questionnaire_long: str | Path,
    eye_aoi_trial_long: str | Path,
    eeg_trial_long: str | Path,
    participants_csv: str | Path,
    scene_manifest_csv: str | Path | None = None,
    outdir: str | Path = "outputs/05_multimodal_fusion",
    bin_size_ms: int = 2000,
    duration_tolerance_s: float = 2.0,
    expected_scenes_per_subject: int = 12,
    duration_outlier_s: float = 5.0,
    residual_outlier_ms: float = 3000.0,
) -> dict[str, Path]:
    questionnaire = read_table(questionnaire_long)
    eye = read_table(eye_aoi_trial_long)
    participants = standardize_participants(read_table(participants_csv))
    eeg = standardize_eeg_scene(read_table(eeg_trial_long), participants)
    scene = _load_scene_for_fusion(scene_manifest_csv, questionnaire, eye, eeg)
    trial_index = build_trial_index(participants, scene, active_only=True)

    aligned_scene = build_aligned_scene_table(trial_index, eye, eeg)
    aligned_timebin = build_aligned_timebin_table(
        trial_index=trial_index,
        eeg=eeg,
        bin_size_ms=bin_size_ms,
    )
    sync_qc = build_sync_qc(
        trial_index=trial_index,
        eeg=eeg,
        bin_size_ms=bin_size_ms,
        duration_tolerance_s=duration_tolerance_s,
        expected_scenes_per_subject=expected_scenes_per_subject,
    )
    alignment_scene_qc, alignment_landmarks, time_sync_map = build_precise_alignment_qc(
        trial_index=trial_index,
        eeg=eeg,
        duration_outlier_s=duration_outlier_s,
        residual_outlier_ms=residual_outlier_ms,
    )

    q_pref = prefix_non_core_columns(questionnaire, "q")
    eeg_pref = prefix_non_core_columns(eeg, "eeg")
    master_pre_qc = trial_index.merge(q_pref, on=KEYS, how="left", suffixes=("", "_qdup"))
    master_pre_qc = master_pre_qc.merge(eeg_pref, on=KEYS, how="left", suffixes=("", "_eegdup"))
    if not eye.empty:
        master_pre_qc = master_pre_qc.merge(eye, on=KEYS, how="left", suffixes=("", "_eye"))
    master_pre_qc = canonical_columns_first(coalesce_design_columns(master_pre_qc))
    analysis_qc = build_analysis_qc_exclusions(
        master=master_pre_qc,
        sync_qc=sync_qc,
        questionnaire=questionnaire,
        eye=eye,
        eeg=eeg,
    )
    master = apply_analysis_qc(master_pre_qc, analysis_qc)
    convergence = modality_convergence(master)
    claim_support = claim_support_matrix(convergence)
    outdir = Path(outdir)
    return {
        "analysis_master_long_pre_qc": write_table(master_pre_qc, outdir / "analysis_master_long_pre_qc.csv"),
        "analysis_qc_exclusions": write_table(analysis_qc, outdir / "analysis_qc_exclusions.csv"),
        "analysis_master_long": write_table(master, outdir / "analysis_master_long.csv"),
        "aligned_scene": write_table(aligned_scene, outdir / "aligned_scene_table.csv"),
        "aligned_timebin": write_table(aligned_timebin, outdir / "aligned_timebin_table.csv"),
        "sync_qc": write_table(sync_qc, outdir / "sync_qc.csv"),
        "alignment_scene_qc": write_table(alignment_scene_qc, outdir / "alignment_scene_qc.csv"),
        "alignment_landmarks": write_table(alignment_landmarks, outdir / "alignment_landmarks.csv"),
        "time_sync_map": write_table(time_sync_map, outdir / "time_sync_map.csv"),
        "modality_convergence_table": write_table(convergence, outdir / "modality_convergence_table.csv"),
        "claim_support_matrix": write_table(claim_support, outdir / "claim_support_matrix.csv"),
    }


def build_aligned_scene_table(trial_index: pd.DataFrame, eye: pd.DataFrame, eeg: pd.DataFrame) -> pd.DataFrame:
    if eye.empty:
        out = trial_index.merge(eeg, on=KEYS, how="left", suffixes=("", "_eeg"))
    else:
        out = eye.merge(trial_index, on=KEYS, how="right", suffixes=("_eye", ""))
        out = out.merge(eeg, on=KEYS, how="left", suffixes=("", "_eeg"))
    missing_col = _first_existing(out, ["subject_id", "view_dur_s", "dur_s", "duration_s", "O_theta", "F_theta"])
    out["missing_eeg_scene"] = out[missing_col].isna() if missing_col else True
    sort_cols = [c for c in ["participant_id", "scene_id", "class_name"] if c in out.columns]
    return canonical_columns_first(out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True))


def build_aligned_timebin_table(
    trial_index: pd.DataFrame,
    eeg: pd.DataFrame,
    bin_size_ms: int = 2000,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    eeg_pref = prefix_non_core_columns(eeg, "eeg")
    for _, trial in trial_index.iterrows():
        eye_csv = resolve_path(trial.get("eye_csv_path"))
        aoi_json = resolve_path(trial.get("aoi_json_path"))
        if not eye_csv or not eye_csv.exists():
            continue
        df = read_table(eye_csv)
        eye_offset_ms = _number_or_default(trial.get("eye_offset_ms"), 0.0)
        if aoi_json and aoi_json.exists():
            metrics = compute_timebin_aoi_metrics(df, load_aoi_json(aoi_json), bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
        else:
            metrics = compute_whole_scene_timebin_metrics(df, bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
        if metrics.empty:
            continue
        for col in [c for c in trial_index.columns if c not in metrics.columns]:
            metrics[col] = trial[col]
        rows.append(metrics)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEYS + ["bin_index", "bin_start_ms", "bin_end_ms", "class_name"])
    if not out.empty:
        out = out.merge(eeg_pref, on=KEYS, how="left", suffixes=("", "_eeg"))
        sort_cols = [c for c in ["participant_id", "scene_id", "bin_index", "class_name"] if c in out.columns]
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return canonical_columns_first(out)


def build_sync_qc(
    trial_index: pd.DataFrame,
    eeg: pd.DataFrame,
    bin_size_ms: int,
    duration_tolerance_s: float,
    expected_scenes_per_subject: int,
) -> pd.DataFrame:
    eeg_duration = eeg_duration_table(eeg)
    scene_counts = trial_index.groupby("participant_id")["scene_id"].nunique().rename("manifest_scene_count")
    rows: list[dict] = []
    for _, trial in trial_index.iterrows():
        participant_id = str(trial["participant_id"])
        scene_id = int(trial["scene_id"])
        eye_csv = resolve_path(trial.get("eye_csv_path"))
        eye_offset_ms = _number_or_default(trial.get("eye_offset_ms"), 0.0)
        base = _trial_base(trial)
        base.update({
            "eye_offset_ms": eye_offset_ms,
            "missing_eye_file": not eye_csv or not eye_csv.exists(),
            "manifest_scene_count": int(scene_counts.get(participant_id, 0)),
            "expected_scenes_per_subject": int(expected_scenes_per_subject),
        })
        base["scene_count_mismatch"] = base["manifest_scene_count"] != int(expected_scenes_per_subject)
        if base["missing_eye_file"]:
            eye_stats = {}
        else:
            eye_stats = eye_file_stats(read_table(eye_csv), bin_size_ms=bin_size_ms, eye_offset_ms=eye_offset_ms)
        eeg_row = eeg_duration.loc[(eeg_duration["participant_id"] == participant_id) & (eeg_duration["scene_id"] == scene_id)]
        eeg_view_dur_s = float(eeg_row["eeg_view_dur_s"].iloc[0]) if not eeg_row.empty else np.nan
        eye_duration_s = eye_stats.get("eye_duration_s", np.nan)
        delta = eye_duration_s - eeg_view_dur_s if pd.notna(eye_duration_s) and pd.notna(eeg_view_dur_s) else np.nan
        rows.append({
            **base,
            **eye_stats,
            "missing_eeg_scene": eeg_row.empty,
            "eeg_view_dur_s": eeg_view_dur_s,
            "duration_delta_s": delta,
            "duration_mismatch": bool(abs(delta) > duration_tolerance_s) if pd.notna(delta) else True,
        })
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def build_analysis_qc_exclusions(
    master: pd.DataFrame,
    sync_qc: pd.DataFrame,
    questionnaire: pd.DataFrame,
    eye: pd.DataFrame,
    eeg: pd.DataFrame,
) -> pd.DataFrame:
    trials = master[KEYS].drop_duplicates().copy()
    q_keys = _key_set(questionnaire)
    eye_keys = _key_set(eye)
    eeg_keys = _key_set(eeg)
    rows: list[dict] = []
    sync_lookup = sync_qc.drop_duplicates(KEYS).set_index(KEYS) if set(KEYS).issubset(sync_qc.columns) else pd.DataFrame()
    eeg_lookup = eeg.drop_duplicates(KEYS).set_index(KEYS) if set(KEYS).issubset(eeg.columns) else pd.DataFrame()
    for _, trial in trials.iterrows():
        key = (str(trial["participant_id"]), int(trial["scene_id"]))
        sync = sync_lookup.loc[key] if not sync_lookup.empty and key in sync_lookup.index else pd.Series(dtype=object)
        eeg_row = eeg_lookup.loc[key] if not eeg_lookup.empty and key in eeg_lookup.index else pd.Series(dtype=object)
        missing_questionnaire = key not in q_keys
        missing_eye = bool(sync.get("missing_eye_file", False)) or key not in eye_keys
        missing_eeg = bool(sync.get("missing_eeg_scene", False)) or key not in eeg_keys
        duration_mismatch = bool(sync.get("duration_mismatch", False))
        scene_count_mismatch = bool(sync.get("scene_count_mismatch", False))
        bad_eeg_quality = _truthy(eeg_row.get("bad_eeg_quality", False)) if not eeg_row.empty else False
        eeg_qc_reasons = str(eeg_row.get("eeg_qc_reasons", "") or "")
        eeg_qc_policy = str(eeg_row.get("eeg_qc_policy", "") or "")
        reasons = []
        if missing_questionnaire:
            reasons.append("missing_questionnaire")
        if missing_eye:
            reasons.append("missing_eye")
        if missing_eeg:
            reasons.append("missing_eeg")
        if bad_eeg_quality:
            reasons.append("bad_eeg_quality")
        if duration_mismatch:
            reasons.append("duration_mismatch")
        if scene_count_mismatch:
            reasons.append("scene_count_mismatch")
        rows.append({
            "participant_id": key[0],
            "scene_id": key[1],
            "excluded_from_analysis": bool(reasons),
            "analysis_exclusion_reasons": ";".join(reasons),
            "missing_questionnaire": missing_questionnaire,
            "missing_eye": missing_eye,
            "missing_eeg": missing_eeg,
            "bad_eeg_quality": bad_eeg_quality,
            "eeg_qc_reasons": eeg_qc_reasons,
            "eeg_qc_policy": eeg_qc_policy,
            "duration_mismatch": duration_mismatch,
            "scene_count_mismatch": scene_count_mismatch,
            "duration_delta_s": sync.get("duration_delta_s", np.nan),
            "manifest_scene_count": sync.get("manifest_scene_count", np.nan),
            "expected_scenes_per_subject": sync.get("expected_scenes_per_subject", np.nan),
        })
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def apply_analysis_qc(master: pd.DataFrame, analysis_qc: pd.DataFrame) -> pd.DataFrame:
    if analysis_qc.empty:
        return master.copy()
    keep = analysis_qc.loc[~analysis_qc["excluded_from_analysis"], KEYS]
    out = master.merge(keep, on=KEYS, how="inner")
    return canonical_columns_first(out.reset_index(drop=True))


def build_precise_alignment_qc(
    trial_index: pd.DataFrame,
    eeg: pd.DataFrame,
    duration_outlier_s: float = 5.0,
    residual_outlier_ms: float = 3000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not {"view_start_s", "view_end_s"}.issubset(eeg.columns):
        empty_scene = trial_index[KEYS].copy()
        empty_scene["alignment_available"] = False
        return empty_scene, pd.DataFrame(), pd.DataFrame()
    eye_timing = build_eye_timing_table(trial_index)
    eeg_timing = eeg[KEYS + ["view_start_s", "view_end_s"]].copy()
    eeg_timing["view_start_s"] = pd.to_numeric(eeg_timing["view_start_s"], errors="coerce")
    eeg_timing["view_end_s"] = pd.to_numeric(eeg_timing["view_end_s"], errors="coerce")
    eeg_timing["view_dur_s"] = eeg_duration_table(eeg)["eeg_view_dur_s"]
    return estimate_time_alignment(
        eye_timing=eye_timing,
        eeg_timing=eeg_timing,
        duration_outlier_s=duration_outlier_s,
        residual_outlier_ms=residual_outlier_ms,
    )


def build_eye_timing_table(trial_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, trial in trial_index.iterrows():
        eye_csv = resolve_path(trial.get("eye_csv_path"))
        base = _trial_base(trial)
        base["missing_eye_file"] = not eye_csv or not eye_csv.exists()
        if base["missing_eye_file"]:
            rows.append(base)
            continue
        df = read_table(eye_csv)
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
    valid = scene_qc.dropna(subset=["eye_start_ms", "eye_end_ms", "view_start_s", "view_end_s"])
    for participant_id, sub in valid.groupby("participant_id"):
        fit_sub = sub.loc[~sub["duration_outlier"].fillna(True)].copy()
        if len(fit_sub) < 2:
            fit_sub = sub.copy()
        x = np.r_[fit_sub["eye_start_ms"].to_numpy(), fit_sub["eye_end_ms"].to_numpy()]
        y = np.r_[fit_sub["view_start_s"].to_numpy() * 1000.0, fit_sub["view_end_s"].to_numpy() * 1000.0]
        slope, offset = _fit_affine(x, y)
        residual = (offset + slope * x) - y
        keep = np.abs(residual) <= residual_outlier_ms
        if keep.sum() >= 4:
            slope, offset = _fit_affine(x[keep], y[keep])
        for _, row in sub.iterrows():
            for landmark, eye_col, eeg_col in [("start", "eye_start_ms", "view_start_s"), ("end", "eye_end_ms", "view_end_s")]:
                eye_ms = row[eye_col]
                eeg_ms = row[eeg_col] * 1000.0
                pred = offset + slope * eye_ms
                landmark_rows.append({
                    "participant_id": participant_id,
                    "scene_id": int(row["scene_id"]),
                    "landmark": landmark,
                    "eye_time_ms": eye_ms,
                    "eeg_time_ms": eeg_ms,
                    "predicted_eeg_time_ms": pred,
                    "residual_ms": pred - eeg_ms,
                    "duration_outlier": bool(row.get("duration_outlier", False)),
                })
        participant_landmarks = pd.DataFrame([r for r in landmark_rows if r["participant_id"] == participant_id])
        usable = participant_landmarks.loc[~participant_landmarks["duration_outlier"]] if not participant_landmarks.empty else participant_landmarks
        if usable.empty:
            usable = participant_landmarks
        map_rows.append({
            "participant_id": participant_id,
            "time_sync_slope": slope,
            "time_sync_offset_ms": offset,
            "landmark_count": int(len(participant_landmarks)),
            "usable_landmark_count": int(len(usable)),
            "median_abs_residual_ms": float(usable["residual_ms"].abs().median()) if not usable.empty else np.nan,
            "p95_abs_residual_ms": float(usable["residual_ms"].abs().quantile(0.95)) if not usable.empty else np.nan,
            "max_abs_residual_ms": float(usable["residual_ms"].abs().max()) if not usable.empty else np.nan,
            "duration_outlier_count": int(sub["duration_outlier"].sum()),
        })
    return scene_qc, pd.DataFrame(landmark_rows), pd.DataFrame(map_rows)


def modality_convergence(master: pd.DataFrame) -> pd.DataFrame:
    families = {
        "questionnaire": [c for c in master.columns if c.startswith("q_") and c[2:] in {"S1", "S2", "S3", "S4", "S5"}],
        "eeg": [c for c in master.columns if c.startswith("eeg_") and any(k in c.lower() for k in ["theta", "alpha", "beta"])],
        "eye": [c for c in master.columns if c in {"FCR", "TFD_ms", "TTFF_ms", "attention_share", "visited"}],
    }
    rows = []
    for family, cols in families.items():
        for col in cols:
            values = pd.to_numeric(master[col], errors="coerce")
            rows.append({"modality": family, "metric": col, "n": int(values.notna().sum()), "mean": float(values.mean()) if values.notna().any() else None})
    return pd.DataFrame(rows)


def claim_support_matrix(convergence: pd.DataFrame) -> pd.DataFrame:
    modalities = set(convergence.loc[convergence["n"] > 0, "modality"]) if not convergence.empty else set()
    rows = [
        {"claim_id": "C1_WWR_NONLINEAR", "required_modalities": "questionnaire,eeg", "available_modalities": ",".join(sorted(modalities)), "support_level": _support(modalities, {"questionnaire", "eeg"})},
        {"claim_id": "C2_COMPLEXITY_PROCESSING", "required_modalities": "questionnaire,eeg,eye", "available_modalities": ",".join(sorted(modalities)), "support_level": _support(modalities, {"questionnaire", "eeg", "eye"})},
        {"claim_id": "C3_EXPERIENCE_MODERATION", "required_modalities": "questionnaire,eeg,eye", "available_modalities": ",".join(sorted(modalities)), "support_level": _support(modalities, {"questionnaire", "eeg", "eye"})},
    ]
    return pd.DataFrame(rows)


def _support(available: set[str], required: set[str]) -> str:
    if required.issubset(available):
        return "moderate_pending_effect_tests"
    if available:
        return "exploratory_incomplete_modalities"
    return "unsupported_no_data"


def _load_scene_for_fusion(
    scene_manifest_csv: str | Path | None,
    questionnaire: pd.DataFrame,
    eye: pd.DataFrame,
    eeg: pd.DataFrame,
) -> pd.DataFrame:
    if scene_manifest_csv is not None:
        from paper_analysis.core.tables import load_scene_table

        return load_scene_table(scene_manifest_csv)
    frames = [df for df in [questionnaire, eye, eeg] if set(KEYS).issubset(df.columns)]
    if not frames:
        raise ValueError("scene_manifest_csv is required when modality tables do not contain participant_id + scene_id")
    scene = pd.concat([df[[c for c in KEYS + DESIGN_COLUMNS if c in df.columns]] for df in frames], ignore_index=True)
    return scene.drop_duplicates(KEYS).reset_index(drop=True)


def _trial_base(trial: pd.Series) -> dict:
    out = {key: trial[key] for key in KEYS}
    for col in ["condition_id", "WWR", "WWR_numeric", "Cond", "Complexity", "block", "position", "round", "participant_order", "order_scheme", "order_code", "eye_csv_path", "aoi_json_path"]:
        if col in trial.index:
            out[col] = trial[col]
    return out


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    slope, offset = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(offset)


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _key_set(df: pd.DataFrame) -> set[tuple[str, int]]:
    if not set(KEYS).issubset(df.columns):
        return set()
    keys = df[KEYS].dropna().drop_duplicates()
    return {(str(row["participant_id"]), int(row["scene_id"])) for _, row in keys.iterrows()}


def _number_or_default(value: object, default: float) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def _truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False
