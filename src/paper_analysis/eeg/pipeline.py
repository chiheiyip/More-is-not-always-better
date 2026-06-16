from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.utils.io import read_table, require_columns, write_table

DEFAULT_EEG_QC_CONFIG = {
    "policy": "robust",
    "legacy_hf_threshold": 0.4,
    "bad_scene_fraction_threshold": 0.3,
    "min_segment_duration_s": 1.0,
    "nan_fraction_threshold": 0.2,
    "flat_fraction_threshold": 0.2,
    "robust_k": 3.5,
    "robust_min_n": 4,
    "robust_metrics": ["hf_ratio_20_40Hz", "rms_mean_uV", "peak_to_peak_uV"],
}


def run_eeg_pipeline(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    eeg_scene_csv: str | Path,
    outdir: str | Path = "outputs/04_eeg",
    eeg_qc_config: str | Path | dict | None = "configs/eeg_qc.json",
) -> dict[str, Path]:
    participants = read_table(participants_csv)
    scene = read_table(scene_manifest_csv)
    eeg = read_table(eeg_scene_csv)
    qc_config = load_eeg_qc_config(eeg_qc_config)
    eeg = normalize_eeg_ids(eeg, participants)
    require_columns(eeg, ["participant_id", "scene_id"], "EEG scene table")
    scene_cols = [c for c in ["participant_id", "scene_id", "WWR", "Complexity", "Cond", "block", "position", "round", "condition_id"] if c in scene.columns]
    out = eeg.merge(scene[scene_cols], on=["participant_id", "scene_id"], how="left", suffixes=("", "_scene"))
    out = add_eeg_derived_metrics(out)
    out, scene_qc, subject_qc, thresholds = apply_eeg_quality_qc(out, qc_config)
    qc = eeg_qc(out)
    outdir = Path(outdir)
    return {
        "eeg_trial_long": write_table(out, outdir / "eeg_trial_long.csv"),
        "eeg_qc_summary": write_table(qc, outdir / "eeg_qc_summary.csv"),
        "eeg_scene_qc": write_table(scene_qc, outdir / "eeg_scene_qc.csv"),
        "eeg_subject_qc": write_table(subject_qc, outdir / "eeg_subject_qc.csv"),
        "eeg_qc_thresholds": write_table(thresholds, outdir / "eeg_qc_thresholds.csv"),
    }


def load_eeg_qc_config(config: str | Path | dict | None) -> dict:
    out = dict(DEFAULT_EEG_QC_CONFIG)
    if config is None:
        return out
    if isinstance(config, dict):
        out.update(config)
        return out
    path = Path(config)
    if path.exists():
        out.update(json.loads(path.read_text(encoding="utf-8")))
    return out


def normalize_eeg_ids(eeg: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    out = eeg.copy()
    if "participant_id" not in out.columns and "subject_id" in out.columns:
        if "eeg_subject_id" in participants.columns:
            mapping = participants[["participant_id", "eeg_subject_id"]].rename(columns={"eeg_subject_id": "subject_id"})
            out = out.merge(mapping, on="subject_id", how="left")
        else:
            out["participant_id"] = out["subject_id"]
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["scene_id"] = pd.to_numeric(out["scene_id"], errors="coerce").astype("Int64")
    return out


def add_eeg_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "view_dur_s" not in out.columns:
        for col in ["duration_s", "dur_s", "view_duration_s", "eeg_view_dur_s"]:
            if col in out.columns:
                out["view_dur_s"] = pd.to_numeric(out[col], errors="coerce")
                break
    if {"O_alpha_gray", "O_alpha_view"}.issubset(out.columns):
        out["delta_O_alpha"] = pd.to_numeric(out["O_alpha_gray"], errors="coerce") - pd.to_numeric(out["O_alpha_view"], errors="coerce")
    elif {"gray_O_alpha", "view_O_alpha"}.issubset(out.columns):
        out["delta_O_alpha"] = pd.to_numeric(out["gray_O_alpha"], errors="coerce") - pd.to_numeric(out["view_O_alpha"], errors="coerce")
    elif {"O_alpha", "gray_O_alpha"}.issubset(out.columns):
        out["delta_O_alpha"] = pd.to_numeric(out["gray_O_alpha"], errors="coerce") - pd.to_numeric(out["O_alpha"], errors="coerce")
    return out


def apply_eeg_quality_qc(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = standardize_qc_columns(df)
    policy = str(config.get("policy", "robust")).strip().lower()
    if policy not in {"off", "audit_only", "legacy_0_4", "robust"}:
        raise ValueError(f"Unsupported EEG QC policy: {policy}")

    has_qc = any(col in out.columns for col in _qc_input_columns())
    out["eeg_qc_policy"] = policy if has_qc else "unavailable"
    out["eeg_quality_available"] = bool(has_qc)
    out["eeg_legacy_hf_flag"] = False
    out["eeg_qc_candidate_reasons"] = ""
    out["eeg_qc_reasons"] = ""
    out["bad_eeg_quality"] = False
    out["eeg_subject_quality_exclusion"] = False

    if not has_qc:
        return out, _scene_qc_table(out), _subject_qc_table(out, config), pd.DataFrame()

    legacy_threshold = float(config.get("legacy_hf_threshold", 0.4))
    hard_reasons = _hard_qc_reasons(out, config)
    robust_reasons, thresholds = _robust_qc_reasons(out, config)
    legacy_reasons = []
    hf = pd.to_numeric(out.get("hf_ratio_20_40Hz"), errors="coerce") if "hf_ratio_20_40Hz" in out.columns else pd.Series(np.nan, index=out.index)
    out["eeg_legacy_hf_flag"] = hf > legacy_threshold
    for is_bad in out["eeg_legacy_hf_flag"].fillna(False):
        legacy_reasons.append(["legacy_hf_ratio"] if bool(is_bad) else [])

    candidate_reasons: list[list[str]] = []
    formal_reasons: list[list[str]] = []
    for i in range(len(out)):
        candidate = hard_reasons[i] + robust_reasons[i] + legacy_reasons[i]
        candidate_reasons.append(candidate)
        if policy == "off":
            formal = []
        elif policy == "audit_only":
            formal = []
        elif policy == "legacy_0_4":
            formal = hard_reasons[i] + legacy_reasons[i]
        else:
            formal = hard_reasons[i] + robust_reasons[i]
        formal_reasons.append(formal)

    out["eeg_qc_candidate_reasons"] = [";".join(dict.fromkeys(r)) for r in candidate_reasons]
    out["eeg_qc_reasons"] = [";".join(dict.fromkeys(r)) for r in formal_reasons]
    out["bad_eeg_quality"] = [bool(r) for r in formal_reasons]
    out = _apply_subject_quality_exclusion(out, config, policy)
    return out, _scene_qc_table(out), _subject_qc_table(out, config), thresholds


def standardize_qc_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_candidates = {
        "view_hf_ratio": "hf_ratio_20_40Hz",
        "hf_ratio": "hf_ratio_20_40Hz",
        "view_rms_mean": "rms_mean_uV",
        "rms_mean": "rms_mean_uV",
        "rms_uV": "rms_mean_uV",
        "ptp_uV": "peak_to_peak_uV",
        "peak_to_peak": "peak_to_peak_uV",
        "valid_duration": "segment_valid_duration",
    }
    for source, target in rename_candidates.items():
        if source in out.columns and target not in out.columns:
            out[target] = out[source]
    return out


def _hard_qc_reasons(df: pd.DataFrame, config: dict) -> list[list[str]]:
    reasons = [[] for _ in range(len(df))]
    if "view_dur_s" in df.columns:
        dur = pd.to_numeric(df["view_dur_s"], errors="coerce")
        min_dur = float(config.get("min_segment_duration_s", 1.0))
        for i, is_bad in enumerate(dur.isna() | (dur < min_dur)):
            if bool(is_bad):
                reasons[i].append("segment_duration")
    if "segment_valid_duration" in df.columns:
        valid = df["segment_valid_duration"].map(_truthy_or_missing)
        for i, is_bad in enumerate(valid == False):  # noqa: E712
            if bool(is_bad):
                reasons[i].append("segment_valid_duration")
    for col, threshold_key, reason in [
        ("nan_fraction", "nan_fraction_threshold", "nan_fraction"),
        ("flat_fraction", "flat_fraction_threshold", "flat_fraction"),
    ]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        threshold = float(config.get(threshold_key, 0.2))
        for i, is_bad in enumerate(values > threshold):
            if bool(is_bad):
                reasons[i].append(reason)
    return reasons


def _robust_qc_reasons(df: pd.DataFrame, config: dict) -> tuple[list[list[str]], pd.DataFrame]:
    reasons = [[] for _ in range(len(df))]
    thresholds = []
    robust_metrics = list(config.get("robust_metrics", DEFAULT_EEG_QC_CONFIG["robust_metrics"]))
    robust_k = float(config.get("robust_k", 3.5))
    min_n = int(config.get("robust_min_n", 4))
    for col in robust_metrics:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.dropna()
        row = {"metric": col, "method": "median_plus_k_mad", "k": robust_k, "n": int(valid.size), "threshold": np.nan, "median": np.nan, "mad": np.nan}
        if valid.size < min_n:
            row["method"] = "not_applied_insufficient_n"
            thresholds.append(row)
            continue
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        threshold = median + robust_k * mad
        row.update({"threshold": threshold, "median": median, "mad": mad})
        thresholds.append(row)
        for i, is_bad in enumerate(values > threshold):
            if bool(is_bad):
                reasons[i].append(f"robust_{col}")
    return reasons, pd.DataFrame(thresholds)


def _apply_subject_quality_exclusion(df: pd.DataFrame, config: dict, policy: str) -> pd.DataFrame:
    out = df.copy()
    if policy in {"off", "audit_only"} or "participant_id" not in out.columns:
        return out
    threshold = float(config.get("bad_scene_fraction_threshold", 0.3))
    for participant_id, idx in out.groupby("participant_id").groups.items():
        sub = out.loc[idx]
        if sub.empty:
            continue
        bad_fraction = float(sub["bad_eeg_quality"].fillna(False).mean())
        if bad_fraction > threshold:
            out.loc[idx, "bad_eeg_quality"] = True
            out.loc[idx, "eeg_subject_quality_exclusion"] = True
            for row_idx in idx:
                existing = str(out.at[row_idx, "eeg_qc_reasons"] or "")
                parts = [p for p in existing.split(";") if p]
                parts.append("eeg_subject_quality_exclusion")
                out.at[row_idx, "eeg_qc_reasons"] = ";".join(dict.fromkeys(parts))
    return out


def _scene_qc_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "participant_id", "scene_id", "eeg_qc_policy", "eeg_quality_available", "bad_eeg_quality",
        "eeg_subject_quality_exclusion", "eeg_qc_reasons", "eeg_qc_candidate_reasons",
        "eeg_legacy_hf_flag", "hf_ratio_20_40Hz", "rms_mean_uV", "peak_to_peak_uV",
        "nan_fraction", "flat_fraction", "near_boundary", "segment_valid_duration",
    ]
    present = [c for c in cols if c in df.columns]
    return df[present].copy() if present else pd.DataFrame()


def _subject_qc_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if "participant_id" not in df.columns:
        return pd.DataFrame()
    rows = []
    for participant_id, sub in df.groupby("participant_id", dropna=False):
        bad = sub.get("bad_eeg_quality", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        subject_ex = sub.get("eeg_subject_quality_exclusion", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        rows.append({
            "participant_id": participant_id,
            "eeg_qc_policy": sub.get("eeg_qc_policy", pd.Series(["unavailable"])).iloc[0],
            "n_eeg_scenes": int(len(sub)),
            "n_bad_eeg_scenes": int(bad.sum()),
            "bad_eeg_scene_fraction": float(bad.mean()) if len(sub) else np.nan,
            "eeg_subject_quality_exclusion": bool(subject_ex.any()),
            "bad_scene_fraction_threshold": float(config.get("bad_scene_fraction_threshold", 0.3)),
        })
    return pd.DataFrame(rows)


def _qc_input_columns() -> set[str]:
    return {
        "hf_ratio_20_40Hz", "view_hf_ratio", "hf_ratio", "rms_mean_uV", "view_rms_mean",
        "rms_mean", "rms_uV", "peak_to_peak_uV", "ptp_uV", "peak_to_peak", "nan_fraction",
        "flat_fraction", "near_boundary", "segment_valid_duration", "valid_duration",
    }


def _truthy_or_missing(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "valid"}:
        return True
    if text in {"false", "0", "no", "n", "invalid"}:
        return False
    return None


def eeg_qc(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if any(token in c.lower() for token in ["theta", "alpha", "beta"])]
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append({"metric": col, "n": int(values.notna().sum()), "missing": int(values.isna().sum()), "mean": float(values.mean()) if values.notna().any() else None})
    if "bad_eeg_quality" in df.columns:
        bad = df["bad_eeg_quality"].fillna(False).astype(bool)
        rows.append({"metric": "bad_eeg_quality", "n": int(bad.notna().sum()), "missing": int(df["bad_eeg_quality"].isna().sum()), "mean": float(bad.mean()) if len(bad) else None})
    return pd.DataFrame(rows)
