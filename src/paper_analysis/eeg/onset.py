"""Configuration and table contracts for scene-onset EEG trimming.

Waveform trimming is deliberately performed by the MATLAB exporter.  This
module validates the exported metadata and builds cross-window QC contracts;
it never pretends that a scene-level table can be trimmed after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ONSET_METADATA_COLUMNS = (
    "onset_trim_s",
    "analysis_start_s",
    "analysis_end_s",
    "analysis_dur_s",
    "onset_samples_removed",
    "trim_status",
)
DEFAULT_EEG_ANALYSIS_CONFIG = {
    "primary_onset_trim_s": 10.0,
    "onset_trim_variants_s": [0.0, 5.0, 10.0, 15.0],
    "equivalence_bound_sd": 0.20,
    "bootstrap_iterations": 5000,
    "random_seed": 20260802,
}


def load_eeg_analysis_config(config: str | Path | dict | None) -> dict:
    """Load and strictly validate the single source of onset-trim settings."""
    out = dict(DEFAULT_EEG_ANALYSIS_CONFIG)
    if isinstance(config, dict):
        out.update(config)
    elif config is not None:
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"EEG analysis config not found: {path}")
        out.update(json.loads(path.read_text(encoding="utf-8")))

    primary = float(out["primary_onset_trim_s"])
    variants = sorted({float(value) for value in out["onset_trim_variants_s"]})
    if primary < 0 or any(value < 0 for value in variants):
        raise ValueError("EEG onset trims must be non-negative")
    if primary not in variants:
        raise ValueError("primary_onset_trim_s must appear in onset_trim_variants_s")
    bound = float(out["equivalence_bound_sd"])
    if not 0 < bound < 1:
        raise ValueError("equivalence_bound_sd must lie between 0 and 1")
    iterations = int(out["bootstrap_iterations"])
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    out.update({
        "primary_onset_trim_s": primary,
        "onset_trim_variants_s": variants,
        "equivalence_bound_sd": bound,
        "bootstrap_iterations": iterations,
        "random_seed": int(out["random_seed"]),
    })
    return out


def validate_onset_metadata(
    frame: pd.DataFrame,
    *,
    expected_trim_s: float | None = None,
    sensitivity: bool = False,
    require: bool = False,
) -> list[str]:
    """Return contract errors for primary or long-form onset-trim exports."""
    errors: list[str] = []
    missing = [column for column in ONSET_METADATA_COLUMNS if column not in frame]
    if missing:
        if require:
            errors.append("missing onset-trim metadata: " + ",".join(missing))
        return errors
    key = ["participant_id" if "participant_id" in frame else "subject_id", "scene_id"]
    if sensitivity:
        key.append("onset_trim_s")
    if all(column in frame for column in key):
        duplicate_count = int(frame.duplicated(key).sum())
        if duplicate_count:
            errors.append(f"duplicate {'+'.join(key)} rows: {duplicate_count}")
    trim = pd.to_numeric(frame["onset_trim_s"], errors="coerce")
    if trim.isna().any() or trim.lt(0).any():
        errors.append("onset_trim_s must be finite and non-negative")
    if expected_trim_s is not None and not np.isclose(trim, float(expected_trim_s), equal_nan=False).all():
        observed = sorted(trim.dropna().unique().tolist())
        errors.append(f"expected onset_trim_s={float(expected_trim_s):g}; observed {observed}")
    missing_view = [column for column in ("view_start_s", "view_end_s", "view_dur_s") if column not in frame]
    if missing_view:
        errors.append("missing full-view timing metadata: " + ",".join(missing_view))
        return errors
    start = pd.to_numeric(frame["analysis_start_s"], errors="coerce")
    end = pd.to_numeric(frame["analysis_end_s"], errors="coerce")
    duration = pd.to_numeric(frame["analysis_dur_s"], errors="coerce")
    view_start = pd.to_numeric(frame["view_start_s"], errors="coerce")
    view_end = pd.to_numeric(frame["view_end_s"], errors="coerce")
    valid = frame.get("trim_status", pd.Series("", index=frame.index)).astype(str).eq("ok")
    if valid.any():
        if not np.isclose(start[valid], view_start[valid] + trim[valid], atol=1e-6, equal_nan=False).all():
            errors.append("analysis_start_s must equal view_start_s + onset_trim_s for valid rows")
        if not np.isclose(end[valid], view_end[valid], atol=1e-6, equal_nan=False).all():
            errors.append("analysis_end_s must preserve view_end_s for valid rows")
        if not np.isclose(duration[valid], end[valid] - start[valid], atol=1e-6, equal_nan=False).all():
            errors.append("analysis_dur_s must equal analysis_end_s - analysis_start_s for valid rows")
    return errors


def assert_primary_matches_sensitivity(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    primary_trim_s: float,
) -> None:
    """Require the formal table to be an exact projection of the long table."""
    subset = sensitivity.loc[
        np.isclose(pd.to_numeric(sensitivity["onset_trim_s"], errors="coerce"), primary_trim_s)
    ].copy()
    keys = ["participant_id", "scene_id"]
    common = [column for column in primary.columns if column in subset.columns]
    left = primary.sort_values(keys).reset_index(drop=True)[common]
    right = subset.sort_values(keys).reset_index(drop=True)[common]
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError(
            f"Primary EEG table is not the exact onset_trim_s={primary_trim_s:g} projection"
        ) from exc


def build_common_qc_table(
    sensitivity_trials: pd.DataFrame,
    variants: Iterable[float] = (5.0, 10.0, 15.0),
) -> pd.DataFrame:
    """Build the predeclared 5/10/15-s common scene-and-subject QC set."""
    required = {"participant_id", "scene_id", "onset_trim_s", "bad_eeg_quality"}
    missing = required - set(sensitivity_trials.columns)
    if missing:
        raise ValueError(f"Onset sensitivity trial table missing {sorted(missing)}")
    variants = tuple(float(value) for value in variants)
    work = sensitivity_trials.copy()
    work["onset_trim_s"] = pd.to_numeric(work["onset_trim_s"], errors="coerce")
    work["_pass"] = ~_truthy(work["bad_eeg_quality"])
    if "eeg_subject_quality_exclusion" in work:
        work["_pass"] &= ~_truthy(work["eeg_subject_quality_exclusion"])
    keys = ["participant_id", "scene_id"]
    base = work[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    for trim in variants:
        label = _trim_label(trim)
        selected = work.loc[np.isclose(work["onset_trim_s"], trim), keys + ["_pass"]]
        selected = selected.rename(columns={"_pass": f"qc_pass_trim_{label}"})
        base = base.merge(selected, on=keys, how="left", validate="one_to_one")
        base[f"qc_pass_trim_{label}"] = base[f"qc_pass_trim_{label}"].fillna(False).astype(bool)
    pass_columns = [f"qc_pass_trim_{_trim_label(trim)}" for trim in variants]
    base["onset_common_qc_pass"] = base[pass_columns].all(axis=1)
    base["common_qc_variants_s"] = ",".join(f"{value:g}" for value in variants)
    return base


def onset_qc_sample_flow(sensitivity_trials: pd.DataFrame, common_qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for trim, sub in sensitivity_trials.groupby("onset_trim_s", dropna=False):
        passed = ~_truthy(sub.get("bad_eeg_quality", pd.Series(False, index=sub.index)))
        if "eeg_subject_quality_exclusion" in sub:
            passed &= ~_truthy(sub["eeg_subject_quality_exclusion"])
        rows.append({
            "sample_strategy": "variant_specific_qc",
            "onset_trim_s": trim,
            "trials_total": int(sub[["participant_id", "scene_id"]].drop_duplicates().shape[0]),
            "trials_retained": int(sub.loc[passed, ["participant_id", "scene_id"]].drop_duplicates().shape[0]),
            "participants_retained": int(sub.loc[passed, "participant_id"].nunique()),
        })
    common_pass = common_qc["onset_common_qc_pass"].fillna(False).astype(bool)
    rows.append({
        "sample_strategy": "common_5_10_15_qc",
        "onset_trim_s": np.nan,
        "trials_total": int(len(common_qc)),
        "trials_retained": int(common_pass.sum()),
        "participants_retained": int(common_qc.loc[common_pass, "participant_id"].nunique()),
    })
    return pd.DataFrame(rows)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _trim_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")
