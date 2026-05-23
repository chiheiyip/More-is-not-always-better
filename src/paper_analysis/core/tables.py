from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.core.schema import CORE_COLUMNS, DESIGN_COLUMNS, EEG_TIMING_COLUMNS, KEYS, PARTICIPANT_COLUMNS, SCENE_PATH_COLUMNS
from paper_analysis.utils.coding import active_rows, condition_id, standardize_participants, wwr_numeric
from paper_analysis.utils.io import assert_unique, read_table, require_columns, resolve_path


def load_participants_table(path: str | Path) -> pd.DataFrame:
    participants = standardize_participants(read_table(path))
    require_columns(participants, ["participant_id"], "participants")
    participants["participant_id"] = participants["participant_id"].astype(str).str.strip()
    return participants


def load_scene_table(path: str | Path) -> pd.DataFrame:
    scene = read_table(path)
    require_columns(scene, ["participant_id", "scene_id"], "scene_manifest")
    base = Path(path).parent
    scene = scene.copy()
    scene["participant_id"] = scene["participant_id"].astype(str).str.strip()
    scene["scene_id"] = pd.to_numeric(scene["scene_id"], errors="coerce").astype("Int64")
    if "WWR" in scene.columns and "WWR_numeric" not in scene.columns:
        scene["WWR_numeric"] = scene["WWR"].map(wwr_numeric)
    if "condition_id" not in scene.columns and {"WWR", "Complexity"}.issubset(scene.columns):
        scene["condition_id"] = scene.apply(condition_id, axis=1)
    for col in ["block", "position", "round"]:
        if col not in scene.columns:
            scene[col] = pd.NA
    for path_col in SCENE_PATH_COLUMNS:
        if path_col in scene.columns:
            scene[path_col] = scene[path_col].map(lambda value: str(resolve_path(value, base).resolve()) if resolve_path(value, base) else "")
    assert_unique(scene, KEYS, "scene_manifest")
    return scene


def build_trial_index(participants: pd.DataFrame, scene: pd.DataFrame, active_only: bool = True) -> pd.DataFrame:
    participants = standardize_participants(participants)
    if active_only:
        participants = active_rows(participants)
    active_ids = set(participants["participant_id"].astype(str))
    trial = scene.loc[scene["participant_id"].astype(str).isin(active_ids)].copy()
    trial = add_participant_fields(trial, participants)
    trial = trial.sort_values(KEYS).reset_index(drop=True)
    assert_unique(trial, KEYS, "canonical trial index")
    return trial


def add_participant_fields(df: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    part_cols = [c for c in PARTICIPANT_COLUMNS if c in participants.columns]
    if not part_cols:
        return df.copy()
    return df.merge(participants[part_cols].drop_duplicates("participant_id"), on="participant_id", how="left", suffixes=("", "_participant"))


def standardize_eeg_scene(eeg: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    out = eeg.copy()
    if "participant_id" not in out.columns:
        if "subject_id" not in out.columns:
            raise ValueError("EEG scene table must contain participant_id or subject_id")
        if "eeg_subject_id" in participants.columns:
            mapping = participants[["participant_id", "eeg_subject_id"]].rename(columns={"eeg_subject_id": "subject_id"})
            out = out.merge(mapping, on="subject_id", how="left")
        else:
            out["participant_id"] = out["subject_id"]
    missing_participant = out["participant_id"].isna() | out["participant_id"].astype(str).str.strip().str.lower().isin({"", "nan", "none"})
    if missing_participant.any():
        missing = out.loc[missing_participant].head(10).to_dict("records")
        raise ValueError(f"EEG subject_id values not found in participants. Sample: {missing}")
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["scene_id"] = pd.to_numeric(out["scene_id"], errors="coerce").astype("Int64")
    require_columns(out, KEYS, "EEG scene table")
    assert_unique(out, KEYS, "EEG scene table")
    return out


def eeg_duration_table(eeg: pd.DataFrame) -> pd.DataFrame:
    out = eeg[KEYS].copy()
    out["eeg_view_dur_s"] = infer_eeg_duration_s(eeg)
    return out


def infer_eeg_duration_s(eeg: pd.DataFrame) -> pd.Series:
    for col in ["eeg_view_dur_s", "view_dur_s", "view_duration_s", "dur_s", "duration_s"]:
        if col in eeg.columns:
            return pd.to_numeric(eeg[col], errors="coerce")
    for start_col, end_col in [("view_start_s", "view_end_s"), ("start_s", "end_s")]:
        if {start_col, end_col}.issubset(eeg.columns):
            start = pd.to_numeric(eeg[start_col], errors="coerce")
            end = pd.to_numeric(eeg[end_col], errors="coerce")
            return end - start
    return pd.Series(np.nan, index=eeg.index)


def prefix_non_core_columns(df: pd.DataFrame, prefix: str, keep: list[str] | None = None) -> pd.DataFrame:
    keep_set = set(keep or []) | set(KEYS) | set(DESIGN_COLUMNS)
    rename = {
        col: f"{prefix}_{col}"
        for col in df.columns
        if col not in keep_set and not col.startswith(f"{prefix}_")
    }
    return df.rename(columns=rename)


def coalesce_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DESIGN_COLUMNS:
        candidates = [
            c for c in out.columns
            if c == col or c.endswith(f"_{col}") or c.startswith(f"q_{col}") or c.startswith(f"eeg_{col}") or c.startswith(f"eye_{col}")
        ]
        if not candidates:
            continue
        if col not in out.columns:
            out[col] = out[candidates[0]]
        for candidate in candidates:
            if candidate != col:
                out[col] = out[col].combine_first(out[candidate])
    return out


def canonical_columns_first(df: pd.DataFrame) -> pd.DataFrame:
    first = list(dict.fromkeys(c for c in CORE_COLUMNS if c in df.columns))
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]
