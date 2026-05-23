from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_analysis.utils.io import read_table, require_columns, write_table


def run_eeg_pipeline(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    eeg_scene_csv: str | Path,
    outdir: str | Path = "outputs/04_eeg",
) -> dict[str, Path]:
    participants = read_table(participants_csv)
    scene = read_table(scene_manifest_csv)
    eeg = read_table(eeg_scene_csv)
    eeg = normalize_eeg_ids(eeg, participants)
    require_columns(eeg, ["participant_id", "scene_id"], "EEG scene table")
    scene_cols = [c for c in ["participant_id", "scene_id", "WWR", "Complexity", "Cond", "block", "position", "round", "condition_id"] if c in scene.columns]
    out = eeg.merge(scene[scene_cols], on=["participant_id", "scene_id"], how="left", suffixes=("", "_scene"))
    out = add_eeg_derived_metrics(out)
    qc = eeg_qc(out)
    outdir = Path(outdir)
    return {
        "eeg_trial_long": write_table(out, outdir / "eeg_trial_long.csv"),
        "eeg_qc_summary": write_table(qc, outdir / "eeg_qc_summary.csv"),
    }


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
    if {"O_alpha_gray", "O_alpha_view"}.issubset(out.columns):
        out["delta_O_alpha"] = pd.to_numeric(out["O_alpha_gray"], errors="coerce") - pd.to_numeric(out["O_alpha_view"], errors="coerce")
    elif {"gray_O_alpha", "view_O_alpha"}.issubset(out.columns):
        out["delta_O_alpha"] = pd.to_numeric(out["gray_O_alpha"], errors="coerce") - pd.to_numeric(out["view_O_alpha"], errors="coerce")
    return out


def eeg_qc(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if any(token in c.lower() for token in ["theta", "alpha", "beta"])]
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append({"metric": col, "n": int(values.notna().sum()), "missing": int(values.isna().sum()), "mean": float(values.mean()) if values.notna().any() else None})
    return pd.DataFrame(rows)
