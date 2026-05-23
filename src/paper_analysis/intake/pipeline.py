from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_analysis.utils.coding import active_rows, condition_id, standardize_participants, wwr_numeric
from paper_analysis.utils.io import assert_unique, read_table, require_columns, resolve_path, write_table


def build_manifests(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path = "outputs/01_sample_qc",
) -> dict[str, Path]:
    participants = standardize_participants(read_table(participants_csv))
    scene = read_table(scene_manifest_csv)
    scene_base = Path(scene_manifest_csv).parent
    require_columns(participants, ["participant_id"], "participants")
    require_columns(scene, ["participant_id", "scene_id", "WWR", "Complexity"], "scene_manifest")

    participants["participant_id"] = participants["participant_id"].astype(str).str.strip()
    scene["participant_id"] = scene["participant_id"].astype(str).str.strip()
    scene["scene_id"] = pd.to_numeric(scene["scene_id"], errors="coerce").astype("Int64")
    scene["WWR_numeric"] = scene["WWR"].map(wwr_numeric)
    for path_col in ["eye_csv_path", "aoi_json_path"]:
        if path_col in scene.columns:
            scene[path_col] = scene[path_col].map(lambda value: str(resolve_path(value, scene_base).resolve()) if resolve_path(value, scene_base) else "")
    if "condition_id" not in scene.columns:
        scene["condition_id"] = scene.apply(condition_id, axis=1)
    for col in ["block", "position", "round"]:
        if col not in scene.columns:
            scene[col] = pd.NA
    assert_unique(scene, ["participant_id", "scene_id"], "scene_manifest")

    active = active_rows(participants)
    flow = participant_flow(participants)
    balance = group_balance(active)
    scene_balance = scene_design_balance(scene.loc[scene["participant_id"].isin(active["participant_id"])])

    outdir = Path(outdir)
    return {
        "participants_standardized": write_table(participants, outdir / "participants_standardized.csv"),
        "scene_manifest_standardized": write_table(scene, outdir / "scene_manifest_standardized.csv"),
        "participant_flow": write_table(flow, outdir / "participant_flow.csv"),
        "group_balance": write_table(balance, outdir / "group_balance_before_after.csv"),
        "scene_design_balance": write_table(scene_balance, outdir / "scene_design_balance.csv"),
    }


def participant_flow(participants: pd.DataFrame) -> pd.DataFrame:
    total = len(participants)
    excluded = int(participants.get("exclude", pd.Series(False, index=participants.index)).astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    rows = [
        {"stage": "recruited_or_imported", "n": total},
        {"stage": "excluded", "n": excluded},
        {"stage": "active_for_analysis", "n": total - excluded},
    ]
    if "RecruitmentBatch" in participants.columns:
        for batch, sub in participants.groupby("RecruitmentBatch", dropna=False):
            rows.append({"stage": f"batch:{batch}", "n": len(sub)})
    return pd.DataFrame(rows)


def group_balance(participants: pd.DataFrame) -> pd.DataFrame:
    factors = [c for c in ["ExperienceGroup", "Gender", "RecruitmentBatch", "SupplementFlag"] if c in participants.columns]
    rows: list[dict] = []
    for factor in factors:
        counts = participants[factor].fillna("Unknown").astype(str).value_counts(dropna=False)
        for level, n in counts.items():
            rows.append({"factor": factor, "level": level, "n": int(n), "percent": float(n / max(len(participants), 1) * 100)})
    return pd.DataFrame(rows)


def scene_design_balance(scene: pd.DataFrame) -> pd.DataFrame:
    factors = [c for c in ["WWR", "Complexity", "block", "position", "round", "condition_id"] if c in scene.columns]
    rows: list[dict] = []
    for factor in factors:
        for level, n in scene[factor].fillna("NA").astype(str).value_counts(dropna=False).items():
            rows.append({"factor": factor, "level": level, "trial_count": int(n)})
    return pd.DataFrame(rows)
