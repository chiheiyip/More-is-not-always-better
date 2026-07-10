from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import pandas as pd

from paper_analysis.utils.coding import active_rows, condition_id, standardize_participants, wwr_numeric
from paper_analysis.utils.io import assert_unique, read_table, require_columns, resolve_path, write_table


def build_manifests(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path = "outputs/01_sample_qc",
) -> dict[str, Path]:
    # Cohort provenance lives on the raw eye-record rows, not necessarily on the
    # questionnaire-derived participant table.  Attach it before standardisation
    # so a later-added cohort cannot silently inherit the default "Original" label.
    participants = read_table(participants_csv)
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

    participants = _attach_collection_cohort(participants, scene)
    participants = standardize_participants(participants)

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


SECOND_BATCH_START = date(2026, 5, 1)


def _attach_collection_cohort(participants: pd.DataFrame, scene: pd.DataFrame) -> pd.DataFrame:
    """Fill participant-level collection metadata from canonical scene records.

    The source data records acquisition time in ``eye_record_id`` (YYMMDD...),
    whereas questionnaire metadata has no such field.  One participant must map
    to one acquisition cohort; conflicting dates are rejected rather than hidden.
    """
    out = participants.copy()
    if "eye_record_id" not in scene.columns:
        return out
    records = scene[["participant_id", "eye_record_id"]].copy()
    records["participant_id"] = records["participant_id"].astype(str).str.strip()
    records["collection_date"] = records["eye_record_id"].map(_date_from_record_id)
    unique_dates = records.groupby("participant_id", dropna=False)["collection_date"].agg(
        lambda values: sorted({value for value in values if value is not None})
    )
    conflicts = {participant_id: values for participant_id, values in unique_dates.items() if len(values) > 1}
    if conflicts:
        raise ValueError(f"Participants map to multiple collection dates: {dict(list(conflicts.items())[:5])}")
    cohort = records.dropna(subset=["collection_date"]).drop_duplicates("participant_id")
    if cohort.empty:
        return out
    cohort["collection_date"] = cohort["collection_date"].map(date.isoformat)
    cohort["DateBatch"] = cohort["collection_date"].map(
        lambda value: "Supplement" if date.fromisoformat(value) >= SECOND_BATCH_START else "Initial"
    )
    cohort["RecruitmentBatch"] = cohort["DateBatch"].map({"Initial": "Original", "Supplement": "Supplement"})
    cohort["SupplementFlag"] = cohort["DateBatch"].eq("Supplement")
    keep = ["participant_id", "eye_record_id", "collection_date", "DateBatch", "RecruitmentBatch", "SupplementFlag"]
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out = out.merge(cohort[keep], on="participant_id", how="left", suffixes=("", "_derived"))
    for col in ["eye_record_id", "collection_date", "DateBatch", "RecruitmentBatch", "SupplementFlag"]:
        derived = f"{col}_derived"
        if derived not in out.columns:
            continue
        missing = out.get(col, pd.Series(pd.NA, index=out.index)).astype(str).str.strip().str.lower().isin({"", "nan", "none", "nat"})
        if col == "SupplementFlag":
            missing = out.get(col, pd.Series(pd.NA, index=out.index)).isna()
        if col not in out.columns:
            out[col] = out[derived]
        else:
            out.loc[missing, col] = out.loc[missing, derived]
        out = out.drop(columns=[derived])
    return out


def _date_from_record_id(value: object) -> date | None:
    match = re.match(r"(\d{6})", str(value or "").strip())
    if not match:
        return None
    try:
        return date(2000 + int(match.group(1)[:2]), int(match.group(1)[2:4]), int(match.group(1)[4:6]))
    except ValueError:
        return None


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
