from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from paper_analysis.utils.io import assert_unique, read_table, require_columns, write_table, write_text
from paper_analysis.utils.markdown import dataframe_to_markdown


S_ITEMS = ["S1", "S2", "S3", "S4", "S5"]
B_ITEMS = ["B1", "B2", "B3"]
QUESTION_ITEMS = S_ITEMS + B_ITEMS + ["IPQ"]


def run_questionnaire_pipeline(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path = "outputs/02_questionnaire",
    questionnaire_wide: str | Path | None = None,
    questionnaire_long: str | Path | None = None,
) -> dict[str, Path]:
    if questionnaire_wide is None and questionnaire_long is None:
        raise ValueError("Provide questionnaire_wide or questionnaire_long")
    participants = read_table(participants_csv)
    scene = read_table(scene_manifest_csv)
    if questionnaire_long is not None:
        long = read_table(questionnaire_long)
    else:
        long = wide_to_long(read_table(questionnaire_wide))

    long = standardize_questionnaire_long(long)
    merged = attach_scene_and_participants(long, scene, participants)
    qc = questionnaire_qc(merged)
    s_desc = item_descriptives(merged, S_ITEMS)
    b_desc = item_descriptives(merged, B_ITEMS)
    paper_md = questionnaire_markdown(s_desc, b_desc)

    outdir = Path(outdir)
    return {
        "questionnaire_long": write_table(merged, outdir / "questionnaire_long.csv"),
        "questionnaire_qc": write_table(qc, outdir / "questionnaire_qc.csv"),
        "s_items_descriptives": write_table(s_desc, outdir / "s_items_descriptives.csv"),
        "b_items_descriptives": write_table(b_desc, outdir / "b_items_descriptives.csv"),
        "questionnaire_paper_tables": write_text(paper_md, outdir / "questionnaire_paper_tables.md"),
    }


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    require_columns(wide, ["participant_id"], "questionnaire wide table")
    id_cols = [c for c in wide.columns if not _parse_item_scene(c)]
    rows: list[dict] = []
    for _, row in wide.iterrows():
        base = {c: row[c] for c in id_cols}
        by_scene: dict[int, dict] = {}
        for col in wide.columns:
            parsed = _parse_item_scene(col)
            if not parsed:
                continue
            item, scene_id = parsed
            by_scene.setdefault(scene_id, {}).update({item: row[col]})
        for scene_id, values in by_scene.items():
            rows.append({**base, "scene_id": scene_id, **values})
    return pd.DataFrame(rows)


def standardize_questionnaire_long(long: pd.DataFrame) -> pd.DataFrame:
    require_columns(long, ["participant_id", "scene_id"], "questionnaire long table")
    out = long.copy()
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["scene_id"] = pd.to_numeric(out["scene_id"], errors="coerce").astype("Int64")
    for item in QUESTION_ITEMS:
        if item in out.columns:
            out[item] = pd.to_numeric(out[item], errors="coerce")
    assert_unique(out, ["participant_id", "scene_id"], "questionnaire long table")
    return out


def attach_scene_and_participants(long: pd.DataFrame, scene: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    scene_cols = [c for c in ["participant_id", "scene_id", "WWR", "Complexity", "Cond", "block", "position", "round", "condition_id"] if c in scene.columns]
    part_cols = [c for c in ["participant_id", "ExperienceGroup", "ExperienceRaw", "Experience", "Gender", "Age", "RecruitmentBatch", "SupplementFlag"] if c in participants.columns]
    out = long.merge(scene[scene_cols], on=["participant_id", "scene_id"], how="left")
    if part_cols:
        out = out.merge(participants[part_cols].drop_duplicates("participant_id"), on="participant_id", how="left")
    return out


def questionnaire_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in [c for c in QUESTION_ITEMS if c in df.columns]:
        values = pd.to_numeric(df[item], errors="coerce")
        rows.append({
            "item": item,
            "n_observations": int(values.notna().sum()),
            "n_missing": int(values.isna().sum()),
            "mean": float(values.mean()) if values.notna().any() else None,
            "sd": float(values.std()) if values.notna().sum() > 1 else None,
            "min": float(values.min()) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
        })
    return pd.DataFrame(rows)


def item_descriptives(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    group_cols = [c for c in ["WWR", "Complexity", "ExperienceGroup"] if c in df.columns]
    rows: list[dict] = []
    for item in [i for i in items if i in df.columns]:
        if group_cols:
            grouped = df.groupby(group_cols, dropna=False)[item]
            for keys, values in grouped:
                keys = keys if isinstance(keys, tuple) else (keys,)
                row = {"item": item, **dict(zip(group_cols, keys))}
                row.update(_summary(values))
                rows.append(row)
        else:
            rows.append({"item": item, **_summary(df[item])})
    return pd.DataFrame(rows)


def questionnaire_markdown(s_desc: pd.DataFrame, b_desc: pd.DataFrame) -> str:
    return "\n".join([
        "# Questionnaire Paper Tables",
        "",
        "## S Items",
        dataframe_to_markdown(s_desc) if not s_desc.empty else "No S item data.",
        "",
        "## B Items",
        dataframe_to_markdown(b_desc) if not b_desc.empty else "No B item data.",
        "",
    ])


def _summary(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "n": int(numeric.notna().sum()),
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
        "sd": float(numeric.std()) if numeric.notna().sum() > 1 else None,
    }


def _parse_item_scene(name: str) -> tuple[str, int] | None:
    text = str(name).strip()
    match = re.match(r"^(S[1-5]|B[1-3]|IPQ)[_\-\. ]?(?:scene)?(\d{1,2})$", text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), int(match.group(2))
    match = re.match(r"^(?:scene)?(\d{1,2})[_\-\. ]?(S[1-5]|B[1-3]|IPQ)$", text, re.IGNORECASE)
    if match:
        return match.group(2).upper(), int(match.group(1))
    return None
