from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


EXPERIENCE_GROUP_RULE = "q1_4_table_tennis_experience_2_by_2"
EXPERIENCE_SOURCE_PRIORITY = ["Experience", "ExperienceRaw", "ExperienceGroup", "SportFreq"]

LOW_EXPERIENCE_PATTERNS = [
    "never",
    "rare",
    "rarely",
    "low",
    "none",
    "从不",
    "极少",
    "较少",
    "少于",
    "每月<1",
    "每月＜1",
    "每月少于",
    "偶尔",
    "1-2次",
    "1–2次",
    "每月1-2",
    "每月1–2",
]

HIGH_EXPERIENCE_PATTERNS = [
    "sometimes",
    "often",
    "frequent",
    "weekly",
    "high",
    "有时",
    "经常",
    "3-4次",
    "3–4次",
    "每月3-4",
    "每月3–4",
    "≥5次",
    ">=5次",
    "每周",
]


def standardize_gender(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return "Unknown"
    low = text.lower()
    if low in {"m", "male", "man"} or text in {"男", "男性", "男生"}:
        return "Male"
    if low in {"f", "female", "woman"} or text in {"女", "女性", "女生"}:
        return "Female"
    return text


def standardize_participants(participants: pd.DataFrame) -> pd.DataFrame:
    out = participants.copy()
    if "exclude" not in out.columns:
        out["exclude"] = False
    source_col, raw = _experience_source(out)
    if "ExperienceGroup" in out.columns and "ExperienceGroupInput" not in out.columns:
        out["ExperienceGroupInput"] = out["ExperienceGroup"]
    out["ExperienceRaw"] = raw
    out["ExperienceGroup"] = out["ExperienceRaw"].map(experience_group)
    out["ExperienceGroupSource"] = source_col
    out["ExperienceGroupRule"] = EXPERIENCE_GROUP_RULE
    if "RecruitmentBatch" not in out.columns:
        out["RecruitmentBatch"] = "Original"
    if "SupplementFlag" not in out.columns:
        out["SupplementFlag"] = out["RecruitmentBatch"].astype(str).str.lower().eq("supplement")
    if "Gender" not in out.columns:
        out["Gender"] = ""
    out["Gender"] = out["Gender"].astype("object")
    if "GenderRaw" not in out.columns:
        out["GenderRaw"] = out["Gender"]
    else:
        out["GenderRaw"] = out["GenderRaw"].astype("object")
        missing_raw = out["GenderRaw"].astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        out.loc[missing_raw, "GenderRaw"] = out.loc[missing_raw, "Gender"]
    out["Gender"] = out["GenderRaw"].map(standardize_gender)
    if "Age" not in out.columns:
        out["Age"] = np.nan
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out


def _experience_source(participants: pd.DataFrame) -> tuple[str, pd.Series]:
    for col in EXPERIENCE_SOURCE_PRIORITY:
        if col not in participants.columns:
            continue
        series = participants[col].fillna("").astype(str)
        nonempty = ~series.str.strip().str.lower().isin({"", "nan", "none", "unknown"})
        if nonempty.any():
            return col, series
    return "", pd.Series("", index=participants.index, dtype="object")


def experience_group(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "Unknown"
    low = text.lower()
    if any(pattern in low for pattern in LOW_EXPERIENCE_PATTERNS):
        return "Low"
    if any(pattern in low for pattern in HIGH_EXPERIENCE_PATTERNS):
        return "High"
    return text


def condition_id(row: pd.Series) -> str:
    wwr = int(float(row["WWR"])) if pd.notna(row.get("WWR")) and str(row.get("WWR")).strip() else "NA"
    raw_complexity = row.get("Complexity")
    if pd.notna(raw_complexity) and str(raw_complexity).strip():
        complexity = str(int(float(raw_complexity)))
    else:
        complexity = re.sub(r"^C", "", str(row.get("Cond") or "NA").strip(), flags=re.IGNORECASE)
    return f"C{complexity}_W{wwr}"


def wwr_numeric(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"\d+(\.\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "exclude" not in df.columns:
        return df.copy()
    excluded = df["exclude"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return df.loc[~excluded].copy()
