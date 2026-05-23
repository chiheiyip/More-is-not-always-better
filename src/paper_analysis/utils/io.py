from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


FALSE_VALUES = {"", "0", "false", "no", "n", "none", "nan"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_text(text: str, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def resolve_path(value: object, base_dir: str | Path | None = None) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (Path(base_dir) if base_dir is not None else Path.cwd()) / path


def is_truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in FALSE_VALUES


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dupes = df[df.duplicated(keys, keep=False)]
    if not dupes.empty:
        sample = dupes[keys].head(10).to_dict("records")
        raise ValueError(f"{name} has duplicate keys {keys}. Sample: {sample}")


def normalize_ids(df: pd.DataFrame, scene_col: str = "scene_id") -> pd.DataFrame:
    out = df.copy()
    for col in ["participant_id", "subject_id", "eeg_subject_id", "eye_subject_id"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    if scene_col in out.columns:
        out[scene_col] = pd.to_numeric(out[scene_col], errors="coerce").astype("Int64")
    return out
