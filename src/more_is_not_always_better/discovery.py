from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import pandas as pd

from .io import write_csv


EYE_FILE_RE = re.compile(r"^raw_(?P<subject>.+?)_(?P<record_id>\d+)_(?P<split_id>\d+)\.csv$", re.IGNORECASE)
SCENE_DIR_RE = re.compile(
    r"^\((?P<code_order1>\d+-\d+-\d+)\u3001(?P<code_order2>\d+-\d+-\d+)\)\s*"
    r"(?P<scene_group>\u7ec4\d+)-C(?P<cond>\d+)W(?P<wwr>\d+)$"
)
SIMPLE_SCENE_DIR_RE = re.compile(r"^(?P<block>\d+)-C(?P<cond>\d+)W(?P<wwr>\d+)$", re.IGNORECASE)
SIMPLE_CONDITION_POSITION = {
    "C0W15": 1,
    "C0W45": 2,
    "C0W75": 3,
    "C1W15": 4,
    "C1W45": 5,
    "C1W75": 6,
}
GENERIC_EYE_SUBJECTS = {"user", "user1", "user2", "test", "pilot", "practice"}
SUFFIX_NOTE_RE = re.compile(r"^(?P<name>.+?)-\d+-\d+$")
ETHNIC_NAME_SEPARATOR_RE = re.compile(r"[·•]")
NEW_ORDER2_START_DATE = date(2026, 5, 1)
NEW_ORDER2_BY_BLOCK = {
    1: {
        "C0W75": ("1", 1),
        "C1W15": ("2", 2),
        "C0W45": ("3", 3),
        "C1W75": ("4", 4),
        "C0W15": ("5", 5),
        "C1W45": ("6", 6),
    },
    2: {
        "C1W15": ("a", 1),
        "C0W15": ("b", 2),
        "C1W75": ("c", 3),
        "C0W75": ("d", 4),
        "C1W45": ("e", 5),
        "C0W45": ("f", 6),
    },
}
QUESTIONNAIRE_COLUMN_CANDIDATES = {
    "participant_id": ["Q1.0_姓名：", "姓名", "name"],
    "Order": ["Q1.8_场景顺序编号", "Q1.8_场景顺序编号：", "Q1.8_场景顺序编号", "Q1.8"],
    "Experience": ["Q1.4_乒乓球经验：", "Q1.4"],
    "SportFreq": ["Q1.5_近 6 个月平均运动频率：", "Q1.5"],
    "Age": ["Q1.1_年龄（岁）：", "Q1.1"],
    "Gender": ["Q1.2_性别：", "Q1.2"],
    "RightHanded": ["Q1.3_是否惯用右手：", "Q1.3"],
    "VRExperience": ["Q1.6_VR 使用经验：", "Q1.6"],
    "MotionSickness": ["Q1.7_是否容易晕动/晕 VR：", "Q1.7"],
}


@dataclass(frozen=True)
class SceneFolderMeta:
    folder_name: str
    code_order1: str
    code_order2: str
    scene_group: str
    cond: str
    wwr: int
    condition_code: str
    complexity: int
    block: int | None = None
    position: int | None = None
    scene_id: int | None = None
    order_scheme: str = "coded_folder"


def scan_eeg_raw(eeg_root: str | Path) -> pd.DataFrame:
    root = Path(eeg_root)
    rows: list[dict] = []
    for set_file in sorted(root.glob("*.set")):
        fdt_file = set_file.with_suffix(".fdt")
        rows.append({
            "participant_id": set_file.stem,
            "eeg_subject_id": set_file.stem,
            "eeg_set_path": str(set_file),
            "eeg_fdt_path": str(fdt_file),
            "has_set": True,
            "has_fdt": fdt_file.exists(),
        })
    return pd.DataFrame(rows)


def scan_eye_raw(eye_root: str | Path, include_adaptation: bool = False) -> pd.DataFrame:
    root = Path(eye_root)
    rows: list[dict] = []
    for csv_file in sorted(root.rglob("*.csv")):
        scene_folder = csv_file.parent.name
        if not include_adaptation and _is_adaptation_folder(scene_folder):
            continue
        match = EYE_FILE_RE.match(csv_file.name)
        if not match:
            continue
        folder_meta = parse_scene_folder(scene_folder)
        base = {
            "participant_id": match.group("subject").strip(),
            "eye_subject_id": match.group("subject").strip(),
            "raw_eye_subject_id": match.group("subject").strip(),
            "eye_csv_path": str(csv_file),
            "aoi_json_path": str(_find_aoi_json(csv_file.parent, scene_folder)),
            "eye_record_id": match.group("record_id"),
            "eye_split_id": match.group("split_id"),
            "source_folder": scene_folder,
            "is_adaptation": _is_adaptation_folder(scene_folder),
        }
        if folder_meta is not None:
            base.update({
                "order_code_1": folder_meta.code_order1,
                "order_code_2": folder_meta.code_order2,
                "scene_group": folder_meta.scene_group,
                "condition_code": folder_meta.condition_code,
                "Cond": f"C{folder_meta.cond}",
                "WWR": folder_meta.wwr,
                "Complexity": folder_meta.complexity,
                "folder_block": folder_meta.block,
                "folder_position": folder_meta.position,
                "folder_scene_id": folder_meta.scene_id,
                "folder_order_scheme": folder_meta.order_scheme,
            })
        rows.append(base)
    return pd.DataFrame(rows)


def apply_eye_aliases(
    eye: pd.DataFrame,
    alias_csv: Optional[str | Path] = None,
    auto_alias: bool = True,
) -> pd.DataFrame:
    out = eye.copy()
    if out.empty:
        return out
    if "eye_subject_alias" not in out.columns:
        out["eye_subject_alias"] = ""
    if "alias_source" not in out.columns:
        out["alias_source"] = ""

    if auto_alias:
        suffix_target = out["participant_id"].map(_strip_suffix_note)
        suffix_mask = suffix_target.ne(out["participant_id"])
        out.loc[suffix_mask, "eye_subject_alias"] = out.loc[suffix_mask, "participant_id"]
        out.loc[suffix_mask, "participant_id"] = suffix_target.loc[suffix_mask]
        out.loc[suffix_mask, "eye_subject_id"] = suffix_target.loc[suffix_mask]
        out.loc[suffix_mask, "alias_source"] = "suffix_note"
        for record_id, sub in out.groupby("eye_record_id", dropna=False):
            subjects = sorted(set(str(v) for v in sub["participant_id"].dropna()))
            generic = [s for s in subjects if _is_generic_eye_subject(s)]
            named = [s for s in subjects if not _is_generic_eye_subject(s)]
            if len(named) != 1 or not generic:
                continue
            target = named[0]
            mask = (out["eye_record_id"] == record_id) & out["participant_id"].map(_is_generic_eye_subject)
            out.loc[mask, "eye_subject_alias"] = out.loc[mask, "participant_id"]
            out.loc[mask, "participant_id"] = target
            out.loc[mask, "eye_subject_id"] = target
            out.loc[mask, "alias_source"] = "record_id"

    if alias_csv is not None:
        aliases = pd.read_csv(alias_csv, encoding="utf-8-sig")
        source_col = _first_existing(aliases.columns, ["eye_subject_id", "source_subject", "alias", "raw_eye_subject_id"])
        target_col = _first_existing(aliases.columns, ["participant_id", "target_subject", "canonical_subject"])
        if source_col is None or target_col is None:
            raise ValueError("alias_csv must contain eye_subject_id/source_subject and participant_id/target_subject columns")
        for _, row in aliases.iterrows():
            source = str(row[source_col]).strip()
            target = str(row[target_col]).strip()
            if not source or not target:
                continue
            mask = out["participant_id"].eq(source) | out["raw_eye_subject_id"].eq(source)
            out.loc[mask, "eye_subject_alias"] = out.loc[mask, "raw_eye_subject_id"]
            out.loc[mask, "participant_id"] = target
            out.loc[mask, "eye_subject_id"] = target
            out.loc[mask, "alias_source"] = "manual"
    return out


def build_participants_from_roots(
    eye_root: str | Path,
    eeg_root: str | Path,
    out_csv: Optional[str | Path] = None,
    include_adaptation: bool = False,
    eye_alias_csv: Optional[str | Path] = None,
    questionnaire_xlsx: Optional[str | Path] = None,
) -> pd.DataFrame:
    eye = apply_eye_aliases(scan_eye_raw(eye_root, include_adaptation=include_adaptation), alias_csv=eye_alias_csv)
    eeg = scan_eeg_raw(eeg_root)
    questionnaire = load_questionnaire_metadata(questionnaire_xlsx) if questionnaire_xlsx else pd.DataFrame()
    questionnaire_index = questionnaire.set_index("participant_id") if not questionnaire.empty else pd.DataFrame()
    eye_counts = eye.groupby("participant_id")["eye_csv_path"].nunique().rename("eye_scene_file_count") if not eye.empty else pd.Series(dtype=int)
    eeg_ids = set(eeg["participant_id"]) if not eeg.empty else set()
    eye_ids = set(eye["participant_id"]) if not eye.empty else set()
    questionnaire_ids = set(questionnaire["participant_id"]) if not questionnaire.empty else set()
    questionnaire_required = questionnaire_xlsx is not None
    all_ids = sorted(eeg_ids | eye_ids | questionnaire_ids)

    eeg_paths = eeg.set_index("participant_id") if not eeg.empty else pd.DataFrame()
    rows: list[dict] = []
    for participant_id in all_ids:
        has_eeg = participant_id in eeg_ids
        has_eye = participant_id in eye_ids
        has_questionnaire = participant_id in questionnaire_ids if questionnaire_required else False
        exclude_reasons = []
        if not has_eeg:
            exclude_reasons.append("missing_eeg")
        if not has_eye:
            exclude_reasons.append("missing_eye")
        if questionnaire_required and not has_questionnaire:
            exclude_reasons.append("missing_questionnaire")
        row = {
            "participant_id": participant_id,
            "eeg_subject_id": participant_id if has_eeg else "",
            "eye_subject_id": participant_id if has_eye else "",
            "SportFreq": "",
            "Experience": "",
            "Order": "",
            "exclude": bool(exclude_reasons),
            "ExcludeReason": ";".join(exclude_reasons),
            "has_eeg_raw": has_eeg,
            "has_eye_raw": has_eye,
            "has_questionnaire": has_questionnaire,
            "eye_scene_file_count": int(eye_counts.get(participant_id, 0)),
        }
        if has_eeg and not eeg_paths.empty:
            row["eeg_set_path"] = eeg_paths.loc[participant_id, "eeg_set_path"]
            row["eeg_fdt_path"] = eeg_paths.loc[participant_id, "eeg_fdt_path"]
            row["has_fdt"] = bool(eeg_paths.loc[participant_id, "has_fdt"])
        if not questionnaire_index.empty and participant_id in questionnaire_index.index:
            qrow = questionnaire_index.loc[participant_id]
            for col in [
                "Order",
                "Experience",
                "SportFreq",
                "Age",
                "GenderRaw",
                "Gender",
                "RightHanded",
                "VRExperience",
                "MotionSickness",
            ]:
                row[col] = qrow.get(col, "")
            row["has_questionnaire"] = True
        rows.append(row)

    out = pd.DataFrame(rows)
    if out_csv is not None:
        write_csv(out, out_csv)
    return out


def build_scene_manifest_from_eye_root(
    eye_root: str | Path,
    participants_csv: str | Path,
    out_csv: Optional[str | Path] = None,
    include_adaptation: bool = False,
    default_order: int = 1,
    eye_offset_ms: float = 0.0,
    eye_alias_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    eye = apply_eye_aliases(scan_eye_raw(eye_root, include_adaptation=include_adaptation), alias_csv=eye_alias_csv)
    participants = pd.read_csv(participants_csv, encoding="utf-8-sig")
    if "participant_id" not in participants.columns:
        raise ValueError("participants_csv must contain participant_id")
    if "Order" not in participants.columns:
        participants["Order"] = ""
    participants["participant_id"] = participants["participant_id"].astype(str).str.strip()
    order_map = {
        row["participant_id"]: _normalize_order(row.get("Order"), default_order)
        for _, row in participants.iterrows()
    }
    if "exclude" in participants.columns:
        active_mask = ~participants["exclude"].map(_truthy)
    else:
        active_mask = pd.Series(True, index=participants.index)
    active_ids = set(participants.loc[active_mask, "participant_id"])
    eye = eye.loc[eye["participant_id"].isin(active_ids)].copy()

    rows: list[dict] = []
    for _, row in eye.iterrows():
        participant_id = str(row["participant_id"])
        order = order_map.get(participant_id, default_order)
        order_scheme, order_code, block, position, scene_id = _resolve_order_position(row, order)
        order_missing = _order_is_missing(participants, participant_id)
        rows.append({
            "participant_id": participant_id,
            "scene_id": scene_id,
            "block": block,
            "position": position,
            "scene_name": row.get("condition_code", row.get("source_folder", "")),
            "eye_csv_path": row["eye_csv_path"],
            "aoi_json_path": row.get("aoi_json_path", ""),
            "WWR": row.get("WWR", ""),
            "Cond": row.get("Cond", ""),
            "Complexity": row.get("Complexity", ""),
            "eye_offset_ms": eye_offset_ms,
            "participant_order": order,
            "order_scheme": order_scheme,
            "order_missing": order_missing,
            "order_code": order_code,
            "order_code_1": row.get("order_code_1", ""),
            "order_code_2": row.get("order_code_2", ""),
            "source_folder": row.get("source_folder", ""),
            "eye_record_id": row.get("eye_record_id", ""),
            "eye_split_id": row.get("eye_split_id", ""),
            "raw_eye_subject_id": row.get("raw_eye_subject_id", ""),
            "eye_subject_alias": row.get("eye_subject_alias", ""),
            "alias_source": row.get("alias_source", ""),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["participant_id", "scene_id", "source_folder"]).reset_index(drop=True)
    duplicates = out[out.duplicated(["participant_id", "scene_id"], keep=False)] if not out.empty else pd.DataFrame()
    if not duplicates.empty:
        sample = duplicates[["participant_id", "scene_id", "source_folder"]].head(10).to_dict("records")
        raise ValueError(f"Generated duplicate participant_id + scene_id rows. Sample: {sample}")
    if out_csv is not None:
        write_csv(out, out_csv)
    return out


def parse_scene_folder(folder_name: str) -> Optional[SceneFolderMeta]:
    match = SCENE_DIR_RE.match(folder_name)
    if not match:
        simple = SIMPLE_SCENE_DIR_RE.match(folder_name)
        if not simple:
            return None
        block = int(simple.group("block"))
        cond = simple.group("cond")
        wwr = int(simple.group("wwr"))
        condition_code = f"C{cond}W{wwr}"
        position = SIMPLE_CONDITION_POSITION.get(condition_code.upper())
        scene_id = (block - 1) * 6 + position if position is not None else None
        return SceneFolderMeta(
            folder_name=folder_name,
            code_order1="",
            code_order2="",
            scene_group=str(block),
            cond=cond,
            wwr=wwr,
            condition_code=condition_code,
            complexity=int(cond),
            block=block,
            position=position,
            scene_id=scene_id,
            order_scheme="simple_condition_folder",
        )
    scene_group = match.group("scene_group")
    group_number_match = re.search(r"\d+", scene_group)
    complexity = int(group_number_match.group(0)) if group_number_match else 0
    cond = match.group("cond")
    wwr = int(match.group("wwr"))
    return SceneFolderMeta(
        folder_name=folder_name,
        code_order1=match.group("code_order1"),
        code_order2=match.group("code_order2"),
        scene_group=scene_group,
        cond=cond,
        wwr=wwr,
        condition_code=f"C{cond}W{wwr}",
        complexity=complexity,
    )


def summarize_roots(
    eye_root: str | Path,
    eeg_root: str | Path,
    eye_alias_csv: Optional[str | Path] = None,
    questionnaire_xlsx: Optional[str | Path] = None,
) -> dict:
    eye_raw = scan_eye_raw(eye_root)
    eye = apply_eye_aliases(eye_raw, alias_csv=eye_alias_csv)
    eeg = scan_eeg_raw(eeg_root)
    eye_ids = set(eye["participant_id"]) if not eye.empty else set()
    eeg_ids = set(eeg["participant_id"]) if not eeg.empty else set()
    scene_folders = sorted(eye["source_folder"].unique().tolist()) if not eye.empty else []
    alias_rows = eye.loc[eye.get("alias_source", "") != ""] if not eye.empty else pd.DataFrame()
    questionnaire = load_questionnaire_metadata(questionnaire_xlsx) if questionnaire_xlsx else pd.DataFrame()
    questionnaire_ids = set(questionnaire["participant_id"]) if not questionnaire.empty else set()
    tri = eye_ids & eeg_ids & questionnaire_ids if questionnaire_xlsx else eye_ids & eeg_ids
    return {
        "eye_csv_count": int(len(eye_raw)),
        "eye_subject_count_raw": int(eye_raw["participant_id"].nunique()) if not eye_raw.empty else 0,
        "eye_subject_count_after_alias": int(len(eye_ids)),
        "eye_suffix_note_alias_rows": int(eye["alias_source"].eq("suffix_note").sum()) if not eye.empty and "alias_source" in eye.columns else 0,
        "eye_suffix_note_alias_subjects": sorted(eye.loc[eye["alias_source"].eq("suffix_note"), "raw_eye_subject_id"].dropna().unique().tolist()) if not eye.empty and "alias_source" in eye.columns else [],
        "eye_scene_folder_count": int(len(scene_folders)),
        "eye_aoi_json_count": int(eye_raw["aoi_json_path"].replace("", pd.NA).dropna().nunique()) if not eye_raw.empty and "aoi_json_path" in eye_raw.columns else 0,
        "eye_missing_aoi_json_rows": int(eye_raw["aoi_json_path"].eq("").sum()) if not eye_raw.empty and "aoi_json_path" in eye_raw.columns else 0,
        "eeg_set_count": int(len(eeg)),
        "eeg_fdt_count": int(eeg["has_fdt"].sum()) if not eeg.empty else 0,
        "matched_subject_count": int(len(eye_ids & eeg_ids)),
        "trimodal_subject_count": int(len(tri)),
        "eye_only_subjects": sorted(eye_ids - eeg_ids),
        "eeg_only_subjects": sorted(eeg_ids - eye_ids),
        "questionnaire_subject_count": int(len(questionnaire_ids)),
        "questionnaire_missing_for_eye_subjects": sorted(eye_ids - questionnaire_ids) if questionnaire_xlsx else [],
        "questionnaire_order_counts": questionnaire["Order"].value_counts(dropna=False).to_dict() if not questionnaire.empty else {},
        "aliased_eye_rows": int(len(alias_rows)),
        "aliases": sorted(alias_rows[["raw_eye_subject_id", "participant_id", "eye_record_id", "alias_source"]].drop_duplicates().to_dict("records"), key=lambda x: str(x)) if not alias_rows.empty else [],
        "scene_folders": scene_folders,
    }


def load_questionnaire_metadata(path: str | Path) -> pd.DataFrame:
    workbook = Path(path)
    if not workbook.exists():
        raise FileNotFoundError(f"Questionnaire workbook not found: {workbook}")
    df = _read_xlsx_first_sheet(workbook)
    resolved: dict[str, str] = {}
    for output_col, candidates in QUESTIONNAIRE_COLUMN_CANDIDATES.items():
        actual = _first_existing(df.columns, candidates)
        if actual is not None:
            resolved[output_col] = actual
    if "participant_id" not in resolved:
        raise ValueError("Questionnaire workbook must contain a participant name column")

    out = pd.DataFrame()
    raw_participant_id = df[resolved["participant_id"]].astype(str).str.strip()
    out["questionnaire_subject_id"] = raw_participant_id
    out["participant_id"] = raw_participant_id.map(_canonical_questionnaire_subject_id)
    for output_col in [
        "Order",
        "Experience",
        "SportFreq",
        "Age",
        "Gender",
        "RightHanded",
        "VRExperience",
        "MotionSickness",
    ]:
        out[output_col] = df[resolved[output_col]] if output_col in resolved else ""

    out = out.loc[out["participant_id"].ne("") & out["participant_id"].ne("nan")].copy()
    out["Order"] = pd.to_numeric(out["Order"], errors="coerce").astype("Int64")
    out["GenderRaw"] = out["Gender"]
    out["Gender"] = out["GenderRaw"].map(_standardize_gender)
    if out["participant_id"].duplicated().any():
        dupes = out.loc[out["participant_id"].duplicated(keep=False), ["questionnaire_subject_id", "participant_id"]].to_dict("records")
        raise ValueError(f"Questionnaire contains duplicate participant names after canonicalization: {dupes[:10]}")
    return out


def load_questionnaire_long_from_wjx(
    path: str | Path,
    max_participants: int | None = None,
    participants: Iterable[str] | None = None,
) -> pd.DataFrame:
    workbook = Path(path)
    if not workbook.exists():
        raise FileNotFoundError(f"Questionnaire workbook not found: {workbook}")
    wide = _read_xlsx_first_sheet(workbook)
    participant_col = _first_matching_column(wide.columns, [r"^Q1\.0_.*姓名", r"^姓名$", r"name"])
    if participant_col is None:
        raise ValueError("Questionnaire workbook must contain Q1.0/name participant column")
    wanted = {str(v).strip() for v in participants or [] if str(v).strip()}
    selected_participants: list[str] = []
    rows_by_key: dict[tuple[str, int], dict] = {}
    ipq_by_participant: dict[str, dict[str, float | None]] = {}
    ipq_cols = {f"IPQ{i}": _first_matching_column(wide.columns, [rf"^Q16\.{i}_", rf"IPQ{i}"]) for i in range(1, 7)}

    for _, row in wide.iterrows():
        questionnaire_subject_id = str(row.get(participant_col, "")).strip()
        participant_id = _canonical_questionnaire_subject_id(questionnaire_subject_id)
        if not participant_id or participant_id.lower() == "nan":
            continue
        if wanted and participant_id not in wanted:
            continue
        if participant_id not in selected_participants:
            selected_participants.append(participant_id)
        if max_participants is not None and len(selected_participants) > max_participants:
            break
        ipq_by_participant[participant_id] = {item: _score_value(row[col]) if col is not None else None for item, col in ipq_cols.items()}
        for col in wide.columns:
            parsed = _parse_wjx_question_column(str(col))
            if parsed is None:
                continue
            scene_id, item = parsed
            key = (participant_id, scene_id)
            rows_by_key.setdefault(
                key,
                {
                    "participant_id": participant_id,
                    "questionnaire_subject_id": questionnaire_subject_id,
                    "scene_id": scene_id,
                },
            )[item] = _score_value(row.get(col))

    out = pd.DataFrame(rows_by_key.values())
    if out.empty:
        return out
    for item in ipq_cols:
        out[item] = out["participant_id"].map(lambda pid, item=item: ipq_by_participant.get(pid, {}).get(item))
    return out.sort_values(["participant_id", "scene_id"]).reset_index(drop=True)


def _parse_wjx_question_column(name: str) -> tuple[int, str] | None:
    match = re.match(r"^Q(?P<q>\d+)\.\d+_(?P<item>S[1-5]|B[1-3])\b", name, re.IGNORECASE)
    if not match:
        return None
    q_number = int(match.group("q"))
    item = match.group("item").upper()
    if item.startswith("S") and 2 <= q_number <= 7:
        scene_id = q_number - 1
    elif item.startswith("S") and 9 <= q_number <= 14:
        scene_id = q_number - 2
    elif item.startswith("B") and q_number == 8:
        scene_id = 6
    elif item.startswith("B") and q_number == 15:
        scene_id = 12
    else:
        return None
    return scene_id, item


def _score_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _first_matching_column(columns: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for col in columns:
            if regex.search(str(col)):
                return str(col)
    return None


def _read_xlsx_first_sheet(workbook: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(workbook, sheet_name=0)
    except ImportError:
        return _read_xlsx_first_sheet_stdlib(workbook)


def _read_xlsx_first_sheet_stdlib(workbook: Path) -> pd.DataFrame:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "officeRel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(workbook) as zf:
        shared_strings = _read_shared_strings(zf, ns)
        workbook_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook_xml.find("main:sheets/main:sheet", ns)
        if first_sheet is None:
            return pd.DataFrame()
        rel_id = first_sheet.attrib.get(f"{{{ns['officeRel']}}}id")
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels_xml.findall("rel:Relationship", ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            raise ValueError(f"Cannot resolve first worksheet in {workbook}")
        sheet_path = "xl/" + target.lstrip("/")
        sheet_xml = ET.fromstring(zf.read(sheet_path))
        rows = []
        max_col = 0
        for row_el in sheet_xml.findall("main:sheetData/main:row", ns):
            values: dict[int, object] = {}
            for cell in row_el.findall("main:c", ns):
                col_index = _xlsx_col_to_index(cell.attrib.get("r", "A1"))
                values[col_index] = _xlsx_cell_value(cell, shared_strings, ns)
                max_col = max(max_col, col_index)
            rows.append([values.get(i, "") for i in range(max_col + 1)])
    if not rows:
        return pd.DataFrame()
    header = [str(v) for v in rows[0]]
    data = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
    return pd.DataFrame(data, columns=header)


def _read_shared_strings(zf: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("main:si", ns):
        text = "".join(t.text or "" for t in si.findall(".//main:t", ns))
        strings.append(text)
    return strings


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//main:t", ns))
    value_el = cell.find("main:v", ns)
    if value_el is None or value_el.text is None:
        return ""
    value = value_el.text
    if cell_type == "s":
        idx = int(value)
        return shared_strings[idx] if idx < len(shared_strings) else ""
    if cell_type in {"str", "b"}:
        return value
    try:
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return value


def _xlsx_col_to_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref.upper())
    if not letters:
        return 0
    index = 0
    for char in letters.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _scene_position_from_order_code(order_code: object) -> tuple[int, int, int]:
    text = str(order_code or "").strip()
    parts = [int(p) for p in text.split("-") if p.isdigit()]
    if len(parts) != 3:
        raise ValueError(f"Invalid order code: {order_code!r}")
    _, block, position = parts
    return block, position, (block - 1) * 6 + position


def _resolve_order_position(row: pd.Series, order: int) -> tuple[str, object, int, int, int]:
    folder_scene_id = row.get("folder_scene_id")
    if pd.notna(folder_scene_id) and str(folder_scene_id).strip():
        block = int(row.get("folder_block"))
        position = int(row.get("folder_position"))
        scene_id = int(folder_scene_id)
        return str(row.get("folder_order_scheme") or "simple_condition_folder"), row.get("condition_code", ""), block, position, scene_id
    selected_code = row.get(f"order_code_{order}") or row.get("order_code_1")
    if order == 2 and _uses_neworder2(row.get("eye_record_id")):
        old_block, _, _ = _scene_position_from_order_code(row.get("order_code_2") or selected_code)
        condition = _condition_key(row)
        try:
            label, position = NEW_ORDER2_BY_BLOCK[old_block][condition]
        except KeyError as exc:
            raise ValueError(
                f"Cannot map condition {condition!r} in block {old_block} to neworder2 "
                f"for folder {row.get('source_folder', '')!r}"
            ) from exc
        return "neworder2", label, old_block, position, (old_block - 1) * 6 + position
    block, position, scene_id = _scene_position_from_order_code(selected_code)
    return f"order{order}", selected_code, block, position, scene_id


def _uses_neworder2(eye_record_id: object) -> bool:
    experiment_date = _experiment_date_from_eye_record_id(eye_record_id)
    return experiment_date is not None and experiment_date >= NEW_ORDER2_START_DATE


def _experiment_date_from_eye_record_id(value: object) -> Optional[date]:
    match = re.match(r"(\d{6})", str(value or "").strip())
    if not match:
        return None
    text = match.group(1)
    year = 2000 + int(text[:2])
    month = int(text[2:4])
    day = int(text[4:6])
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _find_aoi_json(folder: Path, scene_folder: str) -> Path | str:
    preferred = folder / f"{scene_folder}.json"
    if preferred.exists():
        return preferred
    json_files = sorted(folder.glob("*.json"))
    return json_files[0] if len(json_files) == 1 else ""


def _condition_key(row: pd.Series) -> str:
    cond = str(row.get("Cond") or "").strip().upper().replace("_", "")
    if not cond:
        condition_code = str(row.get("condition_code") or "").strip().upper().replace("_", "")
        match = re.search(r"C\d+", condition_code)
        cond = match.group(0) if match else ""
    if not cond and pd.notna(row.get("Complexity")):
        cond = f"C{int(float(row.get('Complexity')))}"
    wwr_match = re.search(r"\d+(\.\d+)?", str(row.get("WWR") or row.get("condition_code") or ""))
    wwr = int(float(wwr_match.group(0))) if wwr_match else None
    if not cond or wwr is None:
        raise ValueError(f"Cannot derive condition key from row: {row.to_dict()}")
    return f"{cond}W{wwr}"


def _normalize_order(value: object, default_order: int) -> int:
    if value is None or pd.isna(value):
        return default_order
    match = re.search(r"[12]", str(value))
    return int(match.group(0)) if match else default_order


def _order_is_missing(participants: pd.DataFrame, participant_id: str) -> bool:
    values = participants.loc[participants["participant_id"] == participant_id, "Order"]
    if values.empty:
        return True
    value = values.iloc[0]
    return value is None or pd.isna(value) or str(value).strip() == ""


def _truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _standardize_gender(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return "Unknown"
    low = text.lower()
    if low in {"m", "male", "man"} or text in {"男", "男性", "男生"}:
        return "Male"
    if low in {"f", "female", "woman"} or text in {"女", "女性", "女生"}:
        return "Female"
    return text


def _is_adaptation_folder(folder_name: str) -> bool:
    return "适应" in folder_name or folder_name.lower() in {"adaptation", "practice"}


def _is_generic_eye_subject(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in GENERIC_EYE_SUBJECTS or re.fullmatch(r"user\d+", text) is not None


def _strip_suffix_note(value: object) -> str:
    text = str(value or "").strip()
    match = SUFFIX_NOTE_RE.match(text)
    return match.group("name") if match else text


def _canonical_questionnaire_subject_id(value: object) -> str:
    text = str(value or "").strip()
    parts = [part.strip() for part in ETHNIC_NAME_SEPARATOR_RE.split(text) if part.strip()]
    return parts[0] if len(parts) > 1 else text


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None
