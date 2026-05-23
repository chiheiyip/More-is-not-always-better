from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

import pandas as pd

from more_is_not_always_better.discovery import (
    build_participants_from_roots,
    build_scene_manifest_from_eye_root,
    load_questionnaire_metadata,
    scan_eye_raw,
)
from more_is_not_always_better.eye_batch import run_eye_aoi_batch
from more_is_not_always_better.fusion import run_fusion
from more_is_not_always_better.alignment import run_alignment_qc


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_eye_batch_and_fusion_smoke(tmp_path: Path) -> None:
    eye_out = tmp_path / "eye"
    fusion_out = tmp_path / "fusion"

    eye_paths = run_eye_aoi_batch(
        participants_csv=FIXTURES / "participants.csv",
        scene_manifest_csv=FIXTURES / "scene_manifest.csv",
        outdir=eye_out,
        dwell_mode="fixation",
    )
    assert eye_paths["class"].exists()
    eye_class = pd.read_csv(eye_paths["class"])
    assert {"participant_id", "scene_id", "class_name", "dwell_time_ms", "TTFF_ms", "fixation_count"}.issubset(eye_class.columns)
    assert not eye_class.duplicated(["participant_id", "scene_id", "class_name"]).any()

    fusion_paths = run_fusion(
        participants_csv=FIXTURES / "participants.csv",
        scene_manifest_csv=FIXTURES / "scene_manifest.csv",
        eeg_scene_csv=FIXTURES / "eeg" / "all_subjects_scene_level.csv",
        eye_aoi_class_csv=eye_paths["class"],
        outdir=fusion_out,
        expected_scenes_per_subject=2,
        duration_tolerance_s=2.0,
    )

    aligned_scene = pd.read_csv(fusion_paths["aligned_scene"])
    sync_qc = pd.read_csv(fusion_paths["sync_qc"])
    aligned_timebin = pd.read_csv(fusion_paths["aligned_timebin"])

    assert {"participant_id", "scene_id", "class_name", "O_alpha", "dwell_time_ms"}.issubset(aligned_scene.columns)
    assert not aligned_scene.duplicated(["participant_id", "scene_id", "class_name"]).any()
    assert len(sync_qc) == 2
    assert sync_qc["duration_mismatch"].astype(str).str.lower().isin(["false", "0"]).all()
    assert {"participant_id", "scene_id", "bin_start_ms", "bin_end_ms", "class_name", "eeg_O_alpha"}.issubset(aligned_timebin.columns)
    assert len(aligned_timebin) > 0


def test_eye_batch_without_aoi_outputs_whole_scene(tmp_path: Path) -> None:
    manifest = tmp_path / "scene_manifest_no_aoi.csv"
    eye_csv = FIXTURES / "eye" / "P01_scene01.csv"
    manifest.write_text(
        "participant_id,scene_id,block,position,scene_name,eye_csv_path,aoi_json_path,WWR,Cond,Complexity,eye_offset_ms\n"
        f"P01,1,1,1,scene_01,{eye_csv.as_posix()},,0.2,A,1,0\n",
        encoding="utf-8",
    )

    eye_paths = run_eye_aoi_batch(
        participants_csv=FIXTURES / "participants.csv",
        scene_manifest_csv=manifest,
        outdir=tmp_path / "eye_no_aoi",
    )
    eye_class = pd.read_csv(eye_paths["class"])
    eye_qc = pd.read_csv(eye_paths["qc"])

    assert eye_class.loc[0, "class_name"] == "whole_scene"
    assert bool(eye_class.loc[0, "aoi_available"]) is False
    assert eye_class.loc[0, "samples"] == 5
    assert bool(eye_qc.loc[0, "missing_aoi_file"]) is True


def test_manifest_builder_generates_raw_root_manifests_without_alias(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    _write_eye_csv(eye_root / "(1-1-1\u30012-1-1) \u7ec41-C1W45" / "raw_\u5f20\u4e09_260101000001_0207000001.csv")
    _write_eye_csv(eye_root / "(1-1-2\u30012-1-2) \u7ec41-C0W15" / "raw_\u5f20\u4e09_260101000001_0207000002.csv")
    _write_eeg_pair(eeg_root, "\u5f20\u4e09")

    participants_csv = tmp_path / "participants.csv"
    scene_manifest_csv = tmp_path / "scene_manifest.csv"
    participants = build_participants_from_roots(eye_root, eeg_root, out_csv=participants_csv)
    scene_manifest = build_scene_manifest_from_eye_root(eye_root, participants_csv, out_csv=scene_manifest_csv)
    eye_raw = scan_eye_raw(eye_root)

    assert participants.loc[0, "participant_id"] == "\u5f20\u4e09"
    assert bool(participants.loc[0, "exclude"]) is False
    assert participants.loc[0, "eye_scene_file_count"] == 2
    assert eye_raw["participant_id"].tolist() == ["\u5f20\u4e09", "\u5f20\u4e09"]
    assert sorted(scene_manifest["scene_id"].tolist()) == [1, 2]
    assert scene_manifest["alias_source"].fillna("").eq("").all()


def test_manifest_builder_supports_manual_alias_csv(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    _write_eye_csv(eye_root / "(1-1-1\u30012-1-1) \u7ec41-C1W45" / "raw_User1_260101000001_0207000001.csv")
    _write_eeg_pair(eeg_root, "\u5f20\u4e09")
    alias_csv = tmp_path / "aliases.csv"
    alias_csv.write_text("eye_subject_id,participant_id\nUser1,\u5f20\u4e09\n", encoding="utf-8")

    participants_csv = tmp_path / "participants.csv"
    scene_manifest_csv = tmp_path / "scene_manifest.csv"
    participants = build_participants_from_roots(eye_root, eeg_root, out_csv=participants_csv, eye_alias_csv=alias_csv)
    scene_manifest = build_scene_manifest_from_eye_root(eye_root, participants_csv, out_csv=scene_manifest_csv, eye_alias_csv=alias_csv)

    assert participants.loc[0, "participant_id"] == "\u5f20\u4e09"
    assert bool(participants.loc[0, "exclude"]) is False
    assert scene_manifest.loc[0, "participant_id"] == "\u5f20\u4e09"
    assert scene_manifest.loc[0, "eye_subject_alias"] == "User1"
    assert scene_manifest.loc[0, "alias_source"] == "manual"


def test_questionnaire_order_flows_into_manifest(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    _write_eye_csv(eye_root / "(1-1-1\u30012-1-1) \u7ec41-C1W45" / "raw_\u5f20\u4e09_260101000001_0207000001.csv")
    _write_eye_csv(eye_root / "(1-2-1\u30012-2-5) \u7ec42-C0W45" / "raw_\u5f20\u4e09_260101000001_0207000002.csv")
    _write_eeg_pair(eeg_root, "\u5f20\u4e09")
    questionnaire = tmp_path / "questionnaire.xlsx"
    _write_minimal_xlsx(
        questionnaire,
        [
            ["\u59d3\u540d", "Q1.8_\u573a\u666f\u987a\u5e8f\u7f16\u53f7", "Q1.4_\u4e52\u4e53\u7403\u7ecf\u9a8c\uff1a", "Q1.5_\u8fd1 6 \u4e2a\u6708\u5e73\u5747\u8fd0\u52a8\u9891\u7387\uff1a"],
            ["\u5f20\u4e09", 2, "experienced", "weekly"],
        ],
    )

    participants_csv = tmp_path / "participants.csv"
    participants = build_participants_from_roots(eye_root, eeg_root, out_csv=participants_csv, questionnaire_xlsx=questionnaire)
    scene_manifest = build_scene_manifest_from_eye_root(eye_root, participants_csv)
    qmeta = load_questionnaire_metadata(questionnaire)

    assert participants.loc[0, "Order"] == 2
    assert participants.loc[0, "Experience"] == "experienced"
    assert qmeta.loc[0, "Order"] == 2
    assert sorted(scene_manifest["scene_id"].tolist()) == [1, 11]


def test_alignment_qc_estimates_time_sync_map(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene_manifest = tmp_path / "scene_manifest.csv"
    eeg_scene = tmp_path / "eeg_scene.csv"
    eye1 = tmp_path / "eye1.csv"
    eye2 = tmp_path / "eye2.csv"
    eye1.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n1000,1,1\n2000,1,1\n3000,1,1\n", encoding="utf-8")
    eye2.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n5000,1,1\n6000,1,1\n7000,1,1\n", encoding="utf-8")
    participants.write_text("participant_id,eeg_subject_id,eye_subject_id,exclude\nP01,P01,P01,false\n", encoding="utf-8")
    scene_manifest.write_text(
        "participant_id,scene_id,eye_csv_path,aoi_json_path,eye_offset_ms\n"
        f"P01,1,{eye1.as_posix()},,0\n"
        f"P01,2,{eye2.as_posix()},,0\n",
        encoding="utf-8",
    )
    eeg_scene.write_text(
        "subject_id,scene_id,view_start_s,view_end_s,view_dur_s\n"
        "P01,1,11,13,2\n"
        "P01,2,15,17,2\n",
        encoding="utf-8",
    )

    out = run_alignment_qc(participants, scene_manifest, eeg_scene, outdir=tmp_path / "alignment")
    sync_map = pd.read_csv(out["sync_map"])
    landmarks = pd.read_csv(out["landmarks"])

    assert abs(sync_map.loc[0, "time_sync_slope"] - 1.0) < 1e-9
    assert abs(sync_map.loc[0, "time_sync_offset_ms"] - 10000.0) < 1e-6
    assert sync_map.loc[0, "median_abs_residual_ms"] < 1e-6
    assert len(landmarks) == 4


def _write_eye_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px]\n0,1,1\n", encoding="utf-8")


def _write_eeg_pair(root: Path, subject: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{subject}.set").write_text("placeholder", encoding="utf-8")
    (root / f"{subject}.fdt").write_text("placeholder", encoding="utf-8")


def _write_minimal_xlsx(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_xlsx_col(c_idx)}{r_idx}"
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>'
        ))
        zf.writestr("_rels/.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        ))
        zf.writestr("xl/workbook.xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ))
        zf.writestr("xl/_rels/workbook.xml.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>'
        ))
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _xlsx_col(index: int) -> str:
    out = ""
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(ord("A") + rem) + out
    return out
