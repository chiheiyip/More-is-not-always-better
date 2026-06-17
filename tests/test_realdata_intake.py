from __future__ import annotations

from pathlib import Path
import importlib.util
from xml.sax.saxutils import escape
import zipfile

import pandas as pd

from more_is_not_always_better.discovery import (
    build_participants_from_roots,
    build_scene_manifest_from_eye_root,
    load_questionnaire_long_from_wjx,
    parse_scene_folder,
    scan_eye_raw,
)


def test_simple_eye_folder_manifest_and_aoi_json(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    _write_eye_csv(eye_root / "1-C0W15" / "raw_张三_260530201640_0617145623.csv")
    (eye_root / "1-C0W15" / "1-C0W15.json").write_text('{"aoi_classes": {}}', encoding="utf-8")
    _write_eeg_pair(eeg_root, "张三")

    meta = parse_scene_folder("1-C0W15")
    assert meta is not None
    assert meta.block == 1
    assert meta.position == 1
    assert meta.scene_id == 1
    assert meta.complexity == 0

    eye = scan_eye_raw(eye_root)
    assert eye.loc[0, "aoi_json_path"].endswith("1-C0W15.json")
    assert eye.loc[0, "folder_order_scheme"] == "simple_condition_folder"

    participants_csv = tmp_path / "participants.csv"
    participants = build_participants_from_roots(eye_root, eeg_root, out_csv=participants_csv)
    scene = build_scene_manifest_from_eye_root(eye_root, participants_csv)

    assert bool(participants.loc[0, "exclude"]) is False
    assert scene.loc[0, "scene_id"] == 1
    assert scene.loc[0, "block"] == 1
    assert scene.loc[0, "position"] == 1
    assert scene.loc[0, "Cond"] == "C0"
    assert scene.loc[0, "WWR"] == 15
    assert scene.loc[0, "aoi_json_path"].endswith("1-C0W15.json")


def test_wjx_questionnaire_long_parser_scores_chinese_points(tmp_path: Path) -> None:
    xlsx = tmp_path / "questionnaire.xlsx"
    _write_minimal_xlsx(
        xlsx,
        [
            [
                "Q1.0_姓名：",
                "Q1.8_场景顺序编号",
                "Q2.1_S1. 这个空间整体上适合打乒乓球。_",
                "Q2.5_S5.愉悦度评价_",
                "Q8.1_B1. 出现功能性器材要素的场景整体给我一种信息更为丰富的感觉。_",
                "Q9.1_S1. 这个空间整体上适合打乒乓球。_",
                "Q15.3_B3. 即便出现这些功能性器材要素，这个空间整体看起来仍然是有序的。_",
                "Q16.1_1. 我在佩戴和使用 VR 设备时感到舒适。_",
            ],
            ["张三", 1, "5分", "8分", "6分", "7分", "2分", "4分"],
        ],
    )

    long = load_questionnaire_long_from_wjx(xlsx)
    by_scene = long.set_index("scene_id")

    assert by_scene.loc[1, "S1"] == 5.0
    assert by_scene.loc[1, "S5"] == 8.0
    assert by_scene.loc[6, "B1"] == 6.0
    assert by_scene.loc[7, "S1"] == 7.0
    assert by_scene.loc[12, "B3"] == 2.0
    assert by_scene["IPQ1"].dropna().eq(4.0).all()


def test_scripts_do_not_reference_order_material_roots() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    checked = [script_dir / "preflight_raw_inputs.py", script_dir / "run_realdata_linktest.py"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in checked)
    assert "顺序1" not in text
    assert "顺序2-new" not in text


def test_preflight_json_safe_converts_non_string_keys() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "preflight_raw_inputs.py"
    spec = importlib.util.spec_from_file_location("preflight_raw_inputs", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    safe = module._json_safe({pd.Series([1], dtype="int64").iloc[0]: {"x": pd.Series([2], dtype="int64").iloc[0]}})
    assert safe == {"1": {"x": 2}}


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
