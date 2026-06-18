from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys
from xml.sax.saxutils import escape
import zipfile

import pandas as pd

from more_is_not_always_better.discovery import (
    apply_eye_aliases,
    build_participants_from_roots,
    build_scene_manifest_from_eye_root,
    load_questionnaire_long_from_wjx,
    parse_scene_folder,
    scan_eye_raw,
)
from paper_analysis.eeg.contract import validate_eeg_scene_summary_frame


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


def test_suffix_note_eye_alias_normalization(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    _write_eye_csv(eye_root / "1-C0W15" / "raw_牛雨雨-325-400_260526150342_0617145623.csv")
    _write_eye_csv(eye_root / "1-C0W45" / "raw_张三_260526150342_0617150451.csv")
    _write_eeg_pair(eeg_root, "牛雨雨")
    _write_eeg_pair(eeg_root, "张三")

    eye = apply_eye_aliases(scan_eye_raw(eye_root))
    by_raw = eye.set_index("raw_eye_subject_id")
    assert by_raw.loc["牛雨雨-325-400", "participant_id"] == "牛雨雨"
    assert by_raw.loc["牛雨雨-325-400", "eye_subject_alias"] == "牛雨雨-325-400"
    assert by_raw.loc["牛雨雨-325-400", "alias_source"] == "suffix_note"
    assert by_raw.loc["张三", "participant_id"] == "张三"
    assert by_raw.loc["张三", "alias_source"] == ""

    participants_csv = tmp_path / "participants.csv"
    participants = build_participants_from_roots(eye_root, eeg_root, out_csv=participants_csv)
    assert set(participants["participant_id"]) == {"牛雨雨", "张三"}
    assert participants["exclude"].astype(str).str.lower().isin({"false", "0"}).all()


def test_manual_alias_overrides_suffix_note(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    _write_eye_csv(eye_root / "1-C0W15" / "raw_牛雨雨-325-400_260526150342_0617145623.csv")
    alias_csv = tmp_path / "alias.csv"
    alias_csv.write_text("eye_subject_id,participant_id\n牛雨雨-325-400,牛雨雨正式\n", encoding="utf-8")

    eye = apply_eye_aliases(scan_eye_raw(eye_root), alias_csv=alias_csv)
    assert eye.loc[0, "participant_id"] == "牛雨雨正式"
    assert eye.loc[0, "alias_source"] == "manual"


def test_questionnaire_middle_dot_names_match_short_raw_ids(tmp_path: Path) -> None:
    questionnaire = tmp_path / "questionnaire.xlsx"
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    _write_minimal_xlsx(
        questionnaire,
        [
            ["Q1.0_姓名：", "Q1.8_场景顺序编号", "Q2.1_S1. 这个空间整体上适合打乒乓球。_"],
            ["沙力木江·吐尔洪", 1, "5分"],
        ],
    )
    _write_eye_csv(eye_root / "1-C0W15" / "raw_沙力木江_260530201640_0617145623.csv")
    _write_eeg_pair(eeg_root, "沙力木江")

    participants = build_participants_from_roots(eye_root, eeg_root, questionnaire_xlsx=questionnaire)
    row = participants.set_index("participant_id").loc["沙力木江"]
    assert bool(row["has_eeg_raw"]) is True
    assert bool(row["has_eye_raw"]) is True
    assert bool(row["has_questionnaire"]) is True

    q_long = load_questionnaire_long_from_wjx(questionnaire, participants=["沙力木江"])
    assert q_long.loc[0, "participant_id"] == "沙力木江"
    assert q_long.loc[0, "questionnaire_subject_id"] == "沙力木江·吐尔洪"


def test_eeg_scene_summary_contract_validator() -> None:
    ok = pd.DataFrame({
        "subject_id": ["P01"],
        "scene_id": [1],
        "view_start_s": [1.0],
        "view_end_s": [3.0],
        "view_dur_s": [2.0],
        "O_theta": [1.2],
        "hf_ratio_20_40Hz": [0.1],
        "rms_mean_uV": [10.0],
        "peak_to_peak_uV": [50.0],
        "nan_fraction": [0.0],
        "flat_fraction": [0.0],
        "segment_valid_duration": [True],
    })
    result = validate_eeg_scene_summary_frame(ok)
    assert result["status"] == "pass"

    missing = validate_eeg_scene_summary_frame(pd.DataFrame({"participant_id": ["P01"], "view_dur_s": [2.0]}))
    assert missing["status"] == "error"
    assert any("scene_id" in e for e in missing["errors"])
    assert any("core EEG metric" in e for e in missing["errors"])

    warn = validate_eeg_scene_summary_frame(pd.DataFrame({"participant_id": ["P01"], "scene_id": [1], "O_alpha": [0.9]}))
    assert warn["status"] == "warning"
    assert warn["warnings"]


def test_scripts_do_not_reference_order_material_roots() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    checked = [script_dir / "preflight_raw_inputs.py", script_dir / "run_realdata_linktest.py", script_dir / "run_realdata_all.py"]
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


def test_linktest_dry_run_does_not_create_outdir(tmp_path: Path) -> None:
    outdir = tmp_path / "scratch"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_realdata_linktest.py"
    result = subprocess.run([sys.executable, str(script), "--dry-run", "--outdir", str(outdir)], capture_output=True, text=True, check=True)
    assert "Raw input directories are read-only" in result.stdout
    assert not outdir.exists()


def test_eeg_validator_cli_json(tmp_path: Path) -> None:
    eeg = tmp_path / "eeg_scene.csv"
    eeg.write_text("subject_id,scene_id,view_dur_s,O_theta\nP01,1,2,1.2\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_eeg_scene_summary.py"
    result = subprocess.run([sys.executable, str(script), str(eeg), "--json"], capture_output=True, text=True, check=True)
    assert '"status": "warning"' in result.stdout
    assert '"rows": 1' in result.stdout


def test_run_eeg_from_raw_dry_run_command(tmp_path: Path) -> None:
    eeg_root = tmp_path / "eeg raw"
    eeglab_root = tmp_path / "eeglab root"
    outdir = tmp_path / "eeg out"
    eeg_root.mkdir()
    eeglab_root.mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_eeg_from_raw.py"
    result = subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--eeg_root",
        str(eeg_root),
        "--eeglab_root",
        str(eeglab_root),
        "--outdir",
        str(outdir),
    ], capture_output=True, text=True, check=True)
    assert "eeglab('nogui')" in result.stdout
    assert "run_eeg_bandpower_from_set" in result.stdout
    assert not outdir.exists()


def test_run_realdata_all_dry_run_does_not_create_outputs(tmp_path: Path) -> None:
    outdir = tmp_path / "real_outputs"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_realdata_all.py"
    result = subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--outputs_root",
        str(outdir),
        "--skip-eeg-export",
        "--skip-models",
        "--skip-diagnostics",
        "--skip-reporting",
        "--skip-figures",
    ], capture_output=True, text=True, check=True)
    assert '"status": "dry_run"' in result.stdout
    assert '"raw_inputs_read_only": true' in result.stdout
    assert not outdir.exists()


def test_run_realdata_all_with_existing_eeg_scene_csv(tmp_path: Path) -> None:
    questionnaire = tmp_path / "questionnaire.xlsx"
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    outputs = tmp_path.parent / "realdata_all_outputs"
    eeg_scene = tmp_path / "eeg_scene.csv"
    _write_realdata_all_fixture(questionnaire, eye_root, eeg_root, eeg_scene)

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_realdata_all.py"
    result = subprocess.run([
        sys.executable,
        str(script),
        "--questionnaire_xlsx",
        str(questionnaire),
        "--eye_root",
        str(eye_root),
        "--eeg_root",
        str(eeg_root),
        "--eeg_scene_csv",
        str(eeg_scene),
        "--outputs_root",
        str(outputs),
        "--participants",
        "张三,李四",
        "--expected-scenes-per-subject",
        "2",
        "--skip-questionnaire-significance",
        "--skip-questionnaire-reliability",
        "--skip-models",
        "--skip-diagnostics",
        "--skip-reporting",
        "--skip-figures",
        "--cleanup-scratch",
    ], capture_output=True, text=True, check=True)
    assert '"status": "complete"' in result.stdout
    assert not (outputs / "_raw_intake").exists()
    summary = pd.read_json(outputs / "realdata_run_summary.json", typ="series")
    assert summary["participants"] == 2
    assert summary["fusion_run"] is True
    assert (outputs / "05_multimodal_fusion" / "analysis_master_long.csv").exists()
    assert (outputs / "04_eeg" / "eeg_trial_long.csv").exists()


def test_run_realdata_all_rejects_outputs_inside_raw_root(tmp_path: Path) -> None:
    eye_root = tmp_path / "eye"
    eeg_root = tmp_path / "eeg"
    eye_root.mkdir()
    eeg_root.mkdir()
    questionnaire = tmp_path / "questionnaire.xlsx"
    _write_minimal_xlsx(questionnaire, [["Q1.0_姓名："], ["张三"]])
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_realdata_all.py"
    result = subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--questionnaire_xlsx",
        str(questionnaire),
        "--eye_root",
        str(eye_root),
        "--eeg_root",
        str(eeg_root),
        "--outputs_root",
        str(eye_root / "bad_outputs"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "outputs_root must not be inside a raw input directory" in result.stderr


def test_run_realdata_all_allows_outputs_next_to_questionnaire_file(tmp_path: Path) -> None:
    raw_parent = tmp_path / "raw_parent"
    questionnaire = raw_parent / "questionnaire.xlsx"
    eye_root = raw_parent / "eye"
    eeg_root = raw_parent / "eeg"
    outputs = raw_parent / "analysis_outputs"
    eye_root.mkdir(parents=True)
    eeg_root.mkdir()
    _write_minimal_xlsx(questionnaire, [["Q1.0_姓名："], ["张三"]])

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_realdata_all.py"
    result = subprocess.run([
        sys.executable,
        str(script),
        "--dry-run",
        "--questionnaire_xlsx",
        str(questionnaire),
        "--eye_root",
        str(eye_root),
        "--eeg_root",
        str(eeg_root),
        "--outputs_root",
        str(outputs),
    ], capture_output=True, text=True, check=True)
    assert '"status": "dry_run"' in result.stdout
    assert not outputs.exists()


def _write_eye_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Recording Time Stamp[ms],Gaze Point X[px],Gaze Point Y[px],Fixation Index,Fixation Duration[ms]\n"
        "0,50,50,1,200\n"
        "1000,50,50,1,200\n"
        "2000,150,50,2,150\n"
        "3000,150,50,2,150\n",
        encoding="utf-8",
    )


def _write_eeg_pair(root: Path, subject: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{subject}.set").write_text("placeholder", encoding="utf-8")
    (root / f"{subject}.fdt").write_text("placeholder", encoding="utf-8")


def _write_realdata_all_fixture(questionnaire: Path, eye_root: Path, eeg_root: Path, eeg_scene: Path) -> None:
    headers = [
        "Q1.0_姓名：",
        "Q1.8_场景顺序编号",
        "Q2.1_S1. 这个空间整体上适合打乒乓球。_",
        "Q2.2_S2. 空间尺度适合。_",
        "Q2.3_S3. 空间边界清晰。_",
        "Q2.4_S4. 空间让我愿意活动。_",
        "Q2.5_S5.愉悦度评价_",
        "Q3.1_S1. 这个空间整体上适合打乒乓球。_",
        "Q3.2_S2. 空间尺度适合。_",
        "Q3.3_S3. 空间边界清晰。_",
        "Q3.4_S4. 空间让我愿意活动。_",
        "Q3.5_S5.愉悦度评价_",
        "Q16.1_1. 我在佩戴和使用 VR 设备时感到舒适。_",
        "Q16.2_2. VR 场景让我有临场感。_",
        "Q16.3_3. 我能自然地观察场景。_",
        "Q16.4_4. 我能投入其中。_",
        "Q16.5_5. 场景反应符合预期。_",
        "Q16.6_6. 总体体验较好。_",
    ]
    rows = [headers]
    for subject, base in [("张三", 5), ("李四", 4)]:
        rows.append([subject, 1, f"{base}分", "5分", "5分", "5分", "6分", f"{base + 1}分", "6分", "6分", "6分", "7分", "5分", "5分", "5分", "5分", "5分", "5分"])
        _write_eeg_pair(eeg_root, subject)
    _write_minimal_xlsx(questionnaire, rows)

    aoi = '{"aoi_classes":{"table":[{"points":[[0,0],[100,0],[100,100],[0,100]]}],"window":[{"points":[[100,0],[200,0],[200,100],[100,100]]}]}}'
    for folder in ["1-C0W15", "1-C0W45"]:
        (eye_root / folder).mkdir(parents=True, exist_ok=True)
        (eye_root / folder / f"{folder}.json").write_text(aoi, encoding="utf-8")
    for subject in ["张三", "李四"]:
        _write_eye_csv(eye_root / "1-C0W15" / f"raw_{subject}_260530201640_0617145623.csv")
        _write_eye_csv(eye_root / "1-C0W45" / f"raw_{subject}_260530201640_0617150451.csv")

    pd.DataFrame([
        {"subject_id": subject, "scene_id": scene_id, "view_start_s": 0.0, "view_end_s": 3.0, "view_dur_s": 3.0, "O_theta": value, "F_theta": value + 0.1, "O_alpha": value + 0.2, "hf_ratio_20_40Hz": 0.1, "rms_mean_uV": 10.0, "peak_to_peak_uV": 50.0, "nan_fraction": 0.0, "flat_fraction": 0.0, "segment_valid_duration": True}
        for subject, value in [("张三", 1.0), ("李四", 1.2)]
        for scene_id in [1, 2]
    ]).to_csv(eeg_scene, index=False, encoding="utf-8-sig")


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
