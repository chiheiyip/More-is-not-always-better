import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from paper_analysis.teacher.hf_qc_delivery import validate_numeric_claims
from paper_analysis.teacher.manuscript_document import build, inspect
from paper_analysis.teacher.state import StageBlockedError


def test_numeric_claims_require_exact_source_row_and_full_precision(tmp_path):
    source = tmp_path / "中文.csv"
    pd.DataFrame([{"window": 10, "outcome": "枕区θ", "estimate": .004123456789},
                  {"window": 0, "outcome": "枕区θ", "estimate": .003}]).to_csv(source, index=False, encoding="utf-8-sig")
    sources = {"theta": {"source_path": str(source), "Source ID": "SRC0001", "flat_name": "SRC0001.csv"}}
    claim = {"source": "theta", "filters": {"window": 10, "outcome": "枕区θ"}, "field": "estimate",
             "expected": .004123456789, "location": "3.3 paragraph 2"}
    result = validate_numeric_claims({"numeric_claims": [claim]}, sources)
    assert result.iloc[0]["核验"] == "一致"
    with pytest.raises(StageBlockedError, match="differs"):
        validate_numeric_claims({"numeric_claims": [dict(claim, expected=.0042)]}, sources)
    with pytest.raises(StageBlockedError, match="one source row"):
        validate_numeric_claims({"numeric_claims": [dict(claim, filters={"outcome": "枕区θ"})]}, sources)


def test_appendix_table_insertion_preserves_existing_table_and_media(tmp_path):
    original = tmp_path / "original.docx"
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    for text in ["Table B.2", "Note. Earlier result", "Data availability"]:
        xml += f'<w:p><w:r><w:t>{text}</w:t></w:r></w:p>'
    xml += '<w:tbl><w:tblGrid><w:gridCol w:w="9000"/></w:tblGrid>'
    for text in ["Header", "Original value"]:
        xml += f'<w:tr><w:tc><w:tcPr><w:tcW w:w="9000" w:type="dxa"/></w:tcPr><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc></w:tr>'
    xml += '</w:tbl></w:body></w:document>'
    with ZipFile(original, "w") as z:
        z.writestr("word/document.xml", xml)
        z.writestr("word/media/image.png", b"original figure")
    before = original.read_bytes()
    rows = [["Window", "RMS", "PTP"], ["0", "10.0001", "90.0002"], ["5", "9.9", "89.8"]]
    payload = {"source_sha256": hashlib.sha256(before).hexdigest(), "table_insertions": [{
        "before_index": 2, "expected_anchor": "Data availability", "table_template_index": 0,
        "caption_style_index": 0, "note_style_index": 1, "caption": "Table B.3 QC thresholds", "note": "Note. No-HF.", "rows": rows}]}
    target = tmp_path / "revised.docx"
    build(original, payload, target)
    result = inspect(target)
    assert original.read_bytes() == before
    assert result["tables"] == [rows, [["Header"], ["Original value"]]]
    assert result["paragraphs"] == ["Table B.2", "Note. Earlier result", "Table B.3 QC thresholds", "Note. No-HF.", "Data availability"]
    with ZipFile(target) as z: assert z.read("word/media/image.png") == b"original figure"


def test_empty_numerical_evidence_cannot_prepare_a_scientific_package():
    with pytest.raises(StageBlockedError, match="No source-backed"):
        validate_numeric_claims({"numeric_claims": []}, {})


@pytest.mark.parametrize("changed", ["artifact", "existing_delivery"])
def test_publication_rejects_stale_artifacts_and_preserves_user_edits(tmp_path, monkeypatch, changed):
    from paper_analysis.teacher import hf_qc_delivery as module
    monkeypatch.setattr(module, "check", lambda _: [])
    monkeypatch.setattr(module, "git_sha", lambda _: "committed")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "")
    work, delivery = tmp_path / "work", tmp_path / "teacher"
    stage = work / "staging"
    stage.mkdir(parents=True)
    delivery.mkdir()
    (delivery / "manual.txt").write_text("user work")
    report = stage / "论文数据分析结果报告.md"
    report.write_text("verified report")
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    config = {"run_root": str(tmp_path / "run"), "work_root": str(work), "delivery_root": str(delivery), "archive_path": str(tmp_path / "archive")}
    prepared = {"code_sha": "committed", "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "old_inventory": module.inventory(delivery), "artifacts": {report.name: digest}}
    (work / "preparation.json").write_text(json.dumps(prepared))
    (work / "artifact_verification.json").write_text(json.dumps({"files": {report.name: {"sha256": digest, "content_verified": True}}}))
    if changed == "artifact": report.write_text("changed")
    else: (delivery / "manual.txt").write_text("new user changes")
    current = (delivery / "manual.txt").read_bytes()
    with pytest.raises(StageBlockedError, match="stale|Teacher delivery changed"):
        module.publish(config, tmp_path)
    assert (delivery / "manual.txt").read_bytes() == current
    assert stage.exists()
