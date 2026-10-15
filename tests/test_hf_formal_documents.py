from paper_analysis.teacher.hf_qc_delivery import formal_document_sources


def test_formal_documents_refresh_current_manuscript_alias(tmp_path):
    revised = tmp_path / "manuscript_REVISED_NO_HF_QC.docx"
    revised.write_bytes(b"current verified manuscript")
    guide = tmp_path / "数据来源交接说明.docx"
    guide.write_bytes(b"current guide")
    (tmp_path / "other.csv").write_text("data")
    sources = formal_document_sources(tmp_path)
    assert sources == {revised.name: revised, guide.name: guide,
                       "论文全文正文_最新结果直接修改版.docx": revised}
