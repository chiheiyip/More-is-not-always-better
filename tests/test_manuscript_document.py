import hashlib
from zipfile import ZipFile

import pytest

from paper_analysis.teacher.manuscript_document import build, inspect


def manuscript(tmp_path, field=False, drawing=False):
    p = tmp_path/'original.docx'
    extra = '<w:r><w:fldChar/></w:r>' if field else '<w:r><w:drawing/></w:r>' if drawing else ''
    with ZipFile(p, 'w') as z:
        z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Original</w:t></w:r>'+extra+'</w:p><w:p><w:r><w:t>4. Discussion</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>42</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
        z.writestr('word/media/image1.png', b'unedited-image')
        z.writestr('word/_rels/document.xml.rels', b'unchanged relationships')
    return p


def test_targeted_edits_keep_original_images_relationships_and_insert_structure(tmp_path):
    p = manuscript(tmp_path); original = p.read_bytes()
    payload = {'source_sha256': hashlib.sha256(original).hexdigest(),
               'paragraph_edits': [{'index': 0, 'expected': 'Original', 'replacement': 'Updated'}],
               'cell_edits': [{'cell': [0, 0, 0], 'expected': '42', 'replacement': '41'}],
               'insertions': [{'before_index': 1, 'expected_anchor': '4. Discussion', 'paragraphs': [{'style_template_index': 0, 'text': '3.4 Joint analysis'}]}]}
    q = tmp_path/'updated.docx';changes = build(p, payload, q)
    assert p.read_bytes() == original
    assert inspect(q)['paragraphs'] == ['Updated', '3.4 Joint analysis', '4. Discussion']
    assert inspect(q)['tables'] == [[['41']]] and len(changes) == 3
    with ZipFile(p) as a, ZipFile(q) as b:
        assert a.read('word/media/image1.png') == b.read('word/media/image1.png')
        assert a.read('word/_rels/document.xml.rels') == b.read('word/_rels/document.xml.rels')


@pytest.mark.parametrize('field,drawing', [(True, False), (False, True)])
def test_edits_refuse_to_destroy_fields_or_media(tmp_path, field, drawing):
    p = manuscript(tmp_path, field, drawing)
    payload = {'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'paragraph_edits': [{'index': 0, 'expected': 'Original', 'replacement': 'Updated'}]}
    with pytest.raises(ValueError, match='remove'): build(p, payload, tmp_path/'new.docx')


def test_wrong_source_and_paragraph_preconditions_stop_build(tmp_path):
    p = manuscript(tmp_path)
    with pytest.raises(ValueError, match='hash'):
        build(p, {'source_sha256': 'wrong'}, tmp_path/'new.docx')
    with pytest.raises(ValueError, match='precondition'):
        build(p, {'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'paragraph_edits': [{'index': 0, 'expected': 'wrong', 'replacement': 'Updated'}]}, tmp_path/'new.docx')
