import importlib.util
import json
from pathlib import Path
from urllib.parse import unquote
import zipfile
import xml.etree.ElementTree as ET

import pytest


def test_native_relative_links_preserve_cells_and_reject_missing_sources(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('links',root/'scripts/add_teacher_relative_links.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    target=tmp_path/'filesource_flat';target.mkdir();(target/'源表.csv').write_text('x',encoding='utf-8')
    payload=tmp_path/'tables.json'
    payload.write_text(json.dumps({'tables':[{'name':'Sources','rows':[['=HYPERLINK("filesource_flat/源表.csv","SRC1")']]}]}),encoding='utf-8')
    book=tmp_path/'test.xlsx'
    with zipfile.ZipFile(book,'w') as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{module.MAIN}" xmlns:r="{module.REL}"><sheets><sheet name="Sources" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',f'<Relationships xmlns="{module.PACKAGE}"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml',f'<worksheet xmlns="{module.MAIN}"><sheetData><row r="2"><c r="A2" t="inlineStr"><is><t>SRC1</t></is></c><c r="B2"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>')
    assert module.add_links(payload,book)==1
    with zipfile.ZipFile(book) as z:
        sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        assert sheet.find(f'.//{{{module.MAIN}}}v').text=='2'
        assert sheet.find(f'.//{{{module.MAIN}}}f').text=='1+1'
        relationship=ET.fromstring(z.read('xl/worksheets/_rels/sheet1.xml.rels'))[0]
        assert unquote(relationship.attrib['Target'])=='filesource_flat/源表.csv'
    payload.write_text(json.dumps({'tables':[{'name':'Sources','rows':[['=HYPERLINK("filesource_flat/missing.csv","SRC2")']]}]}),encoding='utf-8')
    with pytest.raises(ValueError,match='Missing'):module.add_links(payload,book)
