from zipfile import ZipFile
from xml.dom import minidom
import pytest

from paper_analysis.teacher.xlsx_hyperlinks import repair, S


def test_repairs_only_constant_link_caches_and_keeps_formulas_other_data(tmp_path):
    source = tmp_path/'source.xlsx'; target = tmp_path/'repaired.xlsx'
    xml = '<worksheet xmlns="'+S+'"><sheetData><row r="1"><c r="A1" t="e"><f>HYPERLINK("filesource_flat/中文.csv","SRC0001 中文")</f><v>unsupported</v></c><c r="B1"><v>0.012457</v></c><c r="C1" t="e"><f>OTHER_FUNCTION(1)</f><v>#NAME?</v></c><c r="D1" t="e"><f>HYPERLINK(A1,B1)</f><v>#NAME?</v></c></row></sheetData></worksheet>'
    with ZipFile(source, 'w') as z:
        z.writestr('xl/worksheets/sheet1.xml', xml)
        z.writestr('xl/styles.xml', b'unchanged styles')
    original = source.read_bytes()
    assert repair(source, target) == 1
    with ZipFile(target) as z:
        document = minidom.parseString(z.read('xl/worksheets/sheet1.xml'))
        cells = document.getElementsByTagNameNS(S, 'c')
        assert cells[0].getAttribute('t') == 'str'
        assert cells[0].getElementsByTagNameNS(S,'v')[0].firstChild.nodeValue == 'SRC0001 中文'
        assert cells[0].getElementsByTagNameNS(S,'f')[0].firstChild.nodeValue == 'HYPERLINK("filesource_flat/中文.csv","SRC0001 中文")'
        assert cells[1].getElementsByTagNameNS(S,'v')[0].firstChild.nodeValue == '0.012457'
        assert cells[2].getAttribute('t') == cells[3].getAttribute('t') == 'e'
        assert z.read('xl/styles.xml') == b'unchanged styles'
    assert source.read_bytes() == original
    with pytest.raises(ValueError):repair(source, source)
