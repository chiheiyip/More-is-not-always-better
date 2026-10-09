import importlib.util
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location('hyperlink_caches', Path(__file__).parents[1]/'scripts/fix_hyperlink_caches.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_cached_label_repaired_without_changing_formula_numeric_cache_or_other_members(tmp_path):
    path = tmp_path/'index.xlsx'
    xml = '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1">
    <c r="A1" t="e"><f>HYPERLINK("filesource_flat/中文.csv","中文.csv")</f><v>HYPERLINK is not implemented</v></c>
    <c r="B1"><f>1/20</f><v>0.05</v></c></row></sheetData></worksheet>'''
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/worksheets/sheet1.xml', xml)
        z.writestr('xl/workbook.xml', '<workbook xmlns="'+module.NS+'"/>')
        z.writestr('unchanged.bin', b'existing-payload')
    assert module.repair(path) == 1
    with zipfile.ZipFile(path) as z:
        tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        cells = list(tree.iter('{'+module.NS+'}c'))
        assert cells[0].get('t') == 'str' and cells[0].find('{'+module.NS+'}v').text == '中文.csv'
        assert cells[0].find('{'+module.NS+'}f').text == 'HYPERLINK("filesource_flat/中文.csv","中文.csv")'
        assert cells[1].find('{'+module.NS+'}v').text == '0.05' and cells[1].find('{'+module.NS+'}f').text == '1/20'
        assert z.read('unchanged.bin') == b'existing-payload'
