"""Repair Artifact Tool HYPERLINK cached labels while preserving formulas.

Excel can evaluate these supported formulas, but the artifact renderer stores an
error diagnostic as their cached value. Keep the formula, cache its literal
friendly label, and request native recalculation on open. All other cells remain
untouched, including numeric results and their caches.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
ET.register_namespace('', NS)
PATTERN = re.compile(r'^HYPERLINK\("((?:[^"]|"")*)","((?:[^"]|"")*)"\)$', re.IGNORECASE)


def repair(path):
    path = Path(path)
    pending = path.with_suffix('.hyperlink-cache.tmp')
    changed = 0
    try:
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(pending, 'w') as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename.startswith('xl/worksheets/sheet') and info.filename.endswith('.xml'):
                    tree = ET.fromstring(data)
                    edited = False
                    for cell in tree.iter('{' + NS + '}c'):
                        formula = cell.find('{' + NS + '}f')
                        match = PATTERN.fullmatch(formula.text or '') if formula is not None else None
                        if match:
                            value = cell.find('{' + NS + '}v')
                            if value is None: value = ET.SubElement(cell, '{' + NS + '}v')
                            value.text = match[2].replace('""', '"')
                            cell.set('t', 'str')
                            edited = True; changed += 1
                    if edited: data = ET.tostring(tree, encoding='utf-8', xml_declaration=True)
                elif info.filename == 'xl/workbook.xml':
                    tree = ET.fromstring(data)
                    calc = tree.find('{' + NS + '}calcPr')
                    if calc is None: calc = ET.SubElement(tree, '{' + NS + '}calcPr')
                    calc.set('calcMode', 'auto'); calc.set('fullCalcOnLoad', '1')
                    data = ET.tostring(tree, encoding='utf-8', xml_declaration=True)
                dst.writestr(info, data)
        os.replace(pending, path)
    finally:
        if pending.exists(): pending.unlink()
    return changed


if __name__ == '__main__':
    for path in sys.argv[1:]:
        print(str(path) + ': repaired ' + str(repair(path)) + ' hyperlink caches')
