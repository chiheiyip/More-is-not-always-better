"""Repair constant HYPERLINK display caches from unsupported authoring engines."""
from __future__ import annotations

from pathlib import Path
import re
from xml.dom import minidom
from zipfile import ZipFile

S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
LINK = re.compile(r'^=?HYPERLINK\("((?:[^\"]|\"\")*)","((?:[^\"]|\"\")*)"\)$', re.IGNORECASE)


def repair(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('Write a separate repaired workbook before replacement')
    replacements, count = {}, 0
    with ZipFile(source) as archive:
        for name in archive.namelist():
            if not re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name): continue
            document = minidom.parseString(archive.read(name)); changed = False
            for cell in document.getElementsByTagNameNS(S, 'c'):
                formulas = cell.getElementsByTagNameNS(S, 'f')
                if len(formulas) != 1: continue
                formula = ''.join(n.nodeValue for n in formulas[0].childNodes if n.nodeType == n.TEXT_NODE)
                match = LINK.fullmatch(formula)
                if not match: continue
                label = match.group(2).replace('""', '"')
                values = cell.getElementsByTagNameNS(S, 'v')
                if len(values) > 1: raise ValueError('Ambiguous formula cache')
                value = values[0] if values else document.createElementNS(S, 'v')
                if not values: cell.appendChild(value)
                for child in list(value.childNodes): value.removeChild(child)
                value.appendChild(document.createTextNode(label))
                cell.setAttribute('t', 'str'); changed = True; count += 1
            if changed: replacements[name] = document.toxml(encoding='UTF-8')
        with ZipFile(target, 'w') as output:
            for info in archive.infolist():
                output.writestr(info, replacements.get(info.filename, archive.read(info.filename)))
    return count
