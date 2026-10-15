"""Targeted DOCX edits with source-text preconditions and preserved media."""
from __future__ import annotations

import hashlib
from pathlib import Path
from xml.dom import minidom
from zipfile import ZipFile

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def children(node, name):
    return [n for n in node.childNodes if n.nodeType == n.ELEMENT_NODE and n.namespaceURI == W and n.localName == name]


def text(node):
    return ''.join(n.firstChild.nodeValue if n.firstChild else '' for n in node.getElementsByTagNameNS(W, 't'))


def citation_fields(node):
    """Collect complete, non-nested Zotero fields without changing their contents."""
    blocks, current = [], []
    for child in node.childNodes:
        if child.nodeType != child.ELEMENT_NODE:
            continue
        marks = child.getElementsByTagNameNS(W, 'fldChar')
        types = [mark.getAttributeNS(W, 'fldCharType') for mark in marks]
        if 'begin' in types:
            if current: raise ValueError('Nested Word field cannot be moved safely')
            current = [child]
        elif current:
            current.append(child)
        elif marks:
            raise ValueError('Unbalanced Word field')
        if 'end' in types:
            if not current: raise ValueError('Unbalanced Word field')
            instruction = ''.join(text_node.firstChild.nodeValue if text_node.firstChild else ''
                                  for element in current for text_node in element.getElementsByTagNameNS(W, 'instrText'))
            display = ''.join(text(element) for element in current)
            if 'ADDIN ZOTERO_ITEM CSL_CITATION' not in instruction or not display:
                raise ValueError('Only complete Zotero citation fields can be preserved in a text edit')
            blocks.append((display, [element.cloneNode(True) for element in current])); current = []
    if current: raise ValueError('Unbalanced Word field')
    return blocks


def set_paragraph(node, value, preserve_citations=False):
    if node.getElementsByTagNameNS(W, 'drawing') or node.getElementsByTagNameNS(W, 'object'):
        raise ValueError('Paragraph edit would remove embedded media')
    fields = []
    if node.getElementsByTagNameNS(W, 'fldSimple'):
        raise ValueError('Paragraph edit would remove a Word field')
    if node.getElementsByTagNameNS(W, 'fldChar'):
        if not preserve_citations: raise ValueError('Paragraph edit would remove a Word field')
        fields = citation_fields(node)
    fragments, offset = [], 0
    for display, elements in fields:
        position = value.find(display, offset)
        if position < 0: raise ValueError('Edited text must preserve every citation in its original order')
        fragments.append(value[offset:position]); fragments.append(elements); offset = position + len(display)
    fragments.append(value[offset:])
    runs = children(node, 'r')
    props = children(runs[0], 'rPr') if runs else []
    props = props[0].cloneNode(True) if props else None
    for child in list(node.childNodes):
        if child.nodeType == child.ELEMENT_NODE and child.namespaceURI == W and child.localName in {'pPr', 'bookmarkStart', 'bookmarkEnd'}:
            continue
        node.removeChild(child)
    doc = node.ownerDocument
    for fragment in fragments:
        if isinstance(fragment, list):
            for element in fragment: node.appendChild(element)
        else:
            run = doc.createElementNS(W, 'w:r')
            if props: run.appendChild(props.cloneNode(True))
            content = doc.createElementNS(W, 'w:t'); content.setAttribute('xml:space', 'preserve')
            content.appendChild(doc.createTextNode(fragment)); run.appendChild(content); node.appendChild(run)


def inspect(source):
    with ZipFile(source) as archive:
        document = minidom.parseString(archive.read('word/document.xml'))
        body = document.getElementsByTagNameNS(W, 'body')[0]
        paragraphs = children(body, 'p')
        tables = children(body, 'tbl')
        return {'paragraphs': [text(p) for p in paragraphs],
                'tables': [[[text(c) for c in children(r, 'tc')] for r in children(t, 'tr')] for t in tables],
                'media_sha256': {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('word/media/')}}


def clear_revision_formatting(document, colors=None, clear_run_shading=False):
    """Remove only the declared revision colors and yellow highlighting."""
    declared = {'FF0000', 'C00000'} if colors is None else {value.upper() for value in colors}
    if any(len(value) != 6 or any(char not in '0123456789ABCDEF' for char in value) for value in declared):
        raise ValueError('Revision colors must be explicit six-digit RGB values')
    removed = 0
    for name, values in [('color', declared), ('highlight', {'yellow'})]:
        for node in list(document.getElementsByTagNameNS(W, name)):
            value = node.getAttributeNS(W, 'val')
            if (value.upper() if name == 'color' else value) in values:
                node.parentNode.removeChild(node)
                removed += 1
    if clear_run_shading:
        for node in list(document.getElementsByTagNameNS(W, 'shd')):
            if (node.parentNode.localName == 'rPr' and node.getAttributeNS(W, 'val') == 'pct15'
                    and node.getAttributeNS(W, 'fill') == 'FFFFFF'):
                node.parentNode.removeChild(node)
                removed += 1
    return removed


def build(source, payload, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve(): raise ValueError('Original manuscript is protected')
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload['source_sha256']:
        raise ValueError('Manuscript source hash differs')
    with ZipFile(source) as archive:
        document = minidom.parseString(archive.read('word/document.xml'))
        body = document.getElementsByTagNameNS(W, 'body')[0]
        paragraphs = children(body, 'p'); tables = children(body, 'tbl')
        changes = []
        seen = set()
        for item in payload.get('paragraph_edits', []):
            key = item['index']
            if key in seen: raise ValueError('Duplicate paragraph edit')
            seen.add(key); node = paragraphs[key]
            before = text(node)
            if before != item['expected']: raise ValueError('Paragraph precondition differs')
            set_paragraph(node, item['replacement'], item.get('preserve_citations', False))
            changes.append({'kind': 'paragraph', 'location': key, 'before': before, 'after': item['replacement'],
                            'reason': item.get('reason', ''), 'sources': item.get('sources', [])})
        for item in payload.get('cell_edits', []):
            key = tuple(item['cell'])
            if key in seen: raise ValueError('Duplicate cell edit')
            seen.add(key)
            table, row, cell = key
            node = children(children(tables[table], 'tr')[row], 'tc')[cell]
            before = text(node)
            if before != item['expected']: raise ValueError('Table-cell precondition differs')
            ps = children(node, 'p')
            if len(ps) != 1: raise ValueError('Multi-paragraph table cell needs an explicit structural edit')
            set_paragraph(ps[0], item['replacement'])
            changes.append({'kind': 'cell', 'location': key, 'before': before, 'after': item['replacement'],
                            'reason': item.get('reason', ''), 'sources': item.get('sources', [])})
        for item in payload.get('insertions', []):
            anchor = paragraphs[item['before_index']]
            if text(anchor) != item['expected_anchor']: raise ValueError('Insertion anchor differs')
            for block in item['paragraphs']:
                node = paragraphs[block['style_template_index']].cloneNode(True)
                if node.getElementsByTagNameNS(W, 'drawing') or node.getElementsByTagNameNS(W, 'fldChar'):
                    raise ValueError('Invalid paragraph style template')
                for tag in ['bookmarkStart', 'bookmarkEnd']:
                    for mark in list(node.getElementsByTagNameNS(W, tag)): mark.parentNode.removeChild(mark)
                set_paragraph(node, block['text']); body.insertBefore(node, anchor)
                changes.append({'kind': 'insertion', 'location': item['before_index'], 'before': '',
                                'after': block['text'], 'reason': block.get('reason', ''), 'sources': block.get('sources', [])})
        for item in payload.get('table_insertions', []):
            anchor = paragraphs[item['before_index']]
            if text(anchor) != item['expected_anchor']:
                raise ValueError('Table insertion anchor differs')
            template = tables[item['table_template_index']]
            node = template.cloneNode(True)
            old_rows = children(node, 'tr')
            if len(old_rows) < 2:
                raise ValueError('Table insertion needs header and body row templates')
            for old in old_rows:
                node.removeChild(old)
            rows = item['rows']
            if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
                raise ValueError('Ragged or empty inserted table')
            width = int(item.get('width_twips', 9000)) // len(rows[0])
            grids = children(node, 'tblGrid')
            if grids:
                grid = grids[0]
                for old in list(grid.childNodes): grid.removeChild(old)
                for _ in rows[0]:
                    column = document.createElementNS(W, 'w:gridCol')
                    column.setAttribute('w:w', str(width)); grid.appendChild(column)
            for i, values in enumerate(rows):
                row = old_rows[0 if i == 0 else 1].cloneNode(True)
                cells = children(row, 'tc')
                for cell in cells: row.removeChild(cell)
                for j, value in enumerate(values):
                    cell = cells[min(j, len(cells)-1)].cloneNode(True)
                    ps = children(cell, 'p')
                    if not ps: raise ValueError('Missing inserted-cell paragraph template')
                    for extra in ps[1:]: cell.removeChild(extra)
                    set_paragraph(ps[0], str(value))
                    for prop in cell.getElementsByTagNameNS(W, 'tcW'):
                        prop.setAttribute('w:w', str(width)); prop.setAttribute('w:type', 'dxa')
                    row.appendChild(cell)
                node.appendChild(row)
            for key in ['caption', 'note']:
                if key == 'note': body.insertBefore(node, anchor)
                paragraph = paragraphs[item[key+'_style_index']].cloneNode(True)
                set_paragraph(paragraph, item[key]); body.insertBefore(paragraph, anchor)
            changes.append({'kind':'table_insertion','location':item['before_index'],'before':'',
                            'after':rows,'reason':item.get('reason',''),'sources':item.get('sources',[])})
        media = payload.get('media_replacements', {})
        for name, item in media.items():
            if not name.startswith('word/media/') or name not in archive.namelist(): raise ValueError('Unknown media target')
            if hashlib.sha256(archive.read(name)).hexdigest() != item['expected_sha256']: raise ValueError('Media precondition differs')
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']: raise ValueError('Replacement media changed')
        if payload.get('clear_revision_formatting', False):
            removed = clear_revision_formatting(document, payload.get('revision_colors'), payload.get('clear_revision_run_shading', False))
            changes.append({'kind': 'formatting', 'location': 'document', 'before': removed,
                            'after': 0, 'reason': 'Clean copy: remove explicit red text and yellow revision highlighting', 'sources': []})
        if payload.get('compact_trailing_field_paragraph', False):
            trailing = children(body, 'p')[-1]
            fields = trailing.getElementsByTagNameNS(W, 'fldChar')
            if text(trailing) or len(fields) != 1 or fields[0].getAttributeNS(W, 'fldCharType') != 'end':
                raise ValueError('Only an empty final field-closing paragraph may be compacted')
            pprops = children(trailing, 'pPr')
            props = pprops[0] if pprops else document.createElementNS(W, 'w:pPr')
            if not pprops: trailing.insertBefore(props, trailing.firstChild)
            for old in children(props, 'spacing'): props.removeChild(old)
            spacing = document.createElementNS(W, 'w:spacing')
            for key, value in [('before', '0'), ('after', '0'), ('line', '20'), ('lineRule', 'exact')]:
                spacing.setAttribute('w:'+key, value)
            props.appendChild(spacing)
            for run in children(trailing, 'r'):
                runs = children(run, 'rPr')
                rprops = runs[0] if runs else document.createElementNS(W, 'w:rPr')
                if not runs: run.insertBefore(rprops, run.firstChild)
                for tag in ['sz', 'szCs']:
                    for old in children(rprops, tag): rprops.removeChild(old)
                    size = document.createElementNS(W, 'w:'+tag); size.setAttribute('w:val', '2'); rprops.appendChild(size)
            changes.append({'kind':'formatting','location':'trailing_field','before':'inherited paragraph spacing',
                'after':'one-point line; field end preserved','reason':'Avoid an empty final page without deleting bibliography fields','sources':[]})
        target.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(target, 'w') as output:
            for info in archive.infolist():
                value = document.toxml(encoding='UTF-8') if info.filename == 'word/document.xml' else Path(media[info.filename]['path']).read_bytes() if info.filename in media else archive.read(info.filename)
                output.writestr(info, value)
    after = inspect(target); before = inspect(source)
    for name, digest in before['media_sha256'].items():
        if name not in media and after['media_sha256'][name] != digest: raise ValueError('Unedited media changed')
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload['source_sha256']: raise ValueError('Original source changed')
    return changes
