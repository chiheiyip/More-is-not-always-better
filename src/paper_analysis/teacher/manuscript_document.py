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


def clear_revision_formatting(document, colors=None):
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
        media = payload.get('media_replacements', {})
        for name, item in media.items():
            if not name.startswith('word/media/') or name not in archive.namelist(): raise ValueError('Unknown media target')
            if hashlib.sha256(archive.read(name)).hexdigest() != item['expected_sha256']: raise ValueError('Media precondition differs')
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']: raise ValueError('Replacement media changed')
        if payload.get('clear_revision_formatting', False):
            removed = clear_revision_formatting(document, payload.get('revision_colors'))
            changes.append({'kind': 'formatting', 'location': 'document', 'before': removed,
                            'after': 0, 'reason': 'Clean copy: remove explicit red text and yellow revision highlighting', 'sources': []})
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
