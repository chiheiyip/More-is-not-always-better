"""Apply the requested Chinese/Latin fonts without rewriting document content."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.dom import minidom
from zipfile import ZipFile

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'


def _native_math(node):
    while node is not None:
        if node.namespaceURI == M:
            return True
        node = node.parentNode
    return False


def _child(parent, name):
    found = [n for n in parent.childNodes if n.nodeType == n.ELEMENT_NODE
             and n.namespaceURI == W and n.localName == name]
    if found:
        return found[0]
    node = parent.ownerDocument.createElementNS(W, 'w:'+name)
    if name == 'rPr' and parent.localName == 'style':
        later = [n for n in parent.childNodes if n.nodeType == n.ELEMENT_NODE
                 and n.namespaceURI == W and n.localName in {'tblPr','trPr','tcPr','tblStylePr'}]
        parent.insertBefore(node, later[0] if later else None)
    elif name == 'rFonts' and any(n.nodeType == n.ELEMENT_NODE and n.localName == 'rStyle' for n in parent.childNodes):
        style = next(n for n in parent.childNodes if n.nodeType == n.ELEMENT_NODE and n.localName == 'rStyle')
        parent.insertBefore(node, style.nextSibling)
    else:
        parent.insertBefore(node, parent.firstChild)
    return node


def apply_document_fonts(path, *, latin='Times New Roman', east_asia='宋体'):
    """Set editable text fonts in every Word part; retain native math/symbols."""
    path = Path(path)
    changed = []
    with NamedTemporaryFile(dir=path.parent, suffix='.docx', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(path) as source, ZipFile(temporary, 'w') as output:
            for info in source.infolist():
                value = source.read(info.filename)
                if info.filename.startswith('word/') and info.filename.endswith('.xml'):
                    document = minidom.parseString(value)
                    fonts = [n for n in document.getElementsByTagNameNS(W, 'rFonts') if not _native_math(n)]
                    if info.filename == 'word/styles.xml':
                        defaults = _child(document.documentElement, 'docDefaults')
                        fonts.append(_child(_child(_child(defaults, 'rPrDefault'), 'rPr'), 'rFonts'))
                        for style in document.getElementsByTagNameNS(W, 'style'):
                            fonts.append(_child(_child(style, 'rPr'), 'rFonts'))
                    for run in document.getElementsByTagNameNS(W, 'r'):
                        if not _native_math(run) and any(run.getElementsByTagNameNS(W, tag) for tag in ['t','instrText','fldChar']):
                            fonts.append(_child(_child(run, 'rPr'), 'rFonts'))
                    for font in fonts:
                        for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme','hint']:
                            font.removeAttributeNS(W, key) if font.hasAttributeNS(W, key) else None
                        for key, face in [('ascii',latin),('hAnsi',latin),('cs',latin),('eastAsia',east_asia)]:
                            font.setAttributeNS(W, 'w:'+key, face)
                    themed = []
                    if info.filename.startswith('word/theme/'):
                        for scheme in ['majorFont','minorFont']:
                            for parent in document.getElementsByTagNameNS(A, scheme):
                                for font in parent.childNodes:
                                    if font.nodeType != font.ELEMENT_NODE:
                                        continue
                                    if font.localName in ['latin','cs','ea'] or (font.localName=='font' and font.getAttribute('script') in ['Hans','Hant']):
                                        font.setAttribute('typeface', east_asia if font.localName=='ea' or font.localName=='font' else latin)
                                        themed.append(font)
                    if fonts or themed:
                        value = document.toxml(encoding='UTF-8')
                        changed.append(info.filename)
                output.writestr(info, value)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()
    return changed
