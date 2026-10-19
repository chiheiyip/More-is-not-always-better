from zipfile import ZipFile
from lxml import etree as E
from paper_analysis.teacher.document_fonts import apply_document_fonts

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
NS={'w':W,'m':M}


def test_font_policy_preserves_text_fields_math_media_and_covers_all_word_parts(tmp_path):
    path=tmp_path/'fonts.docx'
    xml=f'''<w:document xmlns:w="{W}" xmlns:m="{M}"><w:body>
    <w:p><w:r><w:rPr><w:rFonts w:ascii="Calibri" w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia"/><w:b/><w:sz w:val="22"/></w:rPr><w:t>中文 English 123</w:t></w:r>
    <w:r><w:instrText> ADDIN ZOTERO_ITEM CSL_CITATION unchanged </w:instrText></w:r>
    <w:r><w:sym w:font="Wingdings" w:char="F0A7"/></w:r>
    <m:oMath><m:r><w:rPr><w:rFonts w:ascii="Cambria Math"/></w:rPr><m:t>x</m:t></m:r></m:oMath></w:p>
    </w:body></w:document>'''
    style=f'<w:styles xmlns:w="{W}"><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:pPr/></w:style></w:styles>'
    header=f'<w:hdr xmlns:w="{W}"><w:p><w:r><w:t>Header 页眉</w:t></w:r></w:p></w:hdr>'
    numbering=f'<w:numbering xmlns:w="{W}"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:rPr><w:rFonts w:ascii="Wingdings" w:hAnsi="Wingdings"/></w:rPr></w:lvl></w:abstractNum></w:numbering>'
    with ZipFile(path,'w') as z:
        z.writestr('word/document.xml',xml);z.writestr('word/styles.xml',style);z.writestr('word/header1.xml',header)
        z.writestr('word/numbering.xml',numbering)
        z.writestr('word/media/image1.png',b'image unchanged');z.writestr('word/_rels/document.xml.rels',b'relations unchanged')
    apply_document_fonts(path)
    with ZipFile(path) as z:
        document=E.fromstring(z.read('word/document.xml'))
        assert document.xpath('//w:t/text()',namespaces=NS)==['中文 English 123']
        assert document.xpath('//w:instrText/text()',namespaces=NS)==[' ADDIN ZOTERO_ITEM CSL_CITATION unchanged ']
        assert document.xpath('//w:sym/@w:font',namespaces=NS)==['Wingdings']
        assert b'Wingdings' in z.read('word/numbering.xml')
        assert document.xpath('//m:oMath//w:rFonts/@w:ascii',namespaces=NS)==['Cambria Math']
        assert document.xpath('//w:b',namespaces=NS) and document.xpath('//w:sz/@w:val',namespaces=NS)==['22']
        for name in ['word/styles.xml','word/header1.xml']:
            root=E.fromstring(z.read(name));fonts=root.xpath('//w:rFonts',namespaces=NS);assert fonts
            for font in fonts:
                assert font.get('{'+W+'}ascii')=='Times New Roman' and font.get('{'+W+'}eastAsia')=='宋体'
                assert not any('Theme' in key for key in font.attrib)
        styles=E.fromstring(z.read('word/styles.xml'))
        assert [E.QName(x).localname for x in styles.find('w:style',NS)]==['name','pPr','rPr']
        assert z.read('word/media/image1.png')==b'image unchanged'
        assert z.read('word/_rels/document.xml.rels')==b'relations unchanged'
    first=path.read_bytes();apply_document_fonts(path);assert path.read_bytes()==first
