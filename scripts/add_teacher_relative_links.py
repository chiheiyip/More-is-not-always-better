"""Add native relative links to an Artifact Tool workbook, preserving its cells.

HYPERLINK evaluation is unavailable in the renderer. The Artifact Tool authors
all cells; this standard-library helper adds only the missing OOXML link feature.
"""
from pathlib import Path, PurePosixPath
import argparse
import json
import re
import tempfile
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zipfile

MAIN="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE="http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("",MAIN);ET.register_namespace("r",REL)


def add_links(payload_path,workbook):
    workbook=Path(workbook)
    payload=json.loads(Path(payload_path).read_text(encoding="utf-8"))
    with zipfile.ZipFile(workbook) as z:
        if z.testzip() is not None:raise ValueError("Corrupt input workbook")
        entries={name:z.read(name) for name in z.namelist()}
    book=ET.fromstring(entries["xl/workbook.xml"])
    book_rels=ET.fromstring(entries["xl/_rels/workbook.xml.rels"])
    targets={r.attrib["Id"]:r.attrib["Target"] for r in book_rels}
    sheets={s.attrib["name"]:targets[s.attrib[f"{{{REL}}}id"]] for s in book.find(f"{{{MAIN}}}sheets")}
    count=0
    for table in payload["tables"]:
        target=sheets[table["name"]]
        path=target.lstrip("/") if target.startswith("/") else "xl/"+target
        sheet=ET.fromstring(entries[path])
        relpath=str(PurePosixPath(path).parent/"_rels"/(PurePosixPath(path).name+".rels"))
        relationships=ET.fromstring(entries[relpath]) if relpath in entries else ET.Element(f"{{{PACKAGE}}}Relationships")
        links=sheet.find(f"{{{MAIN}}}hyperlinks")
        for row,values in enumerate(table["rows"],2):
            for col,value in enumerate(values,1):
                if not isinstance(value,str) or not value.startswith("=HYPERLINK("):continue
                match=re.fullmatch(r'=HYPERLINK\("([^"]+)","([^"]+)"\)',value)
                if not match:raise ValueError("Unsupported source hyperlink")
                relative,label=match.groups()
                parts=PurePosixPath(relative)
                if parts.is_absolute() or ".." in parts.parts or parts.parts[0]!="filesource_flat":
                    raise ValueError("Link must remain within the flat source folder")
                if not (workbook.parent/relative).is_file():raise ValueError("Missing relative hyperlink target")
                if links is None:
                    links=ET.Element(f"{{{MAIN}}}hyperlinks")
                    later={"printOptions","pageMargins","pageSetup","headerFooter","rowBreaks","colBreaks","drawing","legacyDrawing","tableParts","extLst"}
                    index=next((i for i,c in enumerate(sheet) if c.tag.split("}")[-1] in later),len(sheet))
                    sheet.insert(index,links)
                reference=f"{chr(64+col)}{row}"
                if any(h.attrib.get("ref")==reference for h in links):raise ValueError("Hyperlink already present")
                rid=f"rIdTeacherLink{row}_{col}"
                if any(r.attrib["Id"]==rid for r in relationships):raise ValueError("Relationship identity collision")
                ET.SubElement(relationships,f"{{{PACKAGE}}}Relationship",Id=rid,Type=REL+"/hyperlink",Target=quote(relative,safe="/_.-"),TargetMode="External")
                ET.SubElement(links,f"{{{MAIN}}}hyperlink",{"ref":reference,f"{{{REL}}}id":rid,"display":label})
                count+=1
        entries[path]=ET.tostring(sheet,encoding="utf-8",xml_declaration=True)
        entries[relpath]=ET.tostring(relationships,encoding="utf-8",xml_declaration=True)
    handle=tempfile.NamedTemporaryFile(dir=workbook.parent,suffix=".xlsx.tmp",delete=False);temporary=Path(handle.name);handle.close()
    try:
        with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as z:
            for name,content in entries.items():z.writestr(name,content)
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:raise ValueError("Workbook integrity failure")
        temporary.replace(workbook)
    finally:
        if temporary.exists():temporary.unlink()
    return count


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables",type=Path);parser.add_argument("workbook",type=Path)
    options=parser.parse_args()
    print("Native relative hyperlinks added:",add_links(options.tables,options.workbook))
