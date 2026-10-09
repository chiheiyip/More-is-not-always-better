import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const [payloadPath, output, previews] = process.argv.slice(2);
const payload=JSON.parse(await fs.readFile(payloadPath,'utf8'));
const wb=Workbook.create();
const col=n=>{let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;};
for(let t=0;t<payload.tables.length;t++){
  const table=payload.tables[t], sh=wb.worksheets.add(table.name), width=table.columns.length;
  sh.showGridLines=false;
  sh.getRange(`A1:${col(width)}1`).values=[table.columns];
  sh.getRange(`A1:${col(width)}1`).format={fill:'#17365D',font:{bold:true,color:'#FFFFFF',size:10},wrapText:true,verticalAlignment:'center'};
  sh.getRange(`A1:${col(width)}1`).format.rowHeight=32;
  if(table.rows.length){
    sh.getRange(`A2:${col(width)}${table.rows.length+1}`).values=table.rows;
    sh.getRange(`A2:${col(width)}${table.rows.length+1}`).format={font:{size:10},verticalAlignment:'center',wrapText:true};
    sh.getRange(`A2:${col(width)}${table.rows.length+1}`).format.rowHeight=38;
    const tab=sh.tables.add(`A1:${col(width)}${table.rows.length+1}`,true,`SourceTable${t}`);tab.style='TableStyleMedium2';
    const linkCol=table.columns.indexOf('平铺文件')>=0?table.columns.indexOf('平铺文件'):table.columns.indexOf('flat_name');
    if(linkCol>=0)for(let r=0;r<table.rows.length;r++){
      const target=String(table.rows[r][linkCol]).replaceAll('"','""');
      const prefix=String(payload.linkPrefix ?? 'filesource_flat/').replaceAll('"','""');
      sh.getCell(r+1,linkCol).formulas=[[`=HYPERLINK("${prefix}${target}","${target}")`]];
    }
  }
  for(let c=0;c<width;c++)sh.getRange(`${col(c+1)}1:${col(c+1)}${table.rows.length+1}`).format.columnWidth=
    /内容|说明|路径|source_path|flat_name|平铺文件/.test(table.columns[c])?65:/SHA256/.test(table.columns[c])?70:23;
  for(let r=0;r<table.rows.length;r++){
    const lines=Math.max(...table.rows[r].map((value,c)=>{
      const text=String(value??''),width=/内容|说明|路径|source_path|flat_name|平铺文件/.test(table.columns[c])?65:/SHA256/.test(table.columns[c])?70:23;
      const visualLength=[...text].reduce((n,ch)=>n+(ch.charCodeAt(0)>255?2:1),0);
      return Math.ceil(visualLength/Math.max(10,width-3));
    }));
    sh.getRange(`A${r+2}:${col(width)}${r+2}`).format.rowHeight=Math.max(38,lines*15+8);
  }
  sh.freezePanes.freezeRows(1);
}
wb.recalculate();
await fs.mkdir(previews,{recursive:true});
await fs.writeFile(path.join(previews,'xlsx_inspection.json'),JSON.stringify(await wb.inspect({kind:'sheet,table',maxChars:4500,tableMaxRows:3,tableMaxCols:5})));
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(output);
// Export diagnostics belong with previews, outside the portable delivery root.
try {
  await fs.rename(`${output}.inspect.ndjson`,path.join(previews,'xlsx_export_inspection.ndjson'));
} catch(error) {
  if(error.code!=='ENOENT')throw error;
}
for(const name of ['先看这里','样本与数据粒度','功率与统计复核']){
  const image=await wb.render({sheetName:name,range:'A1:D14',scale:1.5,format:'png'});
  await fs.writeFile(path.join(previews,`${name}.png`),new Uint8Array(await image.arrayBuffer()));
}
