// Author the complete comparison workbook using the bundled Artifact Tool.
// Python prepares typed JSON tables; this script never recalculates statistics.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [input, output, runtimeModules] = process.argv.slice(2);
if (!input || !output) throw new Error('usage: node export_eeg_denominator_workbook.mjs tables.json result.xlsx [runtime-node_modules]');
const require = createRequire(runtimeModules ? path.resolve(runtimeModules, '_artifact_entry.cjs') : import.meta.url);
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const data = JSON.parse(await fs.readFile(input,'utf8'));
const wb = Workbook.create();
for (const table of data.tables) {
  const sheet = wb.worksheets.add(table.name);
  sheet.showGridLines=false;
  const rows = [table.columns,...table.rows];
  const range = sheet.getRangeByIndexes(0,0,rows.length,table.columns.length);
  range.values=rows;
  range.format.font={name:'Arial',size:10,color:'#202C3A'};
  range.format.rowHeight=19;
  range.format.columnWidth=19;
  range.format.verticalAlignment='center';
  const header=sheet.getRangeByIndexes(0,0,1,table.columns.length);
  header.format={fill:'#233B59',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42,horizontalAlignment:'center',verticalAlignment:'center'};
  sheet.freezePanes.freezeRows(1);
  if (table.columns.length>8) sheet.freezePanes.freezeColumns(Math.min(3,table.columns.length));
  for (let c=0;c<table.columns.length;c++) {
    const name=table.columns[c];
    if (rows.length>1) {
      const cells=sheet.getRangeByIndexes(1,c,rows.length-1,1);
      if (table.numeric?.includes(name)) {
        cells.setNumberFormat(/p\.value|raw_p|_q|p_error/.test(name) ? '0.000E+00' : /estimate|delta|percent|Fstat|error|std\.error/.test(name) ? '0.000000' : '0.0000');
        cells.format.horizontalAlignment='right';
      } else cells.format.horizontalAlignment='left';
      if (/flip_|direction_changed/.test(name)) cells.conditionalFormats.add('containsText',{text:'TRUE',format:{fill:'#FFF0C2',font:{bold:true,color:'#875100'}}});
    }
    if (/outcome|family_id|manuscript_claim|term|method|formula/.test(name)) sheet.getRangeByIndexes(0,c,rows.length,1).format.columnWidth=32;
  }
  if (table.name==='说明') {
    sheet.getRangeByIndexes(0,1,rows.length,1).format.columnWidth=110;
    sheet.getRangeByIndexes(0,1,rows.length,1).format.wrapText=true;
    sheet.getRangeByIndexes(1,0,rows.length-1,2).format.rowHeight=42;
    sheet.freezePanes.unfreeze();sheet.tabColor='#61728A';
  }
}
wb.recalculate();
console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:2500})).ndjson);
const preview=await wb.render({sheetName:'主要结论',range:'A1:H14',scale:1.4,format:'png'});
await fs.writeFile(path.join(path.dirname(output),'workbook_preview.png'),new Uint8Array(await preview.arrayBuffer()));
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(`Exported ${data.tables.length} sheets: ${output}`);
