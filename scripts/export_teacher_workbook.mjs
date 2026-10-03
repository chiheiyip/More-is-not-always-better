// Author compact teacher tables using the bundled Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [input, output, runtimeModules] = process.argv.slice(2);
if (!input || !output || !runtimeModules) throw new Error('usage: node export_teacher_workbook.mjs tables.json result.xlsx runtime-node_modules');
const require = createRequire(path.resolve(runtimeModules, '_artifact_entry.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
for (const table of payload.tables) {
  const sheet = wb.worksheets.add(table.name);
  sheet.showGridLines = false;
  const values = [table.columns, ...table.rows];
  const area = sheet.getRangeByIndexes(0, 0, values.length, table.columns.length);
  area.values = values;
  area.format = {font: {name: 'Arial', size: 10, color: '#202C3A'}, rowHeight: 32, columnWidth: 18, verticalAlignment: 'center'};
  sheet.getRangeByIndexes(0, 0, 1, table.columns.length).format = {
    fill: '#233B59', font: {name: 'Arial', size: 10, bold: true, color: '#FFFFFF'},
    wrapText: true, rowHeight: 44, horizontalAlignment: 'center', verticalAlignment: 'center'
  };
  sheet.freezePanes.freezeRows(1);
  for (let c = 0; c < table.columns.length; c++) {
    const label = table.columns[c];
    const column = sheet.getRangeByIndexes(1, c, Math.max(1, table.rows.length), 1);
    const numeric = table.rows.some(row => typeof row[c] === 'number');
    column.format.horizontalAlignment = numeric ? 'right' : 'left';
    if (numeric) column.setNumberFormat(/p$|q$|β| F$/.test(label) ? '0.00000E+00' : '0');
    if (/实际口径|说明|判断/.test(label)) {
      sheet.getRangeByIndexes(0,c,values.length,1).format.columnWidth = 85;
      column.format.wrapText = true;
      column.format.rowHeight = 46;
    } else if (/SHA256/.test(label)) {
      sheet.getRangeByIndexes(0,c,values.length,1).format.columnWidth = 70;
    } else if (/EEG结果|正文结论|文件|来源/.test(label)) {
      sheet.getRangeByIndexes(0,c,values.length,1).format.columnWidth = 40;
      column.format.wrapText = true;
      column.format.rowHeight = 40;
    }
  }
}
wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:30}})).ndjson);
await fs.mkdir(path.dirname(output), {recursive:true});
for (const table of payload.tables) {
  const end = String.fromCharCode(64 + table.columns.length);
  const preview = await wb.render({sheetName:table.name,range:`A1:${end}${Math.min(8,table.rows.length+1)}`,scale:1.4,format:'png'});
  await fs.writeFile(path.join(path.dirname(output),`${path.basename(output,'.xlsx')}_${table.name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(`Exported ${payload.tables.length} sheets: ${output}`);
