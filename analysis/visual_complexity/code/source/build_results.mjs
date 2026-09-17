import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require = createRequire(import.meta.url);
const artifactModule = process.env.ARTIFACT_TOOL_MODULE || '@oai/artifact-tool';
const {Workbook,SpreadsheetFile} = await import(pathToFileURL(require.resolve(artifactModule)).href);
const run=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(run,'results.json'),'utf8'));
if(data.scenes.length!==6||data.faces.length!==36) throw Error('Expected six scenes and 36 faces');
const wb=Workbook.create();
const pair=wb.worksheets.add('原始指标与差值');
const score=wb.worksheets.add('标准化与综合分');
const raw=wb.worksheets.add('逐面结果与说明');
const navy='#233E59', pale='#EEF3F8', ink='#18212B';
for(const s of [pair,score,raw]){
  s.showGridLines=false;
  s.getRange('A1:L60').format.font={name:'Arial',size:11,color:ink};
  s.getRange('A1:L60').format.verticalAlignment='center';
  s.getRange('A1:L60').format.rowHeight=23;
}
pair.tabColor=navy;score.tabColor='#537796';
function title(s,text,range){
 s.getRange('A2').values=[[text]];
 s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:navy};
 s.getRange(range).format.borders={bottom:{style:'thin',color:navy}};
 s.getRange(range).format.rowHeight=29;
}
function head(s,range,values){
 s.getRange(range).values=[values];
 s.getRange(range).format={fill:navy,font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:40,wrapText:true};
}
function tableBody(s,range,start,end,last){
 s.getRange(range).format.rowHeight=27;
 for(let r=start;r<=end;r++) if((r-start)%2===0)s.getRange(`A${r}:${last}${r}`).format.fill=pale;
 s.getRange(`A${end}:${last}${end}`).format.borders={bottom:{style:'thin',color:'#A4B3C1'}};
}
title(pair,'Scene complexity by window-to-wall ratio','A2:G2');
pair.getRange('A3').values=[['FC and DL are arithmetic means across six cubemap faces.']];
head(pair,'A5:G5',['WWR level','FC in C0','FC in C1','ΔFC (C1−C0)','DL in C0','DL in C1','ΔDL (C1−C0)']);
pair.getRange('A6:A8').values=[['WWR15'],['WWR45'],['WWR75']];
pair.getRange('A1:G12').format.columnWidth=18;
pair.getRange('D1:D12').format.columnWidth=20;
pair.getRange('G1:G12').format.columnWidth=20;
pair.getRange('B6:G8').setNumberFormat('0.000000');
tableBody(pair,'A6:G8',6,8,'G');
pair.getRange('A10').values=[['Δ = C1 − C0 within the same WWR level. C1 uses the LA scenes.']];
pair.getRange('A11').values=[['Image-based measures; differences are descriptive and are not statistical tests.']];
pair.getRange('A10:A11').format.font={name:'Arial',size:10,color:'#52616F'};

title(score,'Standardized complexity scores for six scenes','A2:H2');
score.getRange('A3').values=[['Standardized within the current six scenes using population standard deviations.']];
head(score,'A5:H5',['Scene','WWR (%)','Condition','FC','DL','z(FC)','z(DL)','Score']);
score.getRange('A1:A15').format.columnWidth=23;
score.getRange('B1:C15').format.columnWidth=15;
score.getRange('D1:H15').format.columnWidth=18;
score.getRange('D6:H11').setNumberFormat('0.000000');
score.getRange('B6:C11').format.horizontalAlignment='center';
tableBody(score,'A6:H11',6,11,'H');
score.getRange('A13').values=[['Score = 0.5 × z(FC) + 0.5 × z(DL). Population SD uses STDEV.P (ddof = 0).']];
score.getRange('A14').values=[['These scores are relative to this six-scene set; they are not the historical nine-scene scores.']];
score.getRange('A13:A14').format.font={name:'Arial',size:10,color:'#52616F'};

head(raw,'A1:F1',['Source image','WWR (%)','Condition','Cube face','FC','DL']);
raw.getRange('A2:F37').values=data.faces.map(r=>[r.file,r.WWR,r.condition,r.face,r.FC,r.DL]);
raw.getRange('A1:A37').format.columnWidth=30;
raw.getRange('B1:D37').format.columnWidth=14;
raw.getRange('E1:F37').format.columnWidth=18;
raw.getRange('E2:F37').setNumberFormat('0.000000');
raw.getRange('B2:D37').format.horizontalAlignment='center';
tableBody(raw,'A2:F37',2,37,'F');
raw.freezePanes.freezeRows(1);
raw.getRange('H1:H37').format.columnWidth=23;
raw.getRange('I1:I37').format.columnWidth=72;
raw.getRange('H2:I2').values=[['Method','Details']];
raw.getRange('H2:I2').format.font={bold:true,name:'Arial',size:11,color:navy};
const notes=[
 ['Input','Six 8192 × 4096 RGB panoramas; C1 uses LA.'],
 ['Cubemap','Six 512 × 512 faces; py360convert.e2c; bilinear interpolation.'],
 ['FC','visual_clutter.Vlc.getClutter_FC() with original default parameters.'],
 ['DL model','trained_model_inception_v3.h5, fusionlove/image-complexity.'],
 ['Execution',`${data.method.backend} backend; ${data.method.device}.`],
 ['DL preprocessing','Pillow RGB resize to 299 × 299; Inception V3 preprocess_input; NCHW.'],
 ['Aggregation','Arithmetic mean of all six faces per scene, separately for FC and DL.'],
 ['Standardization','Across all six scene means; population SD (ddof = 0).'],
 ['Composite weights','FC 0.5; DL 0.5. No tertile classification.'],
 ['Interpretation','Image-based scene descriptors, not measured cognitive load or participant ratings.'],
 ['FC source','https://github.com/kargaranamir/visual-clutter'],
 ['DL source','https://github.com/fusionlove/image-complexity'],
 ['Run',path.basename(run)],
 ...Object.entries(data.packages).map(([k,v])=>[k,v])
];
raw.getRange(`H3:I${notes.length+2}`).values=notes;
raw.getRange(`I3:I${notes.length+2}`).format.wrapText=true;
raw.getRange(`H3:I${notes.length+2}`).format.rowHeight=40;
raw.getRange(`H3:I${notes.length+2}`).format.verticalAlignment='top';
for(let i=0;i<6;i++){
 const r=i+6,s=data.scenes[i], first=2+i*6,last=first+5;
 if(!data.faces.slice(i*6,i*6+6).every(f=>f.file===s.file))throw Error('Face order mismatch');
 score.getRange(`A${r}:C${r}`).values=[[`${s.condition}–WWR${s.WWR}`,s.WWR,s.condition]];
 score.getRange(`D${r}:H${r}`).formulas=[[
  `=AVERAGE('逐面结果与说明'!E${first}:E${last})`,
  `=AVERAGE('逐面结果与说明'!F${first}:F${last})`,
  `=(D${r}-AVERAGE($D$6:$D$11))/STDEV.P($D$6:$D$11)`,
  `=(E${r}-AVERAGE($E$6:$E$11))/STDEV.P($E$6:$E$11)`,
  `=0.5*F${r}+0.5*G${r}`]];
}
for(let i=0;i<3;i++){
 const r=6+i,a=6+2*i,b=a+1;
 pair.getRange(`B${r}:G${r}`).formulas=[[
  `='标准化与综合分'!D${a}`,`='标准化与综合分'!D${b}`,`=C${r}-B${r}`,
  `='标准化与综合分'!E${a}`,`='标准化与综合分'!E${b}`,`=F${r}-E${r}`]];
}
const near=(a,b)=>{if(typeof a!=='number'||Math.abs(a-b)>1e-9)throw Error(`Value mismatch ${a} vs ${b}`);};
function check(){
 const values=score.getRange('D6:H11').values;
 values.forEach((row,i)=>row.forEach((v,j)=>near(v,data.scenes[i][['FC','DL','z_FC','z_DL','Score'][j]])));
 const pv=pair.getRange('B6:G8').values;
 pv.forEach((row,i)=>row.forEach((v,j)=>near(v,data.pairs[i][['FC_C0','FC_C1','delta_FC','DL_C0','DL_C1','delta_DL'][j]])));
}
wb.recalculate();check();
// Verify the dependency chain with a temporary raw-data perturbation, then restore.
const original=data.faces[0].FC;
raw.getRange('E2').values=[[original+0.06]];
wb.recalculate();near(score.getRange('D6').values[0][0],data.scenes[0].FC+0.01);
near(pair.getRange('D6').values[0][0],data.pairs[0].delta_FC-0.01);
raw.getRange('E2').values=[[original]];
wb.recalculate();check();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:30},maxChars:2500});
console.log(errors.ndjson);
const previewDir=path.join(run,'verification');await fs.mkdir(previewDir,{recursive:true});
for(const [s,range,name] of [[pair,'A1:G12','pairs'],[score,'A1:H15','scores'],[raw,'A1:I38','faces']]){
 const png=await wb.render({sheetName:s.name,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(previewDir,name+'.png'),new Uint8Array(await png.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
const filename=path.join(run,'六场景视觉复杂度.xlsx');await output.save(filename);
await fs.writeFile(path.join(previewDir,'formula_validation.json'),JSON.stringify({scene_values:score.getRange('D6:H11').values,pair_values:pair.getRange('B6:G8').values,independent_comparison:'passed, absolute tolerance 1e-9',input_perturbation:'passed and restored',errors:errors.ndjson},null,2));
console.log('EXCEL_COMPLETE:',filename);
