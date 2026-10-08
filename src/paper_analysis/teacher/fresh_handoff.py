"""Report and transactional publication for a newly calculated eye/EEG run."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh import dump, git_sha, verify_inputs
from .state import file_sha256, StageBlockedError

SHEETS = ["先看这里", "段落来源映射", "关键数字核对", "文件总清单", "样本与数据粒度", "功率与统计复核", "差异与阴性结果", "未复制原始数据", "术语与字段说明"]


def typed(name, frame):
    return {"name": name, "columns": list(frame.columns),
            "rows": frame.astype(object).where(pd.notna(frame), None).values.tolist()}


def number(value):
    return "无法给出" if pd.isna(value) else f"{float(value):.8g}"


def collect_sources(run):
    records = []
    for p in sorted(run.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".json", ".txt", ".md"}:
            continue
        rel = p.relative_to(run)
        if len(rel.parts)==1 and p.name=="run_manifest.json":continue
        if any(v in rel.parts for v in ["samples", "psd_cache", "handoff_staging", "stage_seals", "artifact_previews"]):
            continue
        if p.name in {"handoff_tables.json", "source_manifest.json", "paragraph_source_map.csv", "artifact_verification.json", "publication.json", "archive_path_mapping.json"}:
            continue
        if p.name == "论文数据分析结果报告.md": continue
        records.append({"Source ID": f"SRC{len(records)+1:04d}", "source_path": str(p),
                        "flat_name": f"SRC{len(records)+1:04d}__{rel.parts[0]}__{p.name}",
                        "stage": rel.parts[0], "bytes": p.stat().st_size, "SHA256": file_sha256(p)})
    # Duplicate filenames inside the same stage are distinguished by Source ID.
    return records


def build_report(config, run, repo):
    comparison_dir=run/"11_historical_comparison";comparison_dir.mkdir(exist_ok=True)
    old_path=config.get("fresh",{}).get("historical_sensitivity_run")
    if old_path:
        from .eeg_denominator import compare_tables, manuscript_focus
        comparisons=[]
        for filename,kind in [("family_coefficients.csv","coefficient"),("family_factors.csv","factor")]:
            old=pd.read_csv(Path(old_path)/"A"/filename)
            current=pd.read_csv(run/"10_eeg_denominator_sensitivity/main_1_45"/filename)
            comparisons.append(compare_tables(old,current,"historical_1_45→fresh_1_45",kind))
        paired=pd.concat(comparisons,ignore_index=True)
        paired.to_csv(comparison_dir/"historical_to_fresh_EEG.csv",index=False,encoding="utf-8-sig")
        manuscript_focus(paired).to_csv(comparison_dir/"manuscript_historical_to_fresh.csv",index=False,encoding="utf-8-sig")
        dump(comparison_dir/"comparison_source.json",{"role":"read only after all fresh numerical stages; never a model input",
             "files":[{"path":str(Path(old_path)/"A"/f),"sha256":file_sha256(Path(old_path)/"A"/f)} for f in ["family_coefficients.csv","family_factors.csv"]]})
    old_eye=config.get("fresh",{}).get("historical_teacher_run")
    if old_eye:
        rows=[]
        for filename in ["09_familyA_primary_models.csv","10_familyB_primary_models.csv","12_CR2_robust_results.csv","11_WWR_posthoc_Holm.csv"]:
            a=pd.read_csv(Path(old_eye)/"02_eye_stage2"/filename)
            b=pd.read_csv(run/"02_eye_stage2"/filename)
            keys=["outcome", "contrast" if "contrast" in a else "term"]
            m=a.merge(b,on=keys,how="outer",suffixes=("_historical","_fresh"),indicator=True,validate="one_to_one")
            m["source_file"]=filename
            for name in ["estimate","p.value","p.value.likelihood","p.value.BH","p.value.Holm"]:
                if name+"_historical" in m and name+"_fresh" in m:
                    m["delta_"+name]=m[name+"_fresh"]-m[name+"_historical"]
                    if name.startswith("p."):
                        m["flip_"+name]=(m[name+"_historical"].lt(.05)!=m[name+"_fresh"].lt(.05)).mask(m[name+"_historical"].isna()|m[name+"_fresh"].isna())
            rows.append(m)
        pd.concat(rows,ignore_index=True).to_csv(comparison_dir/"historical_to_fresh_eye.csv",index=False,encoding="utf-8-sig")
    records = collect_sources(run)
    by_path = {r["source_path"]: r for r in records}
    lines = []; mapping = []; numbers = []
    def add(text="", sources=(), field="", selection="", numeric=False):
        lines.append(text)
        for p in sources:
            record = by_path[str(p)]
            mapping.append({"MD行号": len(lines), "内容": text, "Source ID": record["Source ID"],
                            "平铺文件": record["flat_name"], "字段": field, "筛选": selection,
                            "证据": "源表直接支持" if numeric else "方法或来源记录"})
    def numeric_row(label, value, source, field, selection=""):
        numbers.append({"项目": label, "数值": value, "Source ID": by_path[str(source)]["Source ID"], "字段": field, "筛选": selection})
    flow_path = run / "00_eeg_trial_qc/sample_flow.csv"
    flow = pd.read_csv(flow_path); common = flow.loc[flow.Stage.eq("four_window_common_QC")].iloc[0]
    eye_path = run / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx"
    eye = pd.read_excel(eye_path); selected = eye.loc[eye.IncludedPrimary.astype(str).str.lower().isin(["true", "1"])]
    sensitivity = run / "10_eeg_denominator_sensitivity"
    summary = json.loads((sensitivity / "summary.json").read_text(encoding="utf-8"))
    add("# 眼动与 EEG 全量重跑数据分析结果报告")
    add()
    add(f"本轮使用当前源文件重新计算，代码版本 `{git_sha(repo)}`。EEG 最终四窗口共同样本为 {int(common.Participants)} 人、{int(common.Trials)} 试次；眼动 60% 主分析为 {selected.Participant.nunique()} 人、{len(selected)} 试次。",
        [flow_path, eye_path], "Participants/Trials/IncludedPrimary", "four_window_common_QC; eye IncludedPrimary=true", True)
    add("本轮眼动、EEG及相关交集／同步结果替代此前对应结果。问卷独立分析未运行，其旧结果不在本轮更新范围。论文 Word 未修改，后文列出需要更新的结果及结论。")
    add()
    add("## 当前源文件和方法")
    add("侯集瀚、李泽豪的 SET/FDT 使用已与 8 月 18 日更新后记录核对一致的版本，运行前后验证 SHA256；所有 EEG 来源、原始眼动 CSV、AOI、底图、试次映射和协变量文件登记哈希。",
        [run / "source_hashes_before.json", run / "source_hashes_after.json"], "path/sha256")
    add("EEG从已有预处理波形开始，不重复滤波或人工ICA操作。保留 HF、RMS、峰峰值及原有QC规则，阈值为 median + 3.5 × MAD，坏试次比例 >30% 时整人排除。阈值由本轮数据重新计算。",
        [run / "00_eeg_trial_qc/QC_policy.json", run / "00_eeg_trial_qc/eeg_onset_sensitivity_qc_thresholds.csv"], "robust_metrics/threshold")
    add("0／5／10／15 s 是同等地位的平行窗口，各自计算QC后取共同试次。ROI先时域平均，再Welch PSD；θ 4–7、α 8–12、β 13–30 Hz。1–45 Hz保留为主分母，1–40 Hz为同一PSD和分子的敏感性分析，QC不随分母改变。",
        [sensitivity / "analysis_contract.json", sensitivity / "paired_psd_integrals.csv"], "PSD/factor_levels/total_1_45/total_1_40")
    add("Q1.4四档的前两档为Low、后两档为High，未用Q1.5代替；重新从原问卷推导并与参与者信息核对。",
        [run / "00_teacher_inputs/Q1_4_group_check.csv"], "Q1.4Original/ExerciseFrequency")
    add("眼动保留原fixation、AOI、60%主阈值及50%／70%敏感性、模型和校正方法；眼动主样本不因缺少EEG而排除。AOI人工标注、投影规则或来源材料的已知限制继续保留，不把全量计算等同于消除来源缺口。")
    add()
    add("## EEG 样本与排除")
    add("|阶段|人数|试次数|"); add("|---|---:|---:|")
    for _, r in flow.iterrows():
        add(f"|{r.Stage}|{int(r.Participants)}|{int(r.Trials)}|", [flow_path], "Stage/Participants/Trials", f"Stage={r.Stage}", True)
        numeric_row(r.Stage + "人数", int(r.Participants), flow_path, "Participants", f"Stage={r.Stage}")
        numeric_row(r.Stage + "试次", int(r.Trials), flow_path, "Trials", f"Stage={r.Stage}")
    data = pd.read_csv(run / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv")
    for person in sorted({Path(name).stem for name in config["fresh"].get("locked_source_hashes",{})}):
        d = data.loc[data.participant_id.eq(person) & data.onset_trim_s.eq(0)]
        retained = int(d.CommonQCIncluded.sum())
        add(f"{person}：本轮当前文件已进入导出和QC；四窗口共同保留 {retained} 个试次。{('进入主要模型。' if retained else '未进入主要模型，不能因文件已纠正而免除QC。')}",
            [run / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv"], "CommonQCIncluded/eeg_qc_reasons", f"participant_id={person}; onset_trim_s=0", True)
    add("完整试次和整人排除原因见本轮QC表；HF的全通道均值在平均参考后接近零这一已知局限仍需在解释中保留。本轮按用户确认维持原方法，未调整阈值追求指定人数。")
    add()
    add("## EEG 主模型与正文结论")
    add("主模型保留 WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender + Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)。CR2按参与者聚类，factor-level为等权边际CR2/HTZ重新实现，历史原脚本来源缺口仍在；上一场景保留 PreviousWWR + PreviousComplexity 的加性结构。",
        [sensitivity / "analysis_contract.json"])
    coefficients = pd.read_csv(sensitivity / "main_1_45/family_coefficients.csv")
    factors = pd.read_csv(sensitivity / "main_1_45/family_factors.csv")
    focus = coefficients.loc[(coefficients.family_id.eq("condition_core_relative") & coefficients.outcome.eq("F_theta_relative") & coefficients.term.eq("WWRWWR45:ComplexityC1")) |
        (coefficients.family_id.eq("temporal_relative") & coefficients.outcome.str.contains("alpha|theta") & coefficients.term.isin(["Block", "PositionWithinBlockCentered"])) |
        coefficients.family_id.eq("previous_core_relative")]
    add("|窗口s|结局|效应|β|原始p|联合q|"); add("|---:|---|---|---:|---:|---:|")
    for _, r in focus.iterrows():
        add(f"|{int(r.onset_trim_s)}|{r.outcome}|{r.term}|{number(r.estimate)}|{number(r['p.value'])}|{number(r.joint_q)}|",
            [sensitivity / "main_1_45/family_coefficients.csv"], "estimate/p.value/joint_q", f"window={r.onset_trim_s};outcome={r.outcome};term={r.term};family={r.family_id}", True)
    target = factors.loc[(factors.family_id.eq("factor_core_relative") & factors.outcome.eq("O_theta_relative") & factors.term.eq("Complexity")) |
                        (factors.family_id.eq("factor_expanded_relative") & factors.outcome.eq("P_beta_relative") & factors.term.eq("WWR"))]
    add("|窗口s|结局|总体效应|F|分子df|分母df|原始p|联合q|"); add("|---:|---|---|---:|---:|---:|---:|---:|")
    for _, r in target.iterrows():
        add(f"|{int(r.onset_trim_s)}|{r.outcome}|{r.term}|{number(r.Fstat)}|{number(r.df_num)}|{number(r.df_denom)}|{number(r['p.value'])}|{number(r.joint_q)}|",
            [sensitivity / "main_1_45/family_factors.csv"], "Fstat/df_num/df_denom/p.value/joint_q", f"window={r.onset_trim_s};outcome={r.outcome};term={r.term};family={r.family_id}", True)
    add("上述表格用于更新表6、枕区θ复杂度、alpha／theta时序、顶区β WWR和上一场景结果。显著性只依据各自完整检验族的联合q；无效推断标为无法给出，不等同于不显著。不同窗口不能事后选一个显著窗口代替平行分析。")
    if old_path:
        p=comparison_dir/"manuscript_historical_to_fresh.csv";changes=pd.read_csv(p)
        add("### 历史正文结果与当前全量重跑的差异")
        add("历史对照只在全部新计算完成后读取，用于判断论文结果变化；该比较同时含源文件和QC样本变化，不能解释为纯分母效应。",[comparison_dir/"comparison_source.json"])
        add("|正文项目|联合q翻转数|方向变化数|判断|");add("|---|---:|---:|---|")
        for claim,g in changes.groupby("manuscript_claim",sort=False):
            flips=int(g.flip_joint_q.eq(True).sum());directions=int(g.direction_changed.eq(True).sum())
            verdict="存在显著性变化，需更新正文" if flips else "方向变化需结合系数大小与p/q解释" if directions else "方向与联合显著性未改变，数值需更新"
            add(f"|{claim}|{flips}|{directions}|{verdict}|",[p],"manuscript_claim/flip_joint_q/direction_changed",f"manuscript_claim={claim}",True)
    add()
    add("## 1–40 Hz 分母敏感性")
    comparison_path = sensitivity / "model_comparisons.csv"
    comparisons = pd.read_csv(comparison_path)
    changed = comparisons.loc[comparisons.flip_joint_q.eq(True) | comparisons.direction_changed.eq(True)]
    add(f"同一批新PSD上，预定检验族记录到 {int(comparisons.flip_joint_q.eq(True).sum())} 项联合q显著性翻转。方向变化及近零变号完整保留在配对表，不能一概写成所有结论不变。",
        [comparison_path], "flip_joint_q/direction_changed", numeric=True)
    add("|窗口s|结局|模型／效应|1–45 β或F|1–40 β或F|1–45 q|1–40 q|"); add("|---:|---|---|---:|---:|---:|---:|")
    for _, r in changed.iterrows():
        a = r.estimate_from if pd.notna(r.estimate_from) else r.get("Fstat_from", np.nan)
        b = r.estimate_to if pd.notna(r.estimate_to) else r.get("Fstat_to", np.nan)
        add(f"|{int(r.onset_trim_s)}|{r.outcome}|{r.model}／{r.term}|{number(a)}|{number(b)}|{number(r.joint_q_from)}|{number(r.joint_q_to)}|",
            [comparison_path], "estimate/Fstat/joint_q", f"window={r.onset_trim_s};outcome={r.outcome};model={r.model};term={r.term};family={r.family_id}", True)
    powers_path = sensitivity / "paired_psd_integrals.csv"; powers = pd.read_csv(powers_path)
    add("|窗口s|40–45占1–45总功率均值%|实际分母减少均值%|"); add("|---:|---:|---:|")
    for trim, g in powers.groupby("onset_trim_s"):
        add(f"|{int(trim)}|{g.percent_40_45.mean():.8g}|{g.denominator_reduction_percent.mean():.8g}|", [powers_path], "percent_40_45/denominator_reduction_percent", f"onset_trim_s={trim};all ROIs", True)
    add("独立40–45 Hz积分与两分母之差因离散频率边界而不同，分别报告；未统一按平均百分比缩放试次。")
    add()
    add("## 眼动与同步结果")
    numeric_row("眼动主分析人数", int(selected.Participant.nunique()), eye_path, "Participant", "IncludedPrimary=true")
    numeric_row("眼动主分析试次", len(selected), eye_path, "IncludedPrimary", "IncludedPrimary=true")
    eye_results = []
    for filename in ["09_familyA_primary_models.csv", "10_familyB_primary_models.csv"]:
        p = run / "02_eye_stage2" / filename
        t = pd.read_csv(p); eye_results.append(t)
        add(f"眼动{filename[:2]}结果同时保留主似然模型与CR2 companion，估计、区间、原始p及BH见源表。", [p])
        add("|结局|效应|估计|主模型p|CR2 BH|"); add("|---|---|---:|---:|---:|")
        for _, r in t.loc[~t.term.eq("(Intercept)")].iterrows():
            add(f"|{r.outcome}|{r.term}|{number(r.get('estimate',np.nan))}|{number(r.get('p.value.likelihood',r.get('p.value',np.nan)))}|{number(r.get('p.value.BH',np.nan))}|",
                [p], "estimate/p.value.likelihood/p.value.BH", f"outcome={r.outcome};term={r.term}", True)
    add("眼动主要模型、50%／70%敏感性、Block1、LOPO、AOI边界、补充模型和诊断均在本轮重新运行。结构性缺失、未访问TTFF及失败模型保留原规则；WWR配对的Holm与CR2 BH分别解释。")
    if old_eye:
        p=comparison_dir/"historical_to_fresh_eye.csv";old_to_new=pd.read_csv(p)
        for field in [c for c in old_to_new if c.startswith("flip_")]:
            add(f"眼动历史对照 {field.removeprefix('flip_')}：跨越0.05的配对记录 {int(old_to_new[field].eq(True).sum())} 项；完整配对及未配对记录保留。",[p],field,numeric=True)
    sync_path = run / "08_synchronized_crossmodal/synchronized_crossmodal_sample_and_source.xlsx"
    sync = pd.read_excel(sync_path).iloc[0]
    add(f"重新构建绝对时钟同步并取眼动60%、EEG共同QC和时钟QC的精确交集：{int(sync.Participants)} 人、{int(sync.Trials)} 试次、{int(sync.TimeBins)} 个四窗口time-bin行。缺少可验证时钟的参与者没有补造时间映射。",
        [sync_path], "Participants/Trials/TimeBins", numeric=True)
    add()
    add("## 复核与论文更新")
    add("Python逐一独立积分完整PSD，R独立读取原问卷XLSX、四窗口模型CSV和MAT PSD，并重算完整BH。样本、分组、缺失位置、键和功率均核对；两种分母的absolute结果仅在本轮输入完全相同时复用。",
        [sensitivity / "independent_verification/summary.json", sensitivity / "exporter_PSD_regression.csv"])
    add("5000次参与者聚类bootstrap使用本轮输入重新运行，失败次数与随机种子保留在结果和运行记录。没有把历史bootstrap当作本轮计算。")
    add("论文需要用本轮样本与统计源表更新2.2样本描述、Results 3.2／3.3、表6及相关补充表。分析方法保持不变；若方法文字与实际保留的代码公式不一致，应据实澄清。HF局限、factor历史脚本缺口和AOI来源限制继续记录。")
    report = run / "论文数据分析结果报告.md"; report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    pd.DataFrame(mapping).to_csv(run / "paragraph_source_map.csv", index=False, encoding="utf-8-sig")
    dump(run / "source_manifest.json", records)
    samples = pd.concat([flow, pd.DataFrame([{"Stage":"eye_60pct_main","Participants":selected.Participant.nunique(),"Trials":len(selected)},
                                          {"Stage":"clock_eye_eeg_common","Participants":sync.Participants,"Trials":sync.Trials}])], ignore_index=True)
    omitted = [{"路径":r["path"],"SHA256":r["sha256"],"字节":r["bytes"],"原因":"大型原始波形／眼动采集不重复打包；来源哈希可追溯"}
               for r in json.loads((run / "source_hashes_before.json").read_text(encoding="utf-8")) if Path(r["path"]).suffix.lower() in {".set",".fdt",".easy",".csv"}]
    overview = pd.DataFrame([{"项目":"本轮范围","说明":"眼动、EEG、精确交集和同步；问卷独立模型未运行"},
        {"项目":"代码SHA","说明":git_sha(repo)},{"项目":"阅读顺序","说明":"MD → 段落来源映射 → Source ID → filesource_flat"},
        {"项目":"主分析","说明":"EEG 1–45 Hz；1–40 Hz为敏感性；四窗口平行"},
        {"项目":"旧结果","说明":"本轮替代旧眼动／EEG及相关交集同步结果；问卷不在替代范围"}])
    glossary = pd.DataFrame([{"术语":a,"说明":b} for a,b in [
        ("HF","原QC指标保留，平均参考后全通道均值接近零的局限保留"),("Q1.4","前两档Low、后两档High"),
        ("CR2","按参与者聚类；单系数Satterthwaite"),("HTZ","等权边际总体F检验"),("BH","完整预定族；窗口内与四窗口联合分别计算"),
        ("Holm","WWR配对比较的族内校正"),("NA","缺失／无效不等于零或不显著"),("PreviousScene","同Block前序；加性WWR＋Complexity")]])
    diagnostics = pd.read_csv(sensitivity / "main_1_45/diagnostics.csv")
    issues = diagnostics.loc[~diagnostics.inference_valid.eq(True)]
    tables = [typed(SHEETS[0],overview),typed(SHEETS[1],pd.DataFrame(mapping)),typed(SHEETS[2],pd.DataFrame(numbers)),
              typed(SHEETS[3],pd.DataFrame(records)),typed(SHEETS[4],samples),
              typed(SHEETS[5],pd.read_csv(sensitivity / "exporter_PSD_regression.csv")),
              typed(SHEETS[6],pd.concat([changed,issues],ignore_index=True)),typed(SHEETS[7],pd.DataFrame(omitted)),typed(SHEETS[8],glossary)]
    dump(run / "handoff_tables.json", {"tables":tables})
    guide = ["眼动与 EEG 全量重跑交接说明", f"本轮EEG为{int(common.Participants)}人／{int(common.Trials)}共同试次，眼动为{selected.Participant.nunique()}人／{len(selected)}主分析试次。当前源数据重新计算，方法保持不变。",
        "先阅读论文数据分析结果报告.md，再用Excel段落来源映射的Source ID打开filesource_flat中的实际数据文件。解压后保留文件夹相对位置。",
        "本轮结果替代此前眼动、EEG和相关交集同步结果。问卷独立分析暂未处理，论文Word未直接修改。",
        "四窗口平行，1–45 Hz为主分母，1–40 Hz为敏感性。显著性变化和近零变号应按MD及完整配对表解释；不能概括为所有结论不变。",
        "主似然模型、CR2 companion、factor HTZ、BH和Holm使用不同检验口径，请勿拼接估计和其他模型的p值。失败及无效推断与不显著分别保存。",
        "HF原QC规则保留，其局限继续记录。factor-level历史原脚本缺口、previous-scene加性公式及AOI来源限制见报告。大型波形和PSD留在正式结果位置，索引登记哈希。",
        "个体级源表按此前约定保留参与者标识。论文需按本轮源表更新人数、试次数、Results 3.2／3.3及相关表格。"]
    dump(run / "handoff_guide.json", guide)
    stage = run / "handoff_staging"; stage.mkdir(exist_ok=True)
    flat = stage / "filesource_flat"; flat.mkdir(exist_ok=True)
    for r in records:
        shutil.copyfile(r["source_path"], flat / r["flat_name"])
    shutil.copyfile(report, stage / report.name)
    dump(stage / "filesource_flat/source_manifest.json", records)
    dump(run / "fresh_summary.json", {"status":"complete","scope":"eye-eeg","git_commit":git_sha(repo),"run_root":str(run),
        "eeg":{"participants":int(common.Participants),"trials":int(common.Trials)},
        "eye":{"participants":int(selected.Participant.nunique()),"trials":len(selected)},
        "synchronized":{"participants":int(sync.Participants),"trials":int(sync.Trials),"timebins":int(sync.TimeBins)},
        "questionnaire":"not_rerun","denominator_sensitivity":summary})


def make_artifacts(config, run, repo):
    """Use the bundled Artifact Tool for the nine-sheet workbook."""
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml.ns import qn
    stage = run / "handoff_staging"
    guide = json.loads((run / "handoff_guide.json").read_text(encoding="utf-8"))
    doc = Document(); section = doc.sections[0]
    section.page_width=Inches(8.5); section.page_height=Inches(11)
    section.top_margin=section.bottom_margin=Inches(.8)
    section.left_margin=section.right_margin=Inches(.85)
    for style_name in ["Normal","Title","Heading 1"]:
        style=doc.styles[style_name]; style.font.name="Microsoft YaHei"; style.font.color.rgb=RGBColor(0,0,0)
        style.element.rPr.rFonts.set(qn("w:eastAsia"),"Microsoft YaHei")
    doc.styles["Normal"].font.size=Pt(11); doc.styles["Normal"].paragraph_format.space_after=Pt(8)
    doc.add_paragraph(guide[0],"Title")
    for text in guide[1:]: doc.add_paragraph(text)
    doc.core_properties.title=guide[0]; doc.save(stage / "数据来源交接说明.docx")
    work = repo / ".codex_tmp"; work.mkdir(exist_ok=True)
    script = work / "fresh_index.mjs"; shutil.copyfile(repo / "scripts/build_fresh_handoff_index.mjs", script)
    previews=run / "artifact_previews"; previews.mkdir(exist_ok=True)
    subprocess.run([config["fresh"]["node"],str(script),str(run / "handoff_tables.json"),
                    str(stage / "数据来源交接索引.xlsx"),str(previews)],cwd=repo,check=True)


def publish(config, run, outputs, repo):
    from .request_handoff import archive_publish
    if subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip():
        raise StageBlockedError("Publish from committed clean code")
    manifest=json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete": raise StageBlockedError("Run incomplete")
    verify_inputs(json.loads((run / "source_hashes_before.json").read_text(encoding="utf-8")))
    stage=run / "handoff_staging"
    qa=json.loads((run / "artifact_verification.json").read_text(encoding="utf-8"))
    for name in ["数据来源交接索引.xlsx","数据来源交接说明.docx"]:
        if not qa["files"][name]["visually_reviewed"] or qa["files"][name]["sha256"]!=file_sha256(stage/name):
            raise StageBlockedError("Artifact review missing/stale")
    records=json.loads((run / "source_manifest.json").read_text(encoding="utf-8"))
    for r in records:
        if file_sha256(r["source_path"])!=r["SHA256"] or file_sha256(stage/"filesource_flat"/r["flat_name"])!=r["SHA256"]:
            raise StageBlockedError("Copied source changed")
    if file_sha256(run/"论文数据分析结果报告.md")!=file_sha256(stage/"论文数据分析结果报告.md"):
        raise StageBlockedError("MD copy differs")
    files=[{"path":p.relative_to(stage).as_posix(),"sha256":file_sha256(p)} for p in stage.rglob("*") if p.is_file()]
    dump(stage/"filesource_flat/交付文件哈希.json",files)
    stamp=config["fresh"]["stamp"]
    zpath=stage/f"结果发送_完整交接包_眼动_EEG全量重跑_{stamp}.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in stage.rglob("*"):
            if p.is_file() and p!=zpath:z.write(p,p.relative_to(stage).as_posix())
    with zipfile.ZipFile(zpath) as z:
        if z.testzip():raise StageBlockedError("ZIP integrity failure")
        for r in files:
            if hashlib.sha256(z.read(r["path"])).hexdigest()!=r["sha256"]:raise StageBlockedError("ZIP bytes differ")
    destination=Path(config["fresh"]["delivery_root"]); archive=Path(config["fresh"]["archive_root"])/run.name
    old={p.relative_to(destination).as_posix():file_sha256(p) for p in destination.rglob("*") if p.is_file()}
    def validate(current,historical):
        for rel,digest in old.items():
            if file_sha256(historical/rel)!=digest:raise StageBlockedError("Archived old delivery differs")
        for r in files:
            if file_sha256(current/r["path"])!=r["sha256"]:raise StageBlockedError("Published delivery differs")
    moved=archive_publish(stage,destination,archive,validate);dump(run/"archive_path_mapping.json",moved)
    pointer={"status":"complete","scope":"eye-eeg","run_root":str(run),"git_commit":manifest["git_commit"],
             "delivery_root":str(destination),"zip":str(destination/zpath.name),"zip_sha256":file_sha256(destination/zpath.name),"archive_root":str(archive)}
    dump(run/"publication.json",pointer);dump(outputs/"eye_eeg_fresh_latest.json",pointer)
    old_summary=outputs/"realdata_run_summary.json"
    previous=json.loads(old_summary.read_text(encoding="utf-8")) if old_summary.is_file() else {}
    dump(run/"previous_result_summary.json",previous)
    current=json.loads((run/"fresh_summary.json").read_text(encoding="utf-8"))
    current["teacher_delivery"]=pointer
    current["previous_full_run_summary"]=str(run/"previous_result_summary.json")
    current["questionnaire"]={"status":"not_rerun","source_analysis_git_sha":previous.get("git_commit"),
                              "source_run_root":previous.get("run_root"),"previous_summary":previous.get("questionnaire")}
    current["scope_note"]="眼动/EEG及依赖结果为本轮全量重算；问卷和其余未运行内容保留原来源，不标为新计算"
    dump(old_summary,current)
    shutil.copyfile(run/"论文数据分析结果报告.md",outputs/"论文数据分析结果报告.md")
    for folder in ["00_eeg_trial_qc","01_eye_stage1","02_eye_stage2","03_eye_stage3_plan","04_eye_stage3","05_eeg_order","06_eeg_primary","07_eeg_window_robustness","08_synchronized_crossmodal","09_eye_figures","10_eeg_denominator_sensitivity","11_historical_comparison"]:
        target=outputs/"12_teacher_analysis"/folder
        if target.exists():
            historical=run/"superseded_formal_results"/folder;historical.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(target),str(historical))
        # PSD cache and sample waveforms retain one canonical copy in run_root.
        shutil.copytree(run/folder,target,ignore=shutil.ignore_patterns("psd_cache","samples"))
    for legacy_folder in ["03_eye_tracking","04_eeg","05_multimodal_fusion","06_models"]:
        legacy=outputs/legacy_folder
        if legacy.exists():
            (legacy/"README_历史来源说明.md").write_text(
                f"# 历史结果来源\n\n此目录未作为本轮眼动/EEG正式入口。当前重新计算的正式表位于 {run} 及 12_teacher_analysis 对应阶段。\n\n旧文件保留追溯，不与新运行的样本、功率或模型混用；问卷独立模型未重跑。\n",encoding="utf-8")
    for name in ["eeg_denominator_sensitivity_latest.json","eeg_request_handoff_latest.json"]:
        p=outputs/name
        if p.is_file():
            dump(run/("previous_"+name),json.loads(p.read_text(encoding="utf-8")))
        dump(p,{**pointer,"source_run":str(run/"10_eeg_denominator_sensitivity"),"scope":"eye-eeg fresh","main_denominator":"1-45","sensitivity_denominator":"1-40"})
    (outputs/"README_当前有效结果.md").write_text(f"# 当前有效数据分析结果\n\n本轮眼动与EEG全量重跑：{run.name}。入口：论文数据分析结果报告.md及eye_eeg_fresh_latest.json。\n\n四窗口平行；1–45 Hz主分析、1–40 Hz敏感性；方法及HF QC保留。问卷独立分析未运行，原来源和日期保留在历史记录。\n\n老师当前交付：{destination}；旧交付外部归档：{archive}。\n",encoding="utf-8")
    return pointer
