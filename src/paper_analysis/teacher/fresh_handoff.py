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


def calculation_sha(run):
    """Presentation changes must not relabel the numerical calculation."""
    return json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["git_commit"]


def typed(name, frame):
    return {"name": name, "columns": list(frame.columns),
            "rows": frame.astype(object).where(pd.notna(frame), None).values.tolist()}


def number(value):
    return "无法给出" if pd.isna(value) else f"{float(value):.8g}"


def collect_sources(run, original_design=()):
    records = []
    for p in sorted(run.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".json", ".txt", ".md"}:
            continue
        rel = p.relative_to(run)
        if len(rel.parts)==1 and p.name=="run_manifest.json":continue
        if any(v in rel.parts for v in ["samples", "psd_cache", "irregular_timestamps", "handoff_staging", "stage_seals", "artifact_previews", "superseded_formal_results"]):
            continue
        if p.name.startswith("previous_"): continue
        if p.name in {"handoff_tables.json", "handoff_guide.json", "fresh_summary.json", "source_manifest.json", "paragraph_source_map.csv", "artifact_verification.json", "artifact_data_verification.json", "published_data_verification.json", "publication.json", "archive_path_mapping.json", "结果文件总索引.xlsx", "formal_index_tables.json", "previous_result_summary.json"}:
            continue
        if p.name == "论文数据分析结果报告.md": continue
        records.append({"Source ID": f"SRC{len(records)+1:04d}", "source_path": str(p),
                        "flat_name": f"SRC{len(records)+1:04d}__{rel.parts[0]}__{p.name}",
                        "stage": rel.parts[0], "bytes": p.stat().st_size, "SHA256": file_sha256(p)})
    for source in original_design:
        p=Path(source)
        if str(p) in {r["source_path"] for r in records}:continue
        records.append({"Source ID":f"SRC{len(records)+1:04d}","source_path":str(p),
                        "flat_name":f"SRC{len(records)+1:04d}__00_original_design__{p.name}",
                        "stage":"00_original_design","bytes":p.stat().st_size,"SHA256":file_sha256(p)})
    # Duplicate filenames inside the same stage are distinguished by Source ID.
    return records


def final_readonly_checks(config, run, repo):
    """Rebuild publication evidence from finalized model tables, never inputs."""
    out = run / "12_final_verification"
    out.mkdir(exist_ok=True)
    main = run / "10_eeg_denominator_sensitivity/main_1_45"
    primary_path = run / "06_eeg_primary/06_eeg_CR2_results.csv"
    coefficient_path = main / "coefficients.csv"
    primary = pd.read_csv(primary_path)
    current = pd.read_csv(coefficient_path)
    current = current.loc[current.onset_trim_s.eq(10) & current.model.eq("Model1")]
    paired = primary.merge(current, on=["outcome", "term"], suffixes=("_primary", "_psd"),
                           how="left", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise StageBlockedError("Reference model/PSD coefficient identities differ")
    for field, tolerance in [("estimate", 1e-8), ("p.value", 1e-6)]:
        paired["absolute_difference_" + field] = abs(paired[field+"_primary"]-paired[field+"_psd"])
        paired["pass_" + field] = paired["absolute_difference_"+field].le(tolerance)
        if not paired["pass_"+field].all():
            raise StageBlockedError("Reference model/PSD values differ: " + field)
    paired["raw_p_flip"] = paired['p.value_primary'].lt(.05) != paired['p.value_psd'].lt(.05)
    if paired.raw_p_flip.any():
        raise StageBlockedError("Reference model/PSD p crosses 0.05")
    paired.to_csv(out / "reference_model_PSD_consistency.csv", index=False, encoding="utf-8-sig")
    matrices = pd.read_csv(main / "contrast_matrices.csv")
    directions = []
    for trim in [0, 5, 10, 15]:
        beta = pd.read_csv(coefficient_path)
        beta = beta.loc[beta.onset_trim_s.eq(trim) & beta.model.eq("Model1") & beta.outcome.eq("P_beta_relative")].set_index('term').estimate
        matrix = matrices.loc[matrices.onset_trim_s.eq(trim) & matrices.outcome.eq('P_beta_relative') & matrices.term.eq('WWR')]
        for row_id, contrast in matrix.groupby('contrast_row'):
            if not contrast.coefficient.isin(beta.index).all():
                raise StageBlockedError("Marginal contrast coefficient missing")
            directions.append({'onset_trim_s':trim, 'outcome':'P_beta_relative', 'factor':'WWR',
                'contrast_row':int(row_id), 'equal_margin_difference':float(np.dot(contrast.weight, beta.loc[contrast.coefficient])),
                'contrast':{1:'WWR45−WWR15', 2:'WWR75−WWR15'}[int(row_id)]})
    pd.DataFrame(directions).to_csv(out / "P_beta_WWR_marginal_directions.csv",index=False,encoding='utf-8-sig')
    source_paths = [primary_path, coefficient_path, main/'contrast_matrices.csv']
    historical = config.get('fresh',{}).get('historical_sensitivity_run')
    if historical:
        old = Path(historical)/'A/trim_10s/input.csv'
        new = main/'trim_10s/input.csv'
        keys = ['Participant','GlobalTrialOrder']
        membership = pd.read_csv(old)[keys].merge(pd.read_csv(new)[keys], on=keys, how='outer', indicator=True, validate='one_to_one')
        membership['Membership'] = membership._merge.astype(str).replace({'left_only':'historical_only','right_only':'fresh_only'})
        membership.drop(columns='_merge').to_csv(out/'sample_membership_changes.csv',index=False,encoding='utf-8-sig')
        source_paths.extend([old,new])
    dump(out/'final_readonly_checks.json', {'role':'post-model independent table checks; not a model input',
         'analysis_sha':calculation_sha(run),'verification_sha':git_sha(repo),
         'sources':[{'path':str(p),'sha256':file_sha256(p)} for p in source_paths]})


def build_report(config, run, repo):
    final_readonly_checks(config, run, repo)
    existing=run/"handoff_staging"
    if existing.exists() and any(existing.iterdir()):
        from datetime import datetime, timezone
        preserved=run/"stage_seals/handoff_attempts"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        preserved.parent.mkdir(parents=True,exist_ok=True)
        assert existing.resolve().is_relative_to(run.resolve()) and preserved.resolve().is_relative_to(run.resolve())
        shutil.move(str(existing),str(preserved))
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
        old_sync=Path(old_eye)/"08_synchronized_crossmodal/timebin_model_results.csv"
        current_sync=run/"08_synchronized_crossmodal/timebin_model_results.csv"
        if old_sync.is_file():
            a=pd.read_csv(old_sync);b=pd.read_csv(current_sync)
            sync_keys=['onset_trim_s','family','outcome','term']
            paired_sync=a.merge(b,on=sync_keys,how='outer',suffixes=('_historical','_fresh'),indicator=True,validate='one_to_one')
            for field in ['p_value','p_fdr_bh_family','p_fdr_bh_parallel']:
                paired_sync['flip_'+field]=(paired_sync[field+'_historical'].lt(.05)!=paired_sync[field+'_fresh'].lt(.05)).mask(paired_sync[field+'_historical'].isna()|paired_sync[field+'_fresh'].isna())
            paired_sync['direction_changed']=np.sign(paired_sync.estimate_historical)!=np.sign(paired_sync.estimate_fresh)
            paired_sync.to_csv(comparison_dir/'historical_to_fresh_synchronized.csv',index=False,encoding='utf-8-sig')
            dump(comparison_dir/'synchronized_comparison_sources.json',{'role':'historical read only after fresh models',
                 'files':[{'path':str(p),'sha256':file_sha256(p)} for p in [old_sync,current_sync]]})
    keys=["Participant","GlobalTrialOrder"]
    eye_input=pd.read_excel(run/"02_eye_stage2/06_trial_level_eye_tracking_data.xlsx")
    included=eye_input.IncludedPrimary.astype(str).str.lower().isin(["true","1"])
    eeg_input=pd.read_csv(run/"10_eeg_denominator_sensitivity/main_1_45/trim_10s/input.csv")
    intersection=eye_input.loc[included,keys].merge(eeg_input[keys],on=keys,how="inner",validate="one_to_one")
    cross_path=run/"12_final_verification/cross_modality_common_keys.csv"
    cross_path.parent.mkdir(exist_ok=True)
    intersection.to_csv(cross_path,index=False,encoding="utf-8-sig")
    records = collect_sources(run,[config[k] for k in ["questionnaire_file","participant_information","trial_order_mapping","scene_aoi_mapping"]])
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
    def statistics(row, source, selection, fields):
        for field in fields:
            if field in row:
                numeric_row(selection + " / " + field, row[field], source, field, selection)
    flow_path = run / "00_eeg_trial_qc/sample_flow.csv"
    flow = pd.read_csv(flow_path); common = flow.loc[flow.Stage.eq("four_window_common_QC")].iloc[0]
    eye_path = run / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx"
    eye = pd.read_excel(eye_path); selected = eye.loc[eye.IncludedPrimary.astype(str).str.lower().isin(["true", "1"])]
    sensitivity = run / "10_eeg_denominator_sensitivity"
    summary = json.loads((sensitivity / "summary.json").read_text(encoding="utf-8"))
    add("# 眼动与 EEG 全量重跑数据分析结果报告")
    add()
    add(f"本轮使用当前源文件重新计算，分析代码版本 `{calculation_sha(run)}`；交付整理版本 `{git_sha(repo)}`。EEG 最终四窗口共同样本为 {int(common.Participants)} 人、{int(common.Trials)} 试次；眼动 60% 主分析为 {selected.Participant.nunique()} 人、{len(selected)} 试次。",
        [flow_path, eye_path], "Participants/Trials/IncludedPrimary", "four_window_common_QC; eye IncludedPrimary=true", True)
    add("本轮眼动、EEG及相关交集／同步结果替代此前对应结果。问卷独立分析未运行，其旧结果不在本轮更新范围。论文 Word 未修改，后文列出需要更新的结果及结论。")
    add("本轮Stage 3未提供问卷S3试次结局输入，因此S3预测及S3—眼动精确交集不执行；此前这些结果保留历史来源，不属于本轮重新验证，也不能与新EEG样本混用。",[run/"resolved_config.json"],"stage3.s3_trial_file")
    add()
    add("## 当前源文件和方法")
    add("侯集瀚、李泽豪的 SET/FDT 使用已与 8 月 18 日更新后记录核对一致的版本，运行前后验证 SHA256；所有 EEG 来源、原始眼动 CSV、AOI、底图、试次映射和协变量文件登记哈希。",
        [run / "source_hashes_before.json", run / "source_hashes_after.json"], "path/sha256")
    add("EEG从已有预处理波形开始，不重复滤波或人工ICA操作。保留 HF、RMS、峰峰值及原有QC规则，阈值为 median + 3.5 × MAD，坏试次比例 >30% 时整人排除。阈值由本轮数据重新计算。",
        [run / "00_eeg_trial_qc/QC_policy.json", run / "00_eeg_trial_qc/eeg_onset_sensitivity_qc_thresholds.csv"], "robust_metrics/threshold")
    add("0／5／10／15 s 是同等地位的平行窗口，各自计算QC后取共同试次。ROI先时域平均，再Welch PSD；θ 4–7、α 8–12、β 13–30 Hz。1–45 Hz保留为主分母，1–40 Hz为同一PSD和分子的敏感性分析，QC不随分母改变。",
        [sensitivity / "analysis_contract.json", sensitivity / "paired_psd_integrals.csv"], "PSD/factor_levels/total_1_45/total_1_40")
    add("Q1.4四档的前两档为Low、后两档为High，未用Q1.5代替；重新从原问卷推导并与参与者信息核对。",
        [run / "00_teacher_inputs/Q1_4_group_check.csv",Path(config["questionnaire_file"])], "Q1.4Original/ExerciseFrequency")
    add("眼动保留原fixation、AOI、60%主阈值及50%／70%敏感性、模型和校正方法；眼动主样本不因缺少EEG而排除。AOI人工标注、投影规则或来源材料的已知限制继续保留，不把全量计算等同于消除来源缺口。")
    versions_path = run / "stage_calculation_versions.json"
    if versions_path.is_file():
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
        add("## 阶段修正与计算版本")
        for item in versions.get("stages", []):
            add(f"{item['stage']}：计算版本 `{item['code_sha']}`。{item['reason']} 原输出保留在 `{item['previous_outputs']}`。",
                [versions_path], "stage/code_sha/reason/previous_outputs")
    proof_path = run / "reproducibility_verification.json"
    if proof_path.is_file():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        add("## 环境恢复与三部分复算验收")
        add(f"复算核验代码版本 `{proof['verification_sha']}`，总体状态 `{proof['status']}`。环境可运行与数值复算一致分别验证；失败模型和缺失推断不作为通过证据。",
            [proof_path], "verification_sha/status")
        add("|部分|状态|对照表数量|对照记录数量|")
        add("|---|---|---:|---:|")
        for scope, label in [("eye", "眼动"), ("eeg", "EEG"), ("joint", "眼动＋EEG")]:
            result = proof["scopes"][scope]
            add(f"|{label}|{result['status']}|{result['comparisons']}|{result['rows']}|",
                [proof_path], f"scopes.{scope}", numeric=True)
        for note in proof.get("scope_notes", []):
            add(str(note), [proof_path], "scope_notes")
    add()
    add("## EEG 样本与排除")
    add("|阶段|人数|试次数|"); add("|---|---:|---:|")
    for _, r in flow.iterrows():
        add(f"|{r.Stage}|{int(r.Participants)}|{int(r.Trials)}|", [flow_path], "Stage/Participants/Trials", f"Stage={r.Stage}", True)
        numeric_row(r.Stage + "人数", int(r.Participants), flow_path, "Participants", f"Stage={r.Stage}")
        numeric_row(r.Stage + "试次", int(r.Trials), flow_path, "Trials", f"Stage={r.Stage}")
    thresholds_path=run/"00_eeg_trial_qc/eeg_onset_sensitivity_qc_thresholds.csv"
    thresholds=pd.read_csv(thresholds_path)
    add("|窗口s|QC指标|阈值估计有效试次数|median|MAD|阈值|");add("|---:|---|---:|---:|---:|---:|")
    for _, row in thresholds.iterrows():
        add(f"|{int(row.onset_trim_s)}|{row.metric}|{int(row.n)}|{number(row['median'])}|{number(row.mad)}|{number(row.threshold)}|",[thresholds_path],"n/median/mad/threshold",f"onset_trim_s={row.onset_trim_s};metric={row.metric}",True)
        statistics(row,thresholds_path,f"onset_trim_s={row.onset_trim_s};metric={row.metric}",["n","median","mad","threshold"])
    data = pd.read_csv(run / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv")
    for person in sorted({Path(name).stem for name in config["fresh"].get("locked_source_hashes",{})}):
        d = data.loc[data.participant_id.eq(person) & data.onset_trim_s.eq(0)]
        retained = int(d.CommonQCIncluded.sum())
        add(f"{person}：本轮当前文件已进入导出和QC；四窗口共同保留 {retained} 个试次。{('进入主要模型。' if retained else '未进入主要模型，不能因文件已纠正而免除QC。')}",
            [run / "00_eeg_trial_qc/eeg_onset_sensitivity_trial_long.csv"], "CommonQCIncluded/eeg_qc_reasons", f"participant_id={person}; onset_trim_s=0", True)
        subject_path=run/"00_eeg_trial_qc/eeg_onset_sensitivity_subject_qc.csv"
        subjects=pd.read_csv(subject_path)
        for _, row in subjects.loc[subjects.participant_id.eq(person)].iterrows():
            add(f"{person} {int(row.onset_trim_s)} s：坏试次 {int(row.n_bad_eeg_scenes)}/{int(row.n_eeg_scenes)}，比例 {row.bad_eeg_scene_fraction:.8g}，整人质量排除={row.eeg_subject_quality_exclusion}。",[subject_path],"n_bad_eeg_scenes/n_eeg_scenes/bad_eeg_scene_fraction/eeg_subject_quality_exclusion",f"participant_id={person};onset_trim_s={row.onset_trim_s}",True)
            statistics(row,subject_path,f"participant_id={person};onset_trim_s={row.onset_trim_s}",["n_bad_eeg_scenes","n_eeg_scenes","bad_eeg_scene_fraction"])
    add("完整试次和整人排除原因见本轮QC表；HF的全通道均值在平均参考后接近零这一已知局限仍需在解释中保留。本轮按用户确认维持原方法，未调整阈值追求指定人数。")
    membership_path=run/"12_final_verification/sample_membership_changes.csv"
    if membership_path.is_file():
        membership=pd.read_csv(membership_path)
        counts=membership.Membership.value_counts()
        add(f"与历史模型输入比较：共同试次 {int(counts.get('both',0))} 个；仅历史保留 {int(counts.get('historical_only',0))} 个；仅本轮保留 {int(counts.get('fresh_only',0))} 个。该名单比较在本轮模型完成后进行，旧输入没有供给本轮模型。",[membership_path],"Membership",numeric=True)
        for key,value in counts.items():numeric_row("历史→当前样本 / "+key,int(value),membership_path,"Membership",f"Membership={key}")
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
        statistics(r, sensitivity / "main_1_45/family_coefficients.csv", f"window={r.onset_trim_s};outcome={r.outcome};term={r.term};family={r.family_id}", ["estimate","std.error","df","CI_low","CI_high","p.value","within_q","joint_q"])
    target = factors.loc[(factors.family_id.eq("factor_core_relative") & factors.outcome.eq("O_theta_relative") & factors.term.eq("Complexity")) |
                        (factors.family_id.eq("factor_expanded_relative") & factors.outcome.eq("P_beta_relative") & factors.term.eq("WWR"))]
    add("|窗口s|结局|总体效应|F|分子df|分母df|原始p|联合q|"); add("|---:|---|---|---:|---:|---:|---:|---:|")
    for _, r in target.iterrows():
        add(f"|{int(r.onset_trim_s)}|{r.outcome}|{r.term}|{number(r.Fstat)}|{number(r.df_num)}|{number(r.df_denom)}|{number(r['p.value'])}|{number(r.joint_q)}|",
            [sensitivity / "main_1_45/family_factors.csv"], "Fstat/df_num/df_denom/p.value/joint_q", f"window={r.onset_trim_s};outcome={r.outcome};term={r.term};family={r.family_id}", True)
        statistics(r, sensitivity / "main_1_45/family_factors.csv", f"window={r.onset_trim_s};outcome={r.outcome};term={r.term};family={r.family_id}", ["estimate","std.error","CI_low","CI_high","Fstat","df_num","df_denom","p.value","within_q","joint_q"])
    add("上述表格用于更新表6、枕区θ复杂度、alpha／theta时序、顶区β WWR和上一场景结果。显著性只依据各自完整检验族的联合q；无效推断标为无法给出，不等同于不显著。不同窗口不能事后选一个显著窗口代替平行分析。")
    for label, rows in [("枕区θ复杂度边际效应", target.loc[target.outcome.eq("O_theta_relative")]), ("顶区β的WWR总体效应", target.loc[target.outcome.eq("P_beta_relative")])]:
        windows = "、".join(str(int(v)) for v in rows.loc[rows.joint_q.lt(.05),"onset_trim_s"])
        add(f"{label}：通过四窗口联合校正的窗口为 {windows or '无'} s；其余窗口不能据原始p称为联合校正后显著。", [sensitivity / "main_1_45/family_factors.csv"], "joint_q", label, True)
    previous = coefficients.loc[coefficients.family_id.eq("previous_core_relative")]
    add(f"上一场景：{int(previous.joint_q.lt(.05).sum())} 项通过核心previous-scene完整族的联合校正。上一场景系数应按加性模型及本表解释。", [sensitivity / "main_1_45/family_coefficients.csv"], "joint_q", "family=previous_core_relative", True)
    directions_path=run/"12_final_verification/P_beta_WWR_marginal_directions.csv"
    if directions_path.is_file():
        directions=pd.read_csv(directions_path)
        add("顶区β的WWR方向另以保存的对比矩阵乘系数得到等权边际差，不能为2自由度总体F检验虚构一个β。边际差不附造独立p；总体显著性仍按上表完整检验族读取。",[directions_path,sensitivity/"main_1_45/contrast_matrices.csv"])
        add("|窗口s|等权边际对比|relative-power差值|");add("|---:|---|---:|")
        for _,row in directions.iterrows():
            add(f"|{int(row.onset_trim_s)}|{row.contrast}|{number(row.equal_margin_difference)}|",[directions_path],"contrast/equal_margin_difference",f"window={row.onset_trim_s};contrast={row.contrast}",True)
            statistics(row,directions_path,f"window={row.onset_trim_s};contrast={row.contrast}",["equal_margin_difference"])
    if old_path:
        p=comparison_dir/"manuscript_historical_to_fresh.csv";changes=pd.read_csv(p)
        add("### 历史正文结果与当前全量重跑的差异")
        add("历史对照只在全部新计算完成后读取，用于判断论文结果变化；该比较同时含源文件和QC样本变化，不能解释为纯分母效应。",[comparison_dir/"comparison_source.json"])
        add("|正文项目|联合q翻转数|方向变化数|判断|");add("|---|---:|---:|---|")
        for claim,g in changes.groupby("manuscript_claim",sort=False):
            flips=int(g.flip_joint_q.eq(True).sum());directions=int(g.direction_changed.eq(True).sum())
            scalar_direction=g.estimate_from.notna()&g.estimate_to.notna()
            verdict=("存在显著性变化，需更新正文" if flips else "方向变化需结合系数大小与p/q解释" if directions else
                     "联合显著性未改变；总体F无单一方向，边际对比另列" if not scalar_direction.any() else "可比较的系数方向与联合显著性未改变，数值需更新")
            add(f"|{claim}|{flips}|{directions}|{verdict}|",[p],"manuscript_claim/flip_joint_q/direction_changed",f"manuscript_claim={claim}",True)
        for _, row in changes.loc[changes.flip_joint_q.eq(True)].iterrows():
            add(f"历史→当前：{int(row.onset_trim_s)} s，{row.outcome}，{row.term}，联合q从 {number(row.joint_q_from)} 变为 {number(row.joint_q_to)}，β从 {number(row.estimate_from)} 变为 {number(row.estimate_to)}。这是源数据和QC共同样本变化的结果，不是分母变化。", [p], "estimate_from/estimate_to/joint_q_from/joint_q_to", f"window={row.onset_trim_s};outcome={row.outcome};term={row.term};family={row.family_id}", True)
            statistics(row,p,f"historical→fresh;window={row.onset_trim_s};outcome={row.outcome};term={row.term};family={row.family_id}",["estimate_from","estimate_to","p.value_from","p.value_to","joint_q_from","joint_q_to"])
    add()
    add("## 1–40 Hz 分母敏感性")
    comparison_path = sensitivity / "model_comparisons.csv"
    comparisons = pd.read_csv(comparison_path)
    changed = comparisons.loc[comparisons['flip_p.value'].eq(True) | comparisons.flip_within_q.eq(True) | comparisons.flip_joint_q.eq(True) | comparisons.direction_changed.eq(True)]
    add(f"同一批新PSD上，预定检验族记录到 {int(comparisons.flip_joint_q.eq(True).sum())} 项联合q显著性翻转。方向变化及近零变号完整保留在配对表，不能一概写成所有结论不变。",
        [comparison_path], "flip_joint_q/direction_changed", numeric=True)
    raw_flips = comparisons.loc[comparisons['flip_p.value'].eq(True)]
    unique_flips = raw_flips.drop_duplicates(["onset_trim_s","outcome","model","term","test_kind"])
    add(f"原始p跨越0.05有 {len(raw_flips)} 条检验族记录，对应 {len(unique_flips)} 个唯一检验；窗口内q翻转 {int(comparisons.flip_within_q.eq(True).sum())} 条，联合q翻转 {int(comparisons.flip_joint_q.eq(True).sum())} 条。重复登记于不同检验族不算两个独立模型变化。", [comparison_path], "flip_p.value/flip_within_q/flip_joint_q", numeric=True)
    for _, row in unique_flips.iterrows():
        add(f"原始p边界：{int(row.onset_trim_s)} s，{row.outcome}，{row.term}，p={number(row['p.value_from'])}→{number(row['p.value_to'])}；其校正后的判断请按对应检验族q读取。", [comparison_path], "p.value_from/p.value_to", f"window={row.onset_trim_s};outcome={row.outcome};term={row.term}", True)
    for _, row in comparisons.loc[comparisons.direction_changed.eq(True)].iterrows():
        add(f"方向变化：{int(row.onset_trim_s)} s，{row.outcome}，{row.model}/{row.term}，β={number(row.estimate_from)}→{number(row.estimate_to)}，p={number(row['p.value_from'])}→{number(row['p.value_to'])}。接近零且未校正的补充系数不解读为认知效应反转。", [comparison_path], "estimate_from/estimate_to/p.value_from/p.value_to", f"window={row.onset_trim_s};outcome={row.outcome};model={row.model};term={row.term}", True)
    for col in ['flip_p.value','flip_within_q','flip_joint_q','direction_changed']:
        numeric_row("分母比较 / " + col, int(comparisons[col].eq(True).sum()), comparison_path, col)
    add("|窗口s|结局|模型／效应|1–45 β或F|1–40 β或F|1–45 q|1–40 q|"); add("|---:|---|---|---:|---:|---:|---:|")
    for _, r in changed.iterrows():
        a = r.estimate_from if pd.notna(r.estimate_from) else r.get("Fstat_from", np.nan)
        b = r.estimate_to if pd.notna(r.estimate_to) else r.get("Fstat_to", np.nan)
        add(f"|{int(r.onset_trim_s)}|{r.outcome}|{r.model}／{r.term}|{number(a)}|{number(b)}|{number(r.joint_q_from)}|{number(r.joint_q_to)}|",
            [comparison_path], "estimate/Fstat/joint_q", f"window={r.onset_trim_s};outcome={r.outcome};model={r.model};term={r.term};family={r.family_id}", True)
        statistics(r, comparison_path, f"window={r.onset_trim_s};outcome={r.outcome};model={r.model};term={r.term};family={r.family_id}", ["estimate_from","estimate_to","Fstat_from","Fstat_to","p.value_from","p.value_to","within_q_from","within_q_to","joint_q_from","joint_q_to"])
    powers_path = sensitivity / "paired_psd_integrals.csv"; powers = pd.read_csv(powers_path)
    add("|窗口s|40–45占1–45总功率均值%|实际分母减少均值%|"); add("|---:|---:|---:|")
    for trim, g in powers.groupby("onset_trim_s"):
        add(f"|{int(trim)}|{g.percent_40_45.mean():.8g}|{g.denominator_reduction_percent.mean():.8g}|", [powers_path], "percent_40_45/denominator_reduction_percent", f"onset_trim_s={trim};all ROIs", True)
        for field in ["percent_40_45","denominator_reduction_percent"]:
            numeric_row(f"窗口{int(trim)}s / {field}均值", float(g[field].mean()), powers_path, field, f"onset_trim_s={trim};all ROIs;mean")
    add("独立40–45 Hz积分与两分母之差因离散频率边界而不同，分别报告；未统一按平均百分比缩放试次。")
    frequency_path=run/"12_final_verification/frequency_mask_endpoints.csv"
    if frequency_path.is_file():
        frequency=pd.read_csv(frequency_path)
        cols=["frequency_step","nominal_1_45_first","nominal_1_45_last","nominal_1_40_last","nominal_40_45_first"]
        for _, row in frequency[cols].drop_duplicates().iterrows():
            add(f"实际频率网格步长 {row.frequency_step:.10g} Hz；[1,45]掩码端点 {row.nominal_1_45_first:.10g}–{row.nominal_1_45_last:.10g} Hz，[1,40]上端点 {row.nominal_1_40_last:.10g} Hz，[40,45]下端点 {row.nominal_40_45_first:.10g} Hz。两段之间的离散梯形积分桥段解释了占比与分母减少比例的差别。",[frequency_path],"/".join(cols),numeric=True)
            statistics(row,frequency_path,"全量PSD实际频率网格端点",cols)
    add()
    add("## 眼动与同步结果")
    add(f"眼动60%主样本与EEG四窗口共同QC的精确参与者×试次交集为 {intersection.Participant.nunique()} 人、{len(intersection)} 试次；这是场景级交集，尚未加上时钟条件。",[cross_path],"Participant/GlobalTrialOrder",numeric=True)
    numeric_row("眼动×EEG精确交集人数",int(intersection.Participant.nunique()),cross_path,"Participant","nunique")
    numeric_row("眼动×EEG精确交集试次",len(intersection),cross_path,"GlobalTrialOrder","rows")
    numeric_row("眼动主分析人数", int(selected.Participant.nunique()), eye_path, "Participant", "IncludedPrimary=true")
    numeric_row("眼动主分析试次", len(selected), eye_path, "IncludedPrimary", "IncludedPrimary=true")
    eye_results = []
    robust_path=run/"02_eye_stage2/12_CR2_robust_results.csv"
    eye_robust=pd.read_csv(robust_path)
    for filename in ["09_familyA_primary_models.csv", "10_familyB_primary_models.csv"]:
        p = run / "02_eye_stage2" / filename
        t = pd.read_csv(p); eye_results.append(t)
        add(f"眼动{filename[:2]}结果同时保留主似然模型与CR2 companion，估计、区间、原始p及BH见源表。", [p])
        add("|结局|效应|主模型估计|主模型p|CR2 companion估计|CR2原始p|CR2 BH|"); add("|---|---|---:|---:|---:|---:|---:|")
        for _, r in t.loc[~t.term.eq("(Intercept)")].iterrows():
            companion=eye_robust.loc[eye_robust.outcome.eq(r.outcome)&eye_robust.term.eq(r.term)]
            if len(companion)!=1:raise StageBlockedError("Eye primary/companion key mismatch")
            robust=companion.iloc[0]
            add(f"|{r.outcome}|{r.term}|{number(r.get('estimate',np.nan))}|{number(r.get('p.value.likelihood',r.get('p.value',np.nan)))}|{number(robust.estimate)}|{number(robust['p.value'])}|{number(robust['p.value.BH'])}|",
                [p,robust_path], "estimate/p.value.likelihood/p.value/p.value.BH", f"outcome={r.outcome};term={r.term}", True)
            statistics(r, p, f"primary;outcome={r.outcome};term={r.term}", ["estimate","std.error","std.error.likelihood","std.error.CR2","df","conf.low","conf.high","p.value","p.value.likelihood","p.value.CR2","p.value.BH"])
            statistics(robust,robust_path,f"CR2 companion;outcome={r.outcome};term={r.term}",["estimate","std.error","df","p.value","p.value.BH"])
    add("眼动主要模型、50%／70%敏感性、Block1、LOPO、AOI边界、补充模型和诊断均在本轮重新运行。结构性缺失、未访问TTFF及失败模型保留原规则；WWR配对的Holm与CR2 BH分别解释。")
    add("部分眼动Family B源表未提供主似然p，报告保留为无法给出；其CR2原始p和BH见12_CR2_robust_results.csv及对应源表。此处缺失不自动代表拟合失败。", [run/"02_eye_stage2/12_CR2_robust_results.csv"])
    if old_eye:
        p=comparison_dir/"historical_to_fresh_eye.csv";old_to_new=pd.read_csv(p)
        for field in [c for c in old_to_new if c.startswith("flip_")]:
            add(f"眼动历史对照 {field.removeprefix('flip_')}：跨越0.05的配对记录 {int(old_to_new[field].eq(True).sum())} 项；完整配对及未配对记录保留。",[p],field,numeric=True)
    sync_path = run / "08_synchronized_crossmodal/synchronized_crossmodal_sample_and_source.xlsx"
    sync = pd.read_excel(sync_path).iloc[0]
    add(f"重新构建绝对时钟同步并取眼动60%、EEG共同QC和时钟QC的精确交集：{int(sync.Participants)} 人、{int(sync.Trials)} 试次、{int(sync.TimeBins)} 个四窗口AOI×time-bin数据行。TimeBins按源表行数统计，已包含AOI和窗口维度。缺少可验证时钟的参与者没有补造时间映射。",
        [sync_path], "Participants/Trials/TimeBins", numeric=True)
    statistics(sync, sync_path, "四窗口同步共同样本", ["Participants","Trials","TimeBins"])
    sync_model_path=run/'08_synchronized_crossmodal/timebin_model_results.csv'
    sync_models=pd.read_csv(sync_model_path)
    parallel=sync_models.loc[sync_models.interpretation_tier.eq('parallel')]
    add("同步GEE按参与者聚类，四窗口平行时间效应族以p_fdr_bh_parallel判断；这是同步时间模型，不将其解释成眼动导致EEG变化。",[sync_model_path],"formula/model_type/hypothesis_family")
    add("|窗口s|同步模型类别|联合校正后显著的时间效应项数|");add("|---:|---|---:|")
    for (family,trim), group in parallel.groupby(['family','onset_trim_s']):
        count=int(group.p_fdr_bh_parallel.lt(.05).sum())
        add(f"|{int(trim)}|{family}|{count}|",[sync_model_path],"p_fdr_bh_parallel",f"interpretation_tier=parallel;family={family};onset_trim_s={trim}",True)
        numeric_row(f"同步{trim}s/{family}/显著时间效应项数",count,sync_model_path,"p_fdr_bh_parallel",f"interpretation_tier=parallel;family={family};onset_trim_s={trim}")
    sync_comparison_path=comparison_dir/'historical_to_fresh_synchronized.csv'
    if sync_comparison_path.is_file():
        sync_comparison=pd.read_csv(sync_comparison_path)
        scoped=sync_comparison.loc[sync_comparison.interpretation_tier_fresh.eq('parallel')]
        flips=int(scoped.flip_p_fdr_bh_parallel.eq(True).sum())
        add(f"同步历史→当前：预定时间效应中有 {flips} 项四窗口联合校正显著性变化；逐项估计、区间和p/q配对保留，不能将样本变化视为同一PSD分母效应。",[sync_comparison_path],"flip_p_fdr_bh_parallel", "interpretation_tier_fresh=parallel",True)
        for _,row in scoped.loc[scoped.flip_p_fdr_bh_parallel.eq(True)].iterrows():
            add(f"同步变化：{int(row.onset_trim_s)} s，{row.outcome}/{row.term}，联合q {number(row.p_fdr_bh_parallel_historical)}→{number(row.p_fdr_bh_parallel_fresh)}。",[sync_comparison_path],"p_fdr_bh_parallel_historical/p_fdr_bh_parallel_fresh",f"window={row.onset_trim_s};outcome={row.outcome};term={row.term}",True)
            statistics(row,sync_comparison_path,f"同步历史→当前;window={row.onset_trim_s};outcome={row.outcome};term={row.term}",["estimate_historical","estimate_fresh","p_value_historical","p_value_fresh","p_fdr_bh_parallel_historical","p_fdr_bh_parallel_fresh"])
    add()
    add("## 复核与论文更新")
    add("Python逐一独立积分完整PSD，R独立读取原问卷XLSX、四窗口模型CSV和MAT PSD，并重算完整BH。样本、分组、缺失位置、键和功率均核对；两种分母的absolute结果仅在本轮输入完全相同时复用。",
        [sensitivity / "independent_verification/summary.json", sensitivity / "exporter_PSD_regression.csv"])
    consistency=run/"12_final_verification/reference_model_PSD_consistency.csv"
    if consistency.is_file():
        checks=pd.read_csv(consistency)
        add(f"10 s核心主模型与本轮PSD模型路径独立配对 {len(checks)} 条系数；β最大绝对差 {checks['absolute_difference_estimate'].max():.8g}，原始p最大绝对差 {checks['absolute_difference_p.value'].max():.8g}，均通过预定容差。", [consistency], "absolute_difference_estimate/absolute_difference_p.value", numeric=True)
    waveform_checks=run/"12_final_verification/waveform_Welch_spot_checks.csv"
    if waveform_checks.is_file():
        checks=pd.read_csv(waveform_checks)
        add(f"从当前FDT波形独立重建 {len(checks)} 个参与者×试次×ROI×窗口的Welch PSD，覆盖0/5/10/15 s及F/P/O；SciPy与MATLAB最大PSD绝对差 {checks.max_psd_abs_difference.max():.8g}，全部通过rtol=1e-10、atol=1e-12。频谱抽查与全量PSD积分核对分别保存。", [waveform_checks], "max_psd_abs_difference/Pass", numeric=True)
        numeric_row("波形→Welch独立抽查 / PSD最大绝对差",float(checks.max_psd_abs_difference.max()),waveform_checks,"max_psd_abs_difference","all;max")
    bootstrap_paths = [run/"06_eeg_primary/08_eeg_bootstrap_failures.csv"] + [run/f"07_eeg_window_robustness/trim_{v}s/08_eeg_bootstrap_failures.csv" for v in [0,5,15]]
    for trim, path in zip([10,0,5,15], bootstrap_paths):
        boot = pd.read_csv(path)
        add(f"EEG {trim} s：{len(boot)} 个核心结局各请求 {int(boot.requested.min())} 次参与者聚类bootstrap，失败合计 {int(boot.failed_replicates.sum())} 次。使用本轮输入重新计算，未复用历史抽样。", [path], "outcome/requested/failed_replicates", numeric=True)
        for _, row in boot.iterrows():
            statistics(row, path, f"window={trim};outcome={row.outcome}", ["requested","failed_replicates"])
    eye_boot_path = run/"04_eye_stage3/08d_experience_cluster_bootstrap_5000.csv"
    eye_boot = pd.read_csv(eye_boot_path)
    add(f"眼动经验组bootstrap输出 {len(eye_boot)} 行：按原触发条件执行。空表需结合触发清单和拟合诊断区分未触发与拟合失败，不能声称已执行5000次眼动抽样。", [eye_boot_path], "row count", numeric=True)
    eye_draw_path = run / "04_eye_stage3/bootstrap_draw_provenance.json"
    if eye_draw_path.is_file():
        eye_draws = json.loads(eye_draw_path.read_text(encoding="utf-8"))
        for outcome, item in eye_draws["outcomes"].items():
            add(f"眼动 {outcome}：请求 {item['requested']} 次参与者聚类bootstrap；seed={eye_draws['seed']}，完整抽样哈希及初末随机状态已保存。",
                [eye_draw_path], "seed/participants/outcomes", f"outcome={outcome}", True)
    add("论文需要用本轮样本与统计源表更新2.2样本描述、Results 3.2／3.3、表6及相关补充表。分析方法保持不变；若方法文字与实际保留的代码公式不一致，应据实澄清。HF局限、factor历史脚本缺口和AOI来源限制继续记录。")
    report = run / "论文数据分析结果报告.md"; report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    pd.DataFrame(mapping).to_csv(run / "paragraph_source_map.csv", index=False, encoding="utf-8-sig")
    dump(run / "source_manifest.json", records)
    samples = pd.concat([flow.assign(Grain="Participant×Trial"), pd.DataFrame([{"Stage":"eye_60pct_main","Participants":selected.Participant.nunique(),"Trials":len(selected),"Grain":"Participant×Trial"},
                         {"Stage":"eye_eeg_exact_common","Participants":intersection.Participant.nunique(),"Trials":len(intersection),"Grain":"Participant×Trial"},
                         {"Stage":"clock_eye_eeg_common","Participants":sync.Participants,"Trials":sync.Trials,"DataRows":sync.TimeBins,"Grain":"Participant×Trial×onset_trim×bin×AOI"}])], ignore_index=True)
    omitted = [{"路径":r["path"],"SHA256":r["sha256"],"字节":r["bytes"],"原因":"大型原始波形／眼动采集不重复打包；来源哈希可追溯"}
               for r in json.loads((run / "source_hashes_before.json").read_text(encoding="utf-8")) if Path(r["path"]).suffix.lower() in {".set",".fdt",".easy",".csv"}]
    for path in sorted((sensitivity/"psd_cache").rglob("*.mat")):
        omitted.append({"路径":str(path),"SHA256":file_sha256(path),"字节":path.stat().st_size,"原因":"完整PSD缓存保留正式运行目录；积分及来源证明已打包"})
    for path in sorted((run/"00_clock_cache/irregular_timestamps").glob("*.csv")):
        omitted.append({"路径":str(path),"SHA256":file_sha256(path),"字节":path.stat().st_size,"原因":"逐样本时钟缓存留正式目录；时钟QC、对应记录和来源哈希已打包"})
    overview = pd.DataFrame([{"项目":"本轮范围","说明":"眼动、EEG、精确交集和同步；问卷独立模型未运行"},
        {"项目":"分析代码SHA","说明":calculation_sha(run)},{"项目":"交付代码SHA","说明":git_sha(repo)},{"项目":"阅读顺序","说明":"MD → 段落来源映射 → Source ID → filesource_flat"},
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
        f"同一PSD的分母比较：联合q翻转{int(comparisons.flip_joint_q.eq(True).sum())}项，窗口内q翻转{int(comparisons.flip_within_q.eq(True).sum())}项；原始p跨0.05涉及{len(unique_flips)}个唯一检验。历史→当前样本变化及正文q变化另列，不能归因于分母。",
        "当前主要发现：枕区θ复杂度效应、顶区β的WWR总体检验和alpha／theta时序效应均按四窗口及各自完整检验族报告。总体F不能用一个β代表方向；上一场景的阴性结果不构成无效应证明。",
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
    dump(run / "fresh_summary.json", {"status":"complete","scope":"eye-eeg","git_commit":calculation_sha(run),"publication_git_sha":git_sha(repo),"run_root":str(run),
        "eeg":{"participants":int(common.Participants),"trials":int(common.Trials)},
        "eye":{"participants":int(selected.Participant.nunique()),"trials":len(selected)},
        "synchronized":{"participants":int(sync.Participants),"trials":int(sync.Trials),"timebins":int(sync.TimeBins)},
        "questionnaire":"not_rerun","denominator_sensitivity":summary})


def make_artifacts(config, run, repo):
    """Use the bundled Artifact Tool for the nine-sheet workbook."""
    stage = run / "handoff_staging"
    runtime = config["fresh"].get("artifact_python") or str(Path(config["fresh"]["node"]).parents[2]/"python/python.exe")
    if not Path(runtime).is_file():
        raise StageBlockedError("Bundled document runtime unavailable")
    subprocess.run([runtime,str(repo/"scripts/build_fresh_handoff_guide.py"),
                    str(run/"handoff_guide.json"),str(stage/"数据来源交接说明.docx")],cwd=repo,check=True)
    work = repo / ".codex_tmp"; work.mkdir(exist_ok=True)
    script = work / "fresh_index.mjs"; shutil.copyfile(repo / "scripts/build_fresh_handoff_index.mjs", script)
    previews=run / "artifact_previews"; previews.mkdir(exist_ok=True)
    subprocess.run([config["fresh"]["node"],str(script),str(run / "handoff_tables.json"),
                    str(stage / "数据来源交接索引.xlsx"),str(previews)],cwd=repo,check=True)
    payload=json.loads((run/"handoff_tables.json").read_text(encoding="utf-8"))
    payload["linkPrefix"]=Path(config["fresh"]["delivery_root"]).as_posix()+"/filesource_flat/"
    dump(run/"formal_index_tables.json",payload)
    subprocess.run([config["fresh"]["node"],str(script),str(run/"formal_index_tables.json"),
                    str(run/"结果文件总索引.xlsx"),str(previews/"formal_index")],cwd=repo,check=True)


def refresh_formal_entrypoints(outputs, run, pointer, moved):
    """Archive obsolete active summaries and expose the new scoped provenance."""
    outputs,run=Path(outputs).resolve(),Path(run).resolve()
    if not run.is_relative_to(outputs/"teacher_runs"):
        raise StageBlockedError("Formal archive must remain in the named current run")
    names=["结果文件总索引.xlsx","realdata_run_summary.csv","code_publication.json",
           "老师本次眼动与EEG核查报告.md","老师本次EEG核查报告.md","老师任务完成矩阵.xlsx",
           "老师最新要求_EEG分母敏感性.md","artifact_reuse_manifest.xlsx","README_老师本次EEG交付.md"]
    archive=run/"superseded_formal_results/root_entrypoints"
    archive.mkdir(parents=True,exist_ok=True)
    for name in names:
        source=outputs/name; target=archive/name
        if source.is_file():
            if not source.resolve().is_relative_to(outputs) or not target.resolve().is_relative_to(run):
                raise StageBlockedError("Formal archive target outside the selected roots")
            if target.exists():raise StageBlockedError(f"Previous formal entry already archived: {target}")
            shutil.move(str(source),str(target))
    shutil.copyfile(run/"结果文件总索引.xlsx",outputs/"结果文件总索引.xlsx")
    old=pd.read_csv(archive/"realdata_run_summary.csv") if (archive/"realdata_run_summary.csv").is_file() else pd.DataFrame()
    summary=json.loads((run/"fresh_summary.json").read_text(encoding="utf-8"))
    q_count=old.iloc[0].get("questionnaire_participants",None) if len(old) else None
    pd.DataFrame([{"run_id":run.name,"status":"complete","git_commit":calculation_sha(run),
                   "publication_git_sha":pointer["publication_git_sha"],"questionnaire_status":"not_rerun",
                   "questionnaire_participants":q_count,"eye_participants":summary["eye"]["participants"],
                   "eye_trials":summary["eye"]["trials"],"eeg_participants":summary["eeg"]["participants"],
                   "eeg_common_trials":summary["eeg"]["trials"]}]).to_csv(outputs/"realdata_run_summary.csv",index=False,encoding="utf-8-sig")
    dump(outputs/"code_publication.json",{"analysis_git_sha":calculation_sha(run),
         "publication_git_sha":pointer["publication_git_sha"],"run_root":str(run),
         "previous_publication":str(archive/"code_publication.json"),"scope":"eye-eeg; questionnaire not rerun"})
    incremental=outputs/"eye_eeg_incremental_latest.json"
    if incremental.is_file():
        from .incremental_handoff import renew_delivery_pointer
        previous=json.loads(incremental.read_text(encoding="utf-8"))
        dump(run/"previous_eye_eeg_incremental_latest.json",previous)
        history=renew_delivery_pointer(previous,pointer,moved,run)
        dump(incremental,{**pointer,"previous_audit_and_delivery":history,
                          "scope_note":"当前入口为全量重跑；前次增量审计来源及迁移记录保留"})
    (outputs/"README_老师本次EEG交付.md").write_text(
        f"# 老师本次眼动与EEG交付\n\n当前报告：论文数据分析结果报告.md。当前包：{pointer['zip']}。\n\n本轮源数据全量重算，方法不变；问卷独立分析未运行。旧入口文件归档：{archive}。\n",encoding="utf-8")


def publish(config, run, outputs, repo):
    from .request_handoff import archive_publish
    if subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip():
        raise StageBlockedError("Publish from committed clean code")
    manifest=json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete": raise StageBlockedError("Run incomplete")
    from .reproducibility import verify_seal
    reproducibility_path=run/'reproducibility_verification.json'
    if not reproducibility_path.is_file():
        raise StageBlockedError('Three-scope reproducibility evidence is required before publication')
    reproducibility=json.loads(reproducibility_path.read_text(encoding='utf-8'))
    if reproducibility['status']!='passed' or set(reproducibility['scopes'])!={'eye','eeg','joint'} or any(
        v['status']!='passed' for v in reproducibility['scopes'].values()):
        raise StageBlockedError('Eye, EEG and joint reproducibility must each pass')
    verify_seal(reproducibility['inputs']);verify_seal(reproducibility['evidence'])
    verify_inputs(json.loads((run / "source_hashes_before.json").read_text(encoding="utf-8")))
    stage=run / "handoff_staging"
    expected_entries={"论文数据分析结果报告.md","数据来源交接索引.xlsx","数据来源交接说明.docx","filesource_flat"}
    if {p.name for p in stage.iterdir()} != expected_entries:
        raise StageBlockedError("Unexpected delivery entries; retain diagnostics outside staging")
    qa=json.loads((run / "artifact_verification.json").read_text(encoding="utf-8"))
    for name in ["数据来源交接索引.xlsx","数据来源交接说明.docx"]:
        if not qa["files"][name]["visually_reviewed"] or qa["files"][name]["sha256"]!=file_sha256(stage/name):
            raise StageBlockedError("Artifact review missing/stale")
    index_qa=qa["files"]["结果文件总索引.xlsx"]
    if not index_qa["visually_reviewed"] or index_qa["sha256"]!=file_sha256(run/"结果文件总索引.xlsx"):
        raise StageBlockedError("Formal index review missing/stale")
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
    pointer={"status":"complete","scope":"eye-eeg","run_root":str(run),"git_commit":manifest["git_commit"],"publication_git_sha":git_sha(repo),"source_count":len(records),
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
    previous_report=outputs/"论文数据分析结果报告.md"
    if previous_report.is_file():
        historical_report=run/"superseded_formal_results/root_entrypoints/论文数据分析结果报告.md"
        historical_report.parent.mkdir(parents=True,exist_ok=True)
        if historical_report.exists():raise StageBlockedError("Previous formal report already archived")
        shutil.copy2(previous_report,historical_report)
        if file_sha256(previous_report)!=file_sha256(historical_report):raise StageBlockedError("Previous formal report copy differs")
    shutil.copyfile(run/"论文数据分析结果报告.md",outputs/"论文数据分析结果报告.md")
    for folder in ["00_teacher_inputs","00_clock_cache","00_eeg_raw_export","00_eeg_trial_qc","00_synchronized_source","01_eye_stage1","02_eye_stage2","03_eye_stage3_plan","04_eye_stage3","05_eeg_order","06_eeg_primary","07_eeg_window_robustness","08_synchronized_crossmodal","09_eye_figures","10_eeg_denominator_sensitivity","11_historical_comparison","12_final_verification"]:
        target=outputs/"12_teacher_analysis"/folder
        if target.exists():
            historical=run/"superseded_formal_results"/folder;historical.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(target),str(historical))
        # PSD cache and sample waveforms retain one canonical copy in run_root.
        shutil.copytree(run/folder,target,ignore=shutil.ignore_patterns("psd_cache","samples","irregular_timestamps"))
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
    refresh_formal_entrypoints(outputs,run,pointer,moved)
    return pointer
