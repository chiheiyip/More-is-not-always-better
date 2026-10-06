"""Focused incremental-request handoff; large sources and old deliveries stay local."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
import numpy as np
import pandas as pd
from .independent_eye import CORE, KEY, dump, save, sha, truth
from .incremental_audit import load_config, git_sha

SHEETS=['先看这里','段落来源映射','关键数字核对','文件总清单','样本与数据粒度','功率与统计复核','差异与阴性结果','未复制原始数据','术语与字段说明']


def location(config):
    return Path(config['outputs_root'])/'teacher_request_runs'/config['run_id']


def preflight(config,repo):
    root=location(config)
    destination=Path(config['delivery_root']).resolve();archive=Path(config['archive_root']).resolve()
    if destination==archive or destination in archive.parents or archive in destination.parents:
        raise ValueError('Delivery and archive must be separate directories')
    needed=['eye_comparison_summary.json','eeg_comparison_summary.json','source_readers/Python_R_original_design_summary.csv','eye/canonical_calculation_seal.json','EEG_stage_reuse_manifest.json']
    return {'request_type':'eye_eeg_incremental','ICA_special_audit':'cancelled','run_root':str(root),'delivery_root':config['delivery_root'],'archive_root':config['archive_root'],'missing_completed_stages':[name for name in needed if not (root/name).is_file()],'code_sha':git_sha(repo),'staging':str(root/'handoff_staging'),'sheets':SHEETS}


def require_complete(config,repo):
    plan=preflight(config,repo)
    if plan['missing_completed_stages']:raise ValueError('Incremental stages incomplete: '+str(plan['missing_completed_stages']))
    root=location(config)
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('Publish only from clean committed code')
    for filename in ['independent_calculation_seal.json','canonical_calculation_seal.json']:
        for record in json.loads((root/'eye'/filename).read_text(encoding='utf-8'))['files']:
            if sha(record['path'])!=record['sha256']:raise ValueError('Frozen independent calculation changed')
    if pd.read_csv(root/'source_readers/Python_R_original_design_summary.csv').MismatchedRows.any():raise ValueError('Independent source readers disagree')
    return root


def typed(name,frame):
    return {'name':name,'columns':frame.columns.tolist(),'rows':frame.astype(object).where(pd.notna(frame),None).values.tolist()}


def number(value):
    if value is None or pd.isna(value):return '—'
    return f'{float(value):.8g}'


def prepare(config,repo):
    root=require_complete(config,repo);stage=root/'handoff_staging'
    if stage.exists():raise ValueError('Handoff staging already exists; retain drafts or choose a new request run')
    stage.mkdir();flat=stage/'filesource_flat';flat.mkdir()
    eye=root/'eye';eeg=root/'eeg';source=Path(config['sensitivity_run']);prior=Path(config['prior_verification'])
    es=json.loads((root/'eye_comparison_summary.json').read_text(encoding='utf-8'));gs=json.loads((root/'eeg_comparison_summary.json').read_text(encoding='utf-8'))
    power=json.loads((eeg/'supplementary_evidence_summary.json').read_text(encoding='utf-8'))
    processing=json.loads((eye/'processing_summary.json').read_text(encoding='utf-8'))
    manifest=[];by_path={}
    def pack(path,category):
        path=Path(path).resolve()
        if str(path) in by_path:return by_path[str(path)]
        if not path.is_file():raise ValueError('Missing report evidence: '+str(path))
        sid=f'SRC{len(manifest)+1:04d}';name=sid+'__'+category+'__'+path.name
        digest=sha(path);shutil.copyfile(path,flat/name)
        if sha(flat/name)!=digest:raise ValueError('Flat source copy differs')
        record={'SourceID':sid,'source_path':str(path),'flat_name':name,'category':category,'size_bytes':path.stat().st_size,'sha256':digest}
        manifest.append(record);by_path[str(path)]=sid;return sid
    for key_name in ['questionnaire_file','participant_information','trial_order_mapping','scene_aoi_mapping','aoi_approval']:
        pack(config[key_name],'original_design')
    for p in [eye/'source_hashes_before.json',eye/'source_hashes_after.json']:
        pack(p,'original_source_hashes')
    scene_inputs=pd.read_excel(config['scene_aoi_mapping'])
    for column in ['AOIFile','ValidSceneFile','BaseImageFile']:
        for filename in sorted(set(scene_inputs[column].dropna())):pack(filename,'approved_measurement_input')
    for p in sorted(eye.glob('*.csv')):
        if p.name!='independent_fixations.csv':pack(p,'eye_audit')
    for p in sorted(eeg.glob('*.csv')):pack(p,'eeg_audit')
    for p in sorted((eeg/'current_QC_models').glob('*.csv')):pack(p,'eeg_current_QC')
    for p in sorted((eeg/'current_QC_inputs').rglob('*.csv')):pack(p,'eeg_current_input')
    for p in sorted((root/'source_readers').glob('*.csv')):pack(p,'dual_source_reader')
    for p in [root/'eye_comparison_summary.json',root/'eeg_comparison_summary.json',root/'EEG_stage_reuse_manifest.json',root/'request_config.json',eye/'canonical_measurement_contract.json',eye/'processing_summary.json',eeg/'supplementary_evidence_summary.json',eeg/'source_hashes_before.json',eeg/'current_QC_input_manifest.json',prior/'verification_summary.json',prior/'bootstrap_provenance.json',prior/'preprocessing_history_check.csv',source/'manuscript_comparisons.csv',source/'paired_psd_integrals.csv',source/'summary.json',source/'analysis_contract.json']:
        pack(p,'provenance')
    for version in 'ABC':
        for filename in ['family_coefficients.csv','family_factors.csv']:
            pack(source/version/filename,'EEG_'+version)
        for p in sorted((source/version).glob('trim_*/input.csv')):pack(p,'EEG_'+version+'_input')
    for folder in [eeg,eeg/'current_QC_models',root/'canonical_eye_models',root/'eye_models/main']:
        for p in sorted(folder.glob('*environment*.json')):pack(p,'environment')
        for p in sorted(folder.glob('*provenance*.json')):pack(p,'execution')
    for p in config.get('request_files',[]):pack(p,'teacher_request')
    for p in sorted(root.glob('phase_*_provenance.json')):pack(p,'execution')
    for p in sorted((root/'canonical_eye_models').glob('*provenance.json')):pack(p,'execution')
    for p in sorted((root/'eye_models').glob('*/execution_provenance.json')):pack(p,'execution')
    if (root/'eye_revision_lineage.json').is_file():pack(root/'eye_revision_lineage.json','audit_revision')
    initial=Path(config['eeg_reuse_run'])/'eye/processing_summary.json'
    pack(initial,'superseded_audit_explanation')
    lines=[];mapping=[];numbers=[]
    section='本次结论'
    def add(text,paths=(),fields='',condition=''):
        ids=[pack(p,'report_evidence') for p in paths]
        if ids and not fields:fields='完整源表：按本段列明的 Issue、outcome、term、窗口或试次键定位'
        if ids and not condition:condition='本段所述模型、效应、样本或来源记录；解释段使用完整来源'
        quantitative=bool(re.search(r'\d',text))
        suffix='（'+ '；'.join(ids)+'）' if ids and not text.startswith('|') else ''
        if not text.startswith('|') and lines and lines[-1]!='':lines.append('')
        start=len(lines)+1;lines.extend((text+suffix).splitlines())
        if not text.startswith('|'):lines.append('')
        mapping.append({'MD行号':start,'章节':section,'段落或表格行':text,'Source ID':';'.join(ids),'字段':fields,'筛选条件':condition,'定量':quantitative,'说明':'逐项源表支持' if ids else '由前述证据综合归纳'})
    def key(label,value,path,field,filter_text):
        numbers.append({'项目':label,'原值':value,'Source ID':pack(path,'key_number'),'字段':field,'筛选条件':filter_text,'核对方式':'重新读取源表；复制哈希一致'})
    for path,summary in [(root/'eye_comparison_summary.json',es),(root/'eeg_comparison_summary.json',gs),(eeg/'supplementary_evidence_summary.json',power)]:
        for field,value in summary.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):key(field,value,path,field,'JSON scalar')
    add('# 眼动和 EEG 增量核查报告')
    add(f"本轮合并眼动独立审计与 EEG 最终实施核查。历史 1–45 Hz 主分析继续保留，1–40 Hz 为分母敏感性；ICA 专项按研究者要求取消。眼动同口径复核为 {es['independent_participants']} 人／{es['independent_trials']} 试次；EEG 历史共同样本为 42 人／461 试次，当前源文件重新 QC 为 {gs['current_common_participants']} 人／{gs['current_common_trials']} 试次。",[root/'eye_comparison_summary.json',root/'eeg_comparison_summary.json',source/'summary.json'],'counts;provenance.common_trials')
    eye_core_changed=bool(es['RobustBHThresholdCrossings'] or es['WWRHolmThresholdCrossings'] or es['RawPThresholdCrossings'])
    add(('Results 3.2 的同口径复核存在跨阈值差异，必须逐项调查下文列出的效应。' if eye_core_changed else 'Results 3.2 的主效应、CR2 BH 及 WWR Holm 显著性判定未发生翻转；需要补写测量与推断口径说明。')+f"原表与独立复核的非截距 CR2 BH 阳性系数分别为 {es['OriginalRobustNonInterceptPositive']}、{es['IndependentRobustNonInterceptPositive']} 项。",[root/'eye_comparison_summary.json',eye/'12_eye_tracking_independent_replication_audit.csv'],'RobustBHThresholdCrossings;HolmThresholdCrossings;positive_counts')
    add('单独改变 EEG 分母的 B→C 比较没有改变 Results 3.3 的核心解释，但有 10 s 额区 θ 的 Position 联合 q 跨越 0.05，以及一个不显著近零系数变号。当前源 QC 的差异是另一项输入／QC 敏感性；平均参考后的全通道平均高频 QC 具有数值残差问题，不能据此宣称全部 EEG 实施已无保留通过。',[source/'manuscript_comparisons.csv',eeg/'average_reference_QC_signal_audit.csv',eeg/'current_QC_model_comparisons.csv'],'pair;term;joint_q;reference_RMS_ratio')
    section='运行与复用'
    add('## 运行与复用')
    diag=pd.read_csv(eye/'08_model_specification_audit.csv');canonical_diag=pd.read_csv(eye/'canonical_08_model_specification_audit.csv')
    reused=pd.read_csv(eeg/'current_QC_models/current_QC_diagnostics.csv').reuse.str.startswith('D absolute').sum()
    add(f"此前验证的 {gs['prior_models_reused']} 个 R 模型、{gs['prior_spectra_verified']} 个 PSD、{gs['prior_source_files_verified']} 个来源／输出哈希条目复用；新补 48 个 A/B/C 核心模型普通 SE 与诊断，以及当前 QC D/E 的 240 条模型记录，其中 {int(reused)} 个 E 版 absolute 结果按相同输入复用 D。眼动本轮 {len(diag)} 条增量模型记录另加 {len(canonical_diag)} 个同口径主模型／companion 复拟合。",[root/'eeg_comparison_summary.json',eeg/'current_QC_models/current_QC_diagnostics.csv',eye/'08_model_specification_audit.csv',eye/'canonical_08_model_specification_audit.csv'],'model counts;reuse')
    add('Python 和 R 分别读取原始问卷 XLSX、参与者信息和试次映射；全部试次设计、Q1.4 分组和同 Block 前序字段一致。历史完整 PSD 双分母积分以及 207 个模型的独立核验按源哈希复用；本次 MATLAB 从当前 SET/FDT 重建波形与功率，Python 独立核对统计量关系、BH 与 Holm。bootstrap 只做来源核验，没有再次抽样。',[root/'source_readers/Python_R_original_design_summary.csv',prior/'verification_summary.json',eeg/'current_QC_Python_R_BH_checks.csv',eye/'Python_R_coefficient_BH_checks.csv',eye/'Python_R_Holm_checks.csv'])
    add('首次眼动实现于独立计算冻结后才读取正式表；此后发现有效 gaze 坐标和有效采样行筛选遗漏，修复后写入独立的新运行。修正发生在接触原代码／结果之后，不宣称第二次盲法。原第一轮 47 人／467 试次是已被修正的审计实现，不是正式结果，也不能据此指责原分析。PIL 栅格化及 floor 坐标按历史记录同口径独立重建；rounded/cv2 结果作为像素规则敏感性保留。',[initial,eye/'canonical_measurement_contract.json',eye/'processing_summary.json'])
    version={'repository_git_sha':git_sha(repo),'request_run':config['run_id'],'computed_stages':{p.name:json.loads(p.read_text(encoding='utf-8')) for p in root.glob('phase_*_provenance.json')},'R':'4.5.3','Python':'3.12','ICA_special_audit':'cancelled'}
    dump(root/'CODE_VERSION.json',version);pack(root/'CODE_VERSION.json','execution')
    section='眼动逐项核查'
    add('## 眼动逐项核查')
    groups=pd.read_csv(eye/'00_exercise_frequency_grouping_audit.csv');counts=pd.read_csv(eye/'independent_Q1_4_groups.csv').ExerciseFrequency.value_counts()
    add(f"Q1.4 前两档为 Low、后两档为 High；候选样本 Low={int(counts.get('Low',0))}、High={int(counts.get('High',0))}。正式模型中没有分组不一致记录；Q1.5 不进入模型。主分析使用 60% tracking 阈值，未因单个试次无效整人删除。",[eye/'00_exercise_frequency_grouping_audit.csv',root/'source_readers/R_original_Q1_4_groups.csv',eye/'canonical_eye_trials.csv'],'Q1.4Original;ExerciseFrequency;QC60')
    add('| 阈值 | 参与者 | 试次 | 用途 |')
    add('| --- | ---: | ---: | --- |')
    for row in processing['qc']:
        add(f"| {number(100*row['Threshold'])}% | {row['Participants']} | {row['Trials']} | {'主分析' if row['Threshold']==.6 else '敏感性'} |",[eye/'processing_summary.json'],'qc','threshold='+str(row['Threshold']))
        key('眼动阈值试次',row['Trials'],eye/'processing_summary.json','qc.Trials','Threshold='+str(row['Threshold']))
    input_summary=pd.read_csv(eye/'trial_input_comparison_summary.csv')
    add(f"按 Participant×GlobalTrialOrder 配对比较 {len(input_summary)} 个字段。样本身份、设计与分组均单列，不以总 N 相等代替键相等。所有超过容差的指标和模型差异保留在源表。",[eye/'trial_input_comparison.csv',eye/'trial_input_comparison_summary.csv',eye/'12_eye_tracking_independent_replication_audit.csv'],'_merge;Field;MismatchedRows;MaxAbsDifference')
    eye_answers=[
      ('Q1.4 与 Q1.5','Q1.4 四档按前两档／后两档分组，Python/R 与正式候选字段一致；未见 Q1.5 实际进模型','exact agreement','00_exercise_frequency_grouping_audit.csv'),
      ('参与者与试次结构','57 名候选各 12 试次、2×6、三种 OrderGroup；纳入键单独比较','exact agreement','01_participant_trial_structure_audit.csv'),
      ('fixation 去重','在有效 gaze 采样中按 index 去重，每个事件保留一个 duration；原值与坐标范围均可抽查','near-exact agreement','03_fixation_spot_check.csv'),
      ('AOI 坐标及面积','同口径 PIL/floor 与 independent round/cv2 分开；投影 unknown，无法验证球面面积，授权输入尚有人工核对边界','unable to verify','02_AOI_coordinate_area_audit.csv'),
      ('C0 Equipment 与零值','C0 Equipment 保持结构 NA；未访问 TTFF 为 NA；enrichment 不加常数，只在正 share 上取 log','exact agreement','06_zero_structural_NA_audit.csv'),
      ('60% 与 trial-level','模型使用逐试次数据；participant-condition 均值只用于描述统计','exact agreement','canonical_eye_trials.csv'),
      ('主模型','ordered-beta 主比例模型；LMM 竞争／enrichment；CR2 companion 与主似然独立列出','near-exact agreement','canonical_08_model_specification_audit.csv'),
      ('补充模型','Visited 二项、计数 NB + log(valid seconds) offset、TTFF 访问概率＋条件 log1p；失败／奇异保留','minor numerical discrepancy','08_model_specification_audit.csv'),
      ('BH 与 Holm','按系数效应分 Family A/B 各三项；总体因子检验另列。WWR 三对比各 outcome/inference 单独 Holm，Python/R 核对','exact agreement','Python_R_coefficient_BH_checks.csv'),
      ('对比方向','WWR45−15、75−15、75−45；复杂度 C0−C1；interaction 直接检验，不以两组各自显著推断 moderation','near-exact agreement','10_contrast_direction_audit.csv'),
      ('Block 与前序','PositionCentered=Position−3.5，主模型不加 GlobalTrialOrder；previous 仅同 Block，首位置 NA','exact agreement','11_temporal_order_carryover_audit.csv'),
      ('描述统计','先按 participant×WWR×Complexity 在可用 Block 内均值，再作 participant-level Mean/SD/Median/t CI','exact agreement','descriptive_participant_condition_summary.csv'),
      ('敏感性','50/60/70%、Block1/2、±5 与触发后 ±10、random slope、LOPO、composition；不收敛和阴性均保留','minor numerical discrepancy','08_model_specification_audit.csv'),
      ('Results 3.2','逐系数与 Holm 翻转见比较表；同尺度／同推断层才比较；原主表的 primary 与 companion 说明必须保留','substantive discrepancy' if eye_core_changed else 'near-exact agreement','12_eye_tracking_independent_replication_audit.csv')]
    eye_topics=[('1 正式使用 Q1.4 而非 Q1.5',0),('2 Low/High 分组',0),('3 旧 Q1.5 残留',0),('4 participant/trial 结构',1),('5 fixation 去重',2),('6 AOI mapping',3),('7 AOI area',3),('8 C0 Equipment structural NA',4),('9 60% tracking threshold',5),('10 trial-level dataset',5),('11 预设正式模型',6),('12 各模型经验组来源',0),('13 BH-FDR family',8),('14 Holm pairwise',8),('15 Block/position/order',10),('16 previous-scene',10),('17 sensitivities',12),('18 substantive discrepancy',13),('19 Results 3.2 是否改变',13),('20 重跑与正文更新范围',13)]
    eye_answers=[(label,*eye_answers[index][1:]) for label,index in eye_topics]
    eye_answers[8]=(eye_answers[8][0],'按 ValidTrackingRatio≥0.60 且 ValidSceneTFD>0 纳入；逐试次筛选，没有因单个无效试次删除整名参与者','exact agreement','canonical_eye_trials.csv')
    eye_answers[13]=(eye_answers[13][0],'每个 outcome 与推断层的三个 WWR 配对比较分别 Holm；Python/R 独立核查。原主似然 Holm 与补充 CR2 Holm 分开','exact agreement','Python_R_Holm_checks.csv')
    eye_answers[17]=(eye_answers[17][0],'同口径输入 17 字段无超容差差异，180 条主模型/companion 对照全部达到验收容差；主分析未发现 substantive discrepancy','near-exact agreement','12_eye_tracking_independent_replication_audit.csv')
    eye_answers[18]=(eye_answers[18][0],'原始 p、CR2 BH、WWR Holm 均无阈值翻转，原非截距 BH 阳性 14 项准确复现；Results 3.2 核心解释不需要改变','near-exact agreement','12_eye_tracking_independent_replication_audit.csv')
    eye_answers[19]=(eye_answers[19][0],'正式六个主模型与 CR2 companion 已复拟合核对；本次没有需修复重跑的原主模型。正文补充像素/投影口径、补充模型失败记录及独立审计修正说明','near-exact agreement','canonical_08_model_specification_audit.csv')
    for label,answer,status,file in eye_answers:add(f"**{label}**：{answer}。分类：{status}。",[eye/file])
    save(pd.DataFrame([{'Issue':a,'Answer':b,'Classification':c,'Evidence':d} for a,b,c,d in eye_answers]),eye/'eye_final_question_answers.csv');pack(eye/'eye_final_question_answers.csv','eye_final')
    nonconv=diag.loc[~diag.converged.map(truth)]
    add(f"增量模型中 {len(nonconv)} 条未通过收敛检查。它们不当作阴性证据；相应完整 BH 族不能缩小后重算。球面面积因投影未确认没有计算。raw User 写作“预实验”的一名候选以文件名／映射关联，身份标签无法独立证实；这部分 trial 在主 QC 中不纳入，未改变主模型。",[eye/'08_model_specification_audit.csv',eye/'independent_eye_trials.csv',eye/'processing_summary.json'],'converged;SourceIdentityStatus;QC60')
    section='EEG 逐项核查'
    add('## EEG 逐项核查')
    add('历史及复现参数为 band-pass 0.5–40 Hz、notch 49–51 Hz，band-pass→notch→average reference；PSD 输入为已有预处理 SET/FDT。ROI 先时域平均，再作 Welch：F3/F4，P3/Pz/P4，O1/Oz/O2；θ 4–7、α 8–12、β 13–30 Hz。500 Hz 数据用 2 s numeric Hamming window、50% overlap、NFFT=1024；按频率掩码 trapz，无插值。relative 分母 1–45 或 1–40；QC 的历史分母保持原定义，未随 relative 分母改动。',[prior/'preprocessing_history_check.csv',source/'analysis_contract.json',eeg/'independent_eeg_source_inventory.csv'])
    add('ICA 专项取消。本轮只保留研究者提供的操作事实：EEGLAB GUI Extended Infomax 运行后未进一步选择或删除成分。运行分解不等于 artifact removal，Methods 不应声称已用 ICA 删除眼动／肌肉伪迹；本段不是新做的逐人 ICA 验证。',[root/'request_config.json'])
    add(f"当前 SET/FDT 独立重建的四窗口 QC 与历史有 {gs['current_vs_historical_QC_differences']} 条纳入差异；历史特征重新计算 QC 则完全一致，且历史共同样本键准确恢复 42／461。当前共同样本为 {gs['current_common_participants']}／{gs['current_common_trials']}，没有为了匹配旧 N 改阈值或补样本。",[eeg/'current_source_vs_historical_QC.csv',eeg/'historical_feature_QC_independent_check.csv',eeg/'historical_common_QC_identity.csv'],'InclusionMatch;HistoricalQCMatch;_merge')
    add('| onset-trim 秒 | 当前参与者 | 当前试次 |')
    add('| ---: | ---: | ---: |')
    for row in gs['current_variant_counts']:add(f"| {row['onset_trim_s']} | {row['participants']} | {row['trials']} |",[eeg/'independent_eeg_trial_QC.csv'],'QCIncluded',f"onset_trim_s={row['onset_trim_s']}")
    add(f"波形→PSD 功率比较中，边界未变化的试次全部通过 rtol=1e−10、atol=1e−12；一个历史试次的开始事件变化 0.166 s，产生 {power['changed_endpoint_power_records']} 条窗口×ROI×频带相对功率差异。该差异属于 epoch 输入变化，不是分母效应。",[eeg/'waveform_power_unchanged_endpoint_checks.csv',eeg/'waveform_power_endpoint_differences.csv'],'EndpointSame;CurrentStartSample;FrozenStartSample;Difference')
    raw=pd.read_csv(eeg/'raw_to_SET_source_identity.csv');epochs=pd.read_csv(eeg/'raw_to_SET_epoch_endpoint_matching.csv')
    add(f"发现并独立读取 {int(raw.RawSourceAvailable.sum())} 名参与者的原始 EASY；{int((~raw.RawSourceAvailable).sum())} 名仍缺少已验证来源。可读取者的记录数与 SET/INFO 核对，{int(epochs.EndpointMatchStatus.eq('both').sum())} 个场景对的原始起止样本完全相同。{power['nonadjacent_SET_epochs']} 个 SET 对没有相邻原始 7→8 配对，其中 {power['nonadjacent_original_endpoints_present']} 个双端点实际存在、仅中间出现额外触发；其余端点需要补充事件编辑证据。不能用事件 ordinal 连锁错位当作场景错误。",[eeg/'raw_to_SET_source_identity.csv',eeg/'raw_to_SET_epoch_endpoint_matching.csv',eeg/'raw_SET_nonadjacent_endpoint_investigation.csv'])
    add(f"历史 HF QC 对全部平均参考通道再作时域平均。{power['reference_residual_spot_checks']} 个波形抽查的“全通道均值 RMS／通道 RMS”最高仅 {number(power['reference_residual_max_ratio'])}，提示该高频比值主要依赖近零数值残差，不能视为已验证的生理高频伪迹指标。当前／历史特征差异集中于少量来源记录，使全样本 robust 阈值移动；一名历史参与者由 3 个坏试次变为 4 个，从而触发 >30% 整人排除。此科学 QC 局限需单独确定后续指标方案，本轮没有擅改。",[eeg/'average_reference_QC_signal_audit.csv',eeg/'current_source_vs_historical_QC.csv',eeg/'historical_feature_QC_thresholds.csv',eeg/'independent_eeg_QC_thresholds.csv'])
    section='分母与正文结果对照'
    add('## 分母与正文结果对照')
    add('A 为历史 1–45；B 为同一历史冻结边界上的复现 1–45；C 为 B 的同一 PSD 改用 1–40。B→C 判断纯分母效应，A→B 记录输入复现差异，A→C 对照原稿。D/E 采用当前源事件与当前共同 QC，分别为 1–45／1–40，属于额外输入／QC 敏感性，不替换 A/B/C。',[source/'summary.json',eeg/'current_QC_input_manifest.json'])
    powers=pd.read_csv(source/'paired_psd_integrals.csv')
    tail=100*powers.power_40_45/powers.total_1_45;reduction=100*(powers.total_1_45-powers.total_1_40)/powers.total_1_45
    add(f"0.5–40 Hz 滤波没有把 40–45 Hz 的离散 PSD 功率强制设为零；1–45 分母仍纳入相应频率点。{len(powers)} 条试次×ROI×窗口记录中，40–45 功率占比均值 {number(tail.mean())}%、中位数 {number(tail.median())}%、范围 {number(tail.min())}–{number(tail.max())}%；实际分母减少比例均值 {number(reduction.mean())}%。两者因离散频率掩码边缘的积分跨段不同而不等同；C 按每条 PSD 计算，没有统一乘平均比例。",[source/'paired_psd_integrals.csv'],'power_40_45;total_1_45;total_1_40','all 5532 spectra; formulas in Excel')
    for label,value,formula in [('40–45占比均值%',tail.mean(),'mean(100*power_40_45/total_1_45)'),('40–45占比中位数%',tail.median(),'median(100*power_40_45/total_1_45)'),('实际分母减少均值%',reduction.mean(),'mean(100*(total_1_45-total_1_40)/total_1_45)')]:key(label,float(value),source/'paired_psd_integrals.csv',formula,'all spectra')
    paired=pd.read_csv(source/'manuscript_comparisons.csv');focus=paired.loc[paired.pair.eq('B→C')&paired.manuscript_claim.isin(['表6_额区theta_WWR45×复杂度','枕区theta_复杂度边际效应','顶区beta_WWR总体效应'])]
    add('| 正文项目 | 窗口 s | B β／F | C β／F | B p | C p | B 联合 q | C 联合 q |')
    add('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in focus.itertuples():
        beta=row.Fstat_from if row.test_kind=='factor' and pd.isna(row.estimate_from) else row.estimate_from
        to=row.Fstat_to if row.test_kind=='factor' and pd.isna(row.estimate_to) else row.estimate_to
        record=focus.loc[row.Index]
        add(f"| {row.manuscript_claim} | {row.onset_trim_s} | {number(beta)} | {number(to)} | {number(record['p.value_from'])} | {number(record['p.value_to'])} | {number(row.joint_q_from)} | {number(row.joint_q_to)} |",[source/'manuscript_comparisons.csv'],'estimate/Fstat;p.value;joint_q',f"pair=B→C;claim={row.manuscript_claim};trim={row.onset_trim_s}")
        for field in ['estimate_from','estimate_to','Fstat_from','Fstat_to','p.value_from','p.value_to','joint_q_from','joint_q_to']:
            if pd.notna(record[field]):key(row.manuscript_claim,float(record[field]),source/'manuscript_comparisons.csv',field,f"pair=B→C;trim={row.onset_trim_s};term={row.term};family={row.family_id}")
    boundary=paired.loc[paired.pair.eq('B→C')&paired.onset_trim_s.eq(10)&paired.outcome.eq('F_theta_relative')&paired.term.eq('PositionWithinBlockCentered')]
    hit=boundary.loc[boundary.family_id.eq('temporal_relative')].iloc[0]
    add(f"明确的边界变化：10 s F_theta_relative 的 Position 联合 q 从 {number(hit.joint_q_from)} 变为 {number(hit.joint_q_to)}。5 s O_theta_relative Block1 WWR45 发生近零变号，但 p≈1；不能扩大为实质方向反转。alpha 的 Block 增加、relative theta 的 Block 降低、P alpha Position 以及上一场景结论逐项源表比较，不写 all results were unchanged。",[source/'manuscript_comparisons.csv'])
    currents=pd.read_csv(eeg/'current_QC_model_comparisons.csv')
    counts=current_flip_summary(currents)
    add(f"额外当前 QC 比较：B→D 的联合 q 翻转 {counts.get('B→D_current_QC_1_45',0)} 项，C→E {counts.get('C→E_current_QC_1_40',0)} 项，D→E {counts.get('D→E',0)} 项；factor-level 联合 q 翻转单列在完整比较。当前 10 s 额区 θ Position 的 q 改变不与历史 B→C 的纯分母边界混为一谈。",[eeg/'current_QC_model_comparisons.csv'],'pair;joint_q_crosses_005')
    add('factor-level 使用重新实现并独立核对的等权边际 CR2/HTZ：六种主效应／二阶交互，保存对比矩阵。历史原 factor 脚本来源缺口仍在，不推断老师创造或修改了数据。previous-scene 实际是 PreviousWWR + PreviousComplexity；论文若写交互项需按实际公式修订。',[source/'analysis_contract.json',eeg/'current_QC_models/current_QC_contrast_matrices.csv'])
    add('各版本独立、完整校正：系数核心 144、扩展 relative/absolute 各 324；factor 核心 96、扩展各 216；temporal 各 72；previous 核心 48。窗口内及四窗口联合 q 分开，缺项／无效推断不给缩小后的最终 q。current D/E 的 Python/R BH 和系数—CR2 SE—df—p 关系已核对。',[eeg/'current_QC_Python_R_BH_checks.csv',source/'analysis_contract.json'])
    add('历史 participant-cluster bootstrap 为 source-verified historical bootstrap：5000 次、seed=20260726，整人重采样并带入全部 trial、重命名重复抽取者 cluster。适用的旧主输入是 474 试次，不是冻结共同 461，也不是当前 453；它没有被当成本轮新 QC 样本的 bootstrap。',[prior/'bootstrap_provenance.json'])
    eeg_answers=[('preprocessing','0.5–40、49–51、band-pass→notch、average reference；见逐人历史证据','near-exact agreement','prior_result_reuse_hash_checks.csv'),('ICA','专项取消；GUI 分解且未进一步选删的研究者说明，不宣称新逐人验证','unable to verify','execution_provenance.json'),('onset/QC/common identity','历史特征准确恢复 42/461；当前源 QC 41/453 是另一个输入状态','substantive discrepancy','current_source_vs_historical_QC.csv'),('ROI/频段/PSD','三个 ROI 时域均值；Welch 与 mask+trapz；未改端点者功率一致','near-exact agreement','waveform_power_unchanged_endpoint_checks.csv'),('1–45/1–40','同 PSD／分子，改两种分母；40–45 与减少比例分开','exact agreement','waveform_to_B_C_power_checks.csv'),('Q1.4','原始 XLSX 双路分组，与冻结字段核对；Q1.5 不入模型','exact agreement','prior_result_reuse_hash_checks.csv'),('主模型/CR2','三种二阶交互，Participant cluster；新增普通 SE/CR2 SE/df/CI','near-exact agreement','core_details_regression.csv'),('bootstrap','历史整人 cluster、5000、seed 20260726、旧 474 样本，未再运行','near-exact agreement','prior_result_reuse_hash_checks.csv'),('BH/factor','完整 144/96/216/72/48 各族；等权 CR2/HTZ 为重新实现','near-exact agreement','current_QC_Python_R_BH_checks.csv'),('temporal/previous','位置边界 q 翻转明确保留；仅 Block 内前序，实际前序为加性公式','minor numerical discrepancy','current_QC_model_comparisons.csv'),('Results 3.3','分母核心解释未变；原始端点来源与近零 HF QC 局限必须补充，不能宣称全部实施无保留通过','unable to verify','average_reference_QC_signal_audit.csv')]
    eeg_topics=[('1 preprocessing',0),('2 ICA component removal（取消）',1),('3 onset-trim/QC/common-QC',2),('4 ROI 与频段',3),('5 1–45 denominator',4),('6 1–40 sensitivity',4),('7 Q1.4 分组',5),('8 主模型',6),('9 participant-cluster CR2',6),('10 participant-cluster bootstrap',7),('11 BH family',8),('12 factor-level family',8),('13 temporal effects',9),('14 previous-scene',9),('15 分母是否改变 Results 3.3',10),('16 substantive discrepancy 与重写范围',10)]
    eeg_answers=[(label,*eeg_answers[index][1:]) for label,index in eeg_topics]
    eeg_answers[0]=(*eeg_answers[0][:3],str(prior/'preprocessing_history_check.csv'))
    eeg_answers[9]=(*eeg_answers[9][:3],str(prior/'bootstrap_provenance.json'))
    for index in [4,5]:eeg_answers[index]=(eeg_answers[index][0],'同一 PSD 和 θ/α/β 分子，分母按各条积分改变；全量双路核验按来源哈希复用；40–45 区间与实际减少比例独立登记','exact agreement',str(source/'paired_psd_integrals.csv'))
    eeg_answers[14]=(eeg_answers[14][0],'单纯 B→C 分母变化不改变 Results 3.3 核心解释；10 s F theta Position q=0.050659201→0.049902815，另有不显著近零变号，均需具体报告','minor numerical discrepancy',str(source/'manuscript_comparisons.csv'))
    eeg_answers[15]=(eeg_answers[15][0],'当前源 QC 41/453 与历史 42/461 有实质样本差异。平均参考后全通道均值 HF QC 接近数值残差，需先确定科学 QC 指标，再复算受影响的全部 EEG 模型及完整校正族，才能确认这一 QC 修订对 Results 3.3 的影响；本轮 D/E 仅量化现有 QC 重建，不等同于解决该问题','substantive discrepancy','average_reference_QC_signal_audit.csv')
    for label,answer,status,file in eeg_answers:add(f"**{label}**：{answer}。分类：{status}。",[eeg/file])
    save(pd.DataFrame([{'Issue':a,'Answer':b,'Classification':c,'Evidence':d} for a,b,c,d in eeg_answers]),eeg/'EEG_final_independent_audit.csv');pack(eeg/'EEG_final_independent_audit.csv','eeg_final')
    section='论文修改与交付边界'
    add('## 论文修改与交付边界')
    add('Results 3.2 保留主似然与 CR2 companion 两个层面的说明，按逐项复核表更新有差异的数值或跨阈值效应；PIL/floor 为历史口径，round/cv2 与 ±5/±10 为测量敏感性。随机斜率／补充模型未收敛时不得归入“全部模型成功”。原始采集标签、投影及人工 AOI/ValidScene 来源不足均明确保留。',[eye/'12_eye_tracking_independent_replication_audit.csv',eye/'08_model_specification_audit.csv',eye/'canonical_measurement_contract.json'])
    add('Results 3.3 不因单纯换分母重写核心解释；补充精确 q 边界、当前 QC/事件来源差异。Methods 需按实际写 Hz、ROI、Welch、前序加性公式以及 ICA 分解操作，删去无证据的 artifact-removal 断言。HF QC 的近零信号问题需另行确定指标与相应重算方案；本轮既没有替换历史主分析，也没有将这一问题视为已解决。',[eeg/'EEG_final_independent_audit.csv',eeg/'average_reference_QC_signal_audit.csv',source/'analysis_contract.json'])
    report='\n'.join(lines).rstrip()+'\n';(root/'论文数据分析结果报告.md').write_text(report,encoding='utf-8');shutil.copyfile(root/'论文数据分析结果报告.md',stage/'论文数据分析结果报告.md')
    map_frame=pd.DataFrame(mapping);save(map_frame,root/'paragraph_source_map.csv');num_frame=pd.DataFrame(numbers);save(num_frame,root/'key_numbers.csv')
    pack(root/'paragraph_source_map.csv','report_mapping');pack(root/'key_numbers.csv','report_numbers')
    dump(root/'source_manifest.json',manifest)
    filelist=pd.DataFrame([{'Source ID':r['SourceID'],'阶段':r['category'],'原始绝对路径':r['source_path'],'平铺文件名':r['flat_name'],'大小字节':r['size_bytes'],'SHA256原与副本':r['sha256'],'打开文件':f'=HYPERLINK("filesource_flat/{r["flat_name"]}","{r["SourceID"]}")'} for r in manifest])
    overview=pd.DataFrame([('本次范围','眼动独立审计与 EEG 增量核查；ICA 专项取消'),('阅读顺序','MD 逐项结论 → 段落来源映射 → filesource_flat'),('历史 EEG','1–45 主结果；1–40 敏感性；保留 A/B/C'),('当前 EEG QC','D/E 41/453；原历史 42/461，不能混用'),('重要局限','平均参考后全通道均值 HF QC 接近数值残差；原始端点来源仍有缺口'),('审计修正','第一轮有效 gaze 定义遗漏已修复并保留；修正后不声称盲法'),('主比例推断','ordered-beta likelihood 与 Gaussian CR2 companion 分开'),('代码 SHA',git_sha(repo)),('运行环境','R 4.5.3；lme4 2.0.1、clubSandwich 0.7.0；各阶段环境见来源'),('参与者标识','按研究者此前约定原样保留，个体文件按研究数据管理保存')],columns=['项目','说明'])
    samples=pd.DataFrame([('眼动候选',57,684,'原始来源审计'),('眼动主分析',es['independent_participants'],es['independent_trials'],'60% tracking，同口径 trial-level'),('EEG A/B/C',42,461,'冻结四窗口共同样本'),('EEG D/E',gs['current_common_participants'],gs['current_common_trials'],'当前源事件、当前四窗口共同 QC'),('历史 EEG bootstrap',42,474,'旧单窗口主输入；不复用到当前 QC inference')],columns=['分析层','participants','trials','实际口径'])
    integrals=pd.read_csv(source/'paired_psd_integrals.csv');spots=integrals.groupby('onset_trim_s',group_keys=False).head(3).reset_index(drop=True)
    calc=spots[['Participant','GlobalTrialOrder','onset_trim_s','roi','total_1_45','total_1_40','power_40_45']].copy()
    calc['40–45占比%']=[f'=100*G{i}/E{i}' for i in range(2,len(calc)+2)];calc['实际分母减少%']=[f'=100*(E{i}-F{i})/E{i}' for i in range(2,len(calc)+2)];calc['relative增加%']=[f'=100*(E{i}/F{i}-1)' for i in range(2,len(calc)+2)];calc['Source ID']=pack(source/'paired_psd_integrals.csv','PSD_integrals')
    changes=currents.loc[currents.joint_q_crosses_005.fillna(False),['pair','onset_trim_s','outcome','term','family_id','joint_q_from','joint_q_to']].copy();changes['说明']='当前 QC/epoch 敏感性，非历史纯分母比较'
    failures=nonconv[['variant','outcome','model_kind','participants','trials','warnings','error']].copy();failures.columns=['pair','outcome','term','participants','trials','说明','原因'];changes=pd.concat([changes,failures],ignore_index=True)
    omitted=[]
    for r in json.loads((eye/'source_hashes_before.json').read_text(encoding='utf-8')):
        if Path(r['path']).suffix.lower()=='.csv':omitted.append({'资料':'原始眼动 CSV','来源入口':r['path'],'大小字节':r['bytes'],'SHA256':r['sha256'],'说明':'原始逐采样数据不重复交付；冻结 fixation 抽查、trial/model 输入与全部来源哈希已保留'})
    for r in json.loads((eeg/'source_hashes_before.json').read_text(encoding='utf-8')):omitted.append({'资料':'SET/FDT','来源入口':r['path'],'大小字节':r['bytes'],'SHA256':r['sha256'],'说明':'大型波形不重复交付；原数据只读'})
    for r in pd.read_csv(eeg/'independent_raw_acquisition_audit.csv').to_dict('records'):
        p=Path(r['SourceEASY']);omitted.append({'资料':'原始 EASY','来源入口':str(p),'大小字节':p.stat().st_size if p.exists() else None,'SHA256':r['SHA256'],'说明':'原始采集留本地，逐人审计表已交付'})
    for r in json.loads((prior/'input_hashes_after.json').read_text(encoding='utf-8')):
        if Path(r['path']).suffix.lower()=='.mat':omitted.append({'资料':'完整 PSD 缓存','来源入口':r['path'],'大小字节':r.get('size_bytes'),'SHA256':r['sha256'],'说明':'完整 f/pxx 大型缓存复用并核验，不重复打包'})
    glossary=pd.DataFrame([('Participant×GlobalTrialOrder','试次键；模型不先平均两个 Block'),('ExerciseFrequency','Q1.4 前两档 Low、后两档 High；Q1.5 不入模型'),('Ordered beta','主比例模型允许 0/1；Gaussian companion 是独立推断层'),('CR2','Participant cluster；系数 Satterthwaite，SE 与估计来自同一模型'),('HTZ','等权边际对比矩阵的总体 Wald F；多自由度不填单一 β'),('BH/Holm','Family A/B 各 effect 三 outcome；WWR 各 outcome 三 pairwise Holm'),('Within q/Joint q','EEG 窗口内 BH／四窗口联合 BH；各数据版本及族分开'),('40–45占比','独立区间积分／1–45 total；与分母减少比例不等同'),('零值/NA','未访问为 NA TTFF，C0 Equipment 结构 NA；不加 epsilon'),('Previous scene','仅同 Block 前序；首位置为 NA；EEG 实际加性公式'),('Source-verified bootstrap','历史整人抽样来源核验，未重跑、未移用于新 QC 样本'),('无法验证','资料／规范缺口，不能当作通过或不显著')],columns=['术语','说明'])
    tables=[typed(SHEETS[0],overview),typed(SHEETS[1],map_frame),typed(SHEETS[2],num_frame),typed(SHEETS[3],filelist),typed(SHEETS[4],samples),typed(SHEETS[5],calc),typed(SHEETS[6],changes),typed(SHEETS[7],pd.DataFrame(omitted)),typed(SHEETS[8],glossary)]
    dump(root/'handoff_tables.json',{'tables':tables})
    guide=['# 眼动和 EEG 本次核查交接说明','','本次材料只包含老师新发眼动与 EEG 核查，以及判断本轮结论所需的旧分母敏感性摘要。先读论文数据分析结果报告.md，再通过 Excel 的 Source ID 打开 filesource_flat 中的原始字节源表。','','## 主要发现','',f"眼动主样本 {es['independent_participants']} 人／{es['independent_trials']} 试次。Q1.4 与设计字段 Python/R 双路一致，主 ordered-beta 和 CR2 companion 分别核对。显著性翻转：原始 p {es['RawPThresholdCrossings']}、CR2 BH {es['RobustBHThresholdCrossings']}、WWR Holm {es['WWRHolmThresholdCrossings']} 项；逐项结果见 MD。",'', '历史 EEG 分母敏感性的核心解释不变，但有已记录的 q 边界与不显著近零变号。当前源 QC 是 41 人／453 试次，历史为 42／461；两种输入状态分开。高频 QC 的全通道均值在平均参考后接近零，需另行确定科学 QC 方案。原始事件端点及投影／人工 AOI 来源缺口均保留。','','## 使用和统计口径','','九张 Excel 表提供段落、表格行、字段、筛选与原值。大波形和 PSD 不重复打包，路径及 SHA256 登记在未复制原始数据。ZIP 解压后保持 filesource_flat 的相对位置，链接可离线打开。','','Primary likelihood、CR2 companion、系数 BH、factor HTZ 与 WWR Holm 的模型和校正族不同，不能拼接估计与另一个模型的 p。失败、奇异、无法验证均与不显著分开；没有按显著结果筛选交付。','','## 来源和后续事项','','第一轮独立眼动结果冻结后发现审计实现遗漏，修复后的运行明确标为接触原结果后的修订。历史已核验 EEG、PSD 和 bootstrap 原始计算版本保留，不将复用描述为新计算。ICA 专项取消，仅记录研究者 GUI 操作说明。','','factor-level 是重新实现，历史脚本来源缺口仍在。EEG previous-scene 实际是加性公式，稿件如写交互项需改。独立核查不以结论必须不变为验收条件；平均参考后 HF QC 与缺少原始端点证据尚需明确。','','个体级源表含参与者标识，按此前约定保留并按研究数据管理保存。']
    (root/'handoff_guide.md').write_text('\n'.join(guide)+'\n',encoding='utf-8')
    dump(root/'handoff_prepared.json',{'source_count':len(manifest),'report_sha256':sha(stage/'论文数据分析结果报告.md'),'code_sha':git_sha(repo),'eye_core_threshold_changes':eye_core_changed,'staging':str(stage)})
    return stage


def current_flip_summary(frame):
    return {key:int(g.joint_q_crosses_005.fillna(False).sum()) for key,g in frame.groupby('pair')}


def document(config,repo,maker):
    root=require_complete(config,repo);path=root/'handoff_staging/数据来源交接说明.docx'
    maker((root/'handoff_guide.md').read_text(encoding='utf-8'),path)
    from docx import Document
    doc=Document(path);doc.core_properties.title='眼动与 EEG 增量核查交接说明';doc.save(path)


def renew_delivery_pointer(value,pointer,relocated,root):
    previous=dict(value)
    for field in ['zip','delivery_zip']:
        if field in previous:
            normalized=str(Path(previous[field]))
            previous[field]=relocated.get(normalized,previous[field])
    value.update(previous_delivery=previous,delivery_root=pointer['delivery_root'],
                 delivery_zip=pointer['zip'],zip=pointer['zip'],zip_sha256=pointer['zip_sha256'],
                 current_delivery_source_count=pointer['source_count'],
                 publication_git_sha=pointer['publication_git_sha'],
                 current_incremental_request=str(root),archive_mapping=str(Path(root)/'archive_path_mapping.json'))
    return value


def refresh_publication_pointers(config,repo):
    root=require_complete(config,repo)
    publication=json.loads((root/'publication.json').read_text(encoding='utf-8'))
    relocated=json.loads((root/'archive_path_mapping.json').read_text(encoding='utf-8'))
    for filename in ['eeg_request_handoff_latest.json','eeg_denominator_sensitivity_latest.json']:
        path=Path(config['outputs_root'])/filename
        value=json.loads(path.read_text(encoding='utf-8'))
        previous=value.get('previous_delivery',{})
        previous,status=verify_legacy_zip_reference(previous,relocated,publication['archive_root'])
        value['previous_delivery']=previous;value['legacy_zip_reference_verification']=status
        value['zip']=publication['zip'];value['zip_sha256']=publication['zip_sha256']
        value['delivery_zip']=publication['zip'];value['pointer_refresh_git_sha']=git_sha(repo)
        dump(path,value)
    dump(root/'publication_pointer_refresh.json',{'code_sha':git_sha(repo),'publication_sha':publication['publication_git_sha'],'scope':'Normalize legacy paths and verify archived ZIP references by recorded hash; missing historical references explicitly unverified. Current ZIP and validated artifacts unchanged'})


def verify_legacy_zip_reference(previous,relocated,archive):
    previous=dict(previous)
    for field in ['zip','delivery_zip']:
        if field in previous:
            normalized=str(Path(previous[field]));previous[field]=relocated.get(normalized,previous[field])
    reference=previous.get('zip',previous.get('delivery_zip'))
    expected=previous.get('zip_sha256')
    if reference and Path(reference).is_file() and (not expected or sha(reference)==expected):
        return previous,{'state':'verified_archived_path','recorded_hash_verified':bool(expected)}
    candidates=[{'path':str(p),'sha256':sha(p)} for p in Path(archive).rglob('*.zip')]
    matches=[r for r in candidates if expected and r['sha256']==expected]
    if len(matches)==1:
        for field in ['zip','delivery_zip']:
            if field in previous:previous[field]=matches[0]['path']
        return previous,{'state':'verified_by_recorded_hash','recorded_hash_verified':True}
    return previous,{'state':'unable_to_verify_historical_ZIP_reference','recorded_reference':reference,'recorded_sha256':expected,'actual_archived_zip_candidates':candidates,'note':'Actual previous-directory contents archived byte-for-byte; historical pointer path/hash is not proof of an unavailable ZIP. No guessed redirection.'}


def publish(config,repo):
    from .request_handoff import archive_publish
    root=require_complete(config,repo);stage=root/'handoff_staging'
    expected={'论文数据分析结果报告.md','数据来源交接索引.xlsx','数据来源交接说明.docx','filesource_flat'}
    if {p.name for p in stage.iterdir()}!=expected:raise ValueError('Current request staging has unexpected entries')
    qa=json.loads((root/'artifact_verification.json').read_text(encoding='utf-8'))
    for name in ['数据来源交接索引.xlsx','数据来源交接说明.docx']:
        if not qa['files'][name]['visually_reviewed'] or sha(stage/name)!=qa['files'][name]['sha256']:raise ValueError('Artifact QA missing or stale')
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    for record in manifest:
        if sha(record['source_path'])!=record['sha256'] or sha(stage/'filesource_flat'/record['flat_name'])!=record['sha256']:raise ValueError('Packaged source changed')
    if sha(root/'论文数据分析结果报告.md')!=sha(stage/'论文数据分析结果报告.md'):raise ValueError('MD is not a byte-identical copy')
    records=[{'path':p.relative_to(stage).as_posix(),'sha256':sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    dump(stage/'filesource_flat/交付文件哈希.json',records)
    name=f"结果发送_完整交接包_眼动_EEG增量核查_{config['stamp']}.zip";archive_file=stage/name
    with zipfile.ZipFile(archive_file,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file() and p!=archive_file:z.write(p,p.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive_file) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity failure')
        for record in records:
            if hashlib.sha256(z.read(record['path'])).hexdigest()!=record['sha256']:raise ValueError('ZIP file bytes differ')
    destination=Path(config['delivery_root']);archive=Path(config['archive_root'])/config['run_id']
    old={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    def validate(current,historical):
        for rel,digest in old.items():
            if sha(historical/rel)!=digest:raise ValueError('Old delivery archive differs')
        for record in records:
            if sha(current/record['path'])!=record['sha256']:raise ValueError('Published file differs')
    relocated=archive_publish(stage,destination,archive,validate);dump(root/'archive_path_mapping.json',relocated)
    pointer={'status':'complete_with_documented_limitations','run_root':str(root),'delivery_root':str(destination),'zip':str(destination/name),'zip_sha256':sha(destination/name),'archive_root':str(archive),'publication_git_sha':git_sha(repo),'source_count':len(manifest),'ICA_special_audit':'cancelled','historical_main':'A 1–45 Hz','denominator_sensitivity':'B→C 1–45→1–40','current_source_QC_sensitivity':'D/E 41/453','unresolved':['average-reference all-channel-mean HF QC','four missing raw sources and unmatched event endpoints','projection/AOI human confirmation','historical factor script provenance']}
    dump(root/'publication.json',pointer);outputs=Path(config['outputs_root'])
    dump(outputs/'eye_eeg_incremental_latest.json',pointer);shutil.copyfile(root/'论文数据分析结果报告.md',outputs/'老师本次眼动与EEG核查报告.md')
    summary_path=outputs/'realdata_run_summary.json';summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary.setdefault('teacher_delivery_history',[]).append(summary.get('teacher_delivery',{}));summary['teacher_delivery']={**pointer,'package':str(destination)};summary['current_teacher_request']=pointer;dump(summary_path,summary)
    for filename in ['eeg_request_handoff_latest.json','eeg_denominator_sensitivity_latest.json']:
        p=outputs/filename
        if p.exists():
            value=json.loads(p.read_text(encoding='utf-8'));dump(p,renew_delivery_pointer(value,pointer,relocated,root))
    (outputs/'README_老师本次EEG交付.md').write_text('当前交付已合并为眼动与 EEG 增量核查。请使用 eye_eeg_incremental_latest.json 和 老师本次眼动与EEG核查报告.md；历史入口保留原计算来源，旧交付位置见 archive_path_mapping.json。\n',encoding='utf-8')
    (outputs/'README_当前有效结果.md').write_text(f"# 当前有效数据分析结果\n\n完整多模态正式运行 teacher_latest_20261003 保留。\n\n本次眼动与 EEG 入口：老师本次眼动与EEG核查报告.md；eye_eeg_incremental_latest.json。核查运行：{root}。\n\n当前老师交付：{destination}；旧交付完整保存在 {archive}。\n\n历史 EEG 1–45 为主分析，1–40 为敏感性；当前源 QC D/E 独立列出。HF QC 与源事件缺口见本次报告，不视为已解决。\n",encoding='utf-8')
    return pointer
