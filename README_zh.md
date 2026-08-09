# More is not always better 论文级数据分析仓库

本仓库是对原 `More-is-not-always-better` 项目的深度改造，不是另建子项目。现在的目标是支撑论文返修与重投：把问卷、眼动、EEG、EEG+眼动融合、稳健性诊断、图表源数据和审稿意见回应证据整合到同一个可复现分析流水线中。

底层统一使用 `participant_id + scene_id` 作为 canonical trial index。问卷、眼动、EEG、同步 QC、time-bin 融合、统计模型和审稿回应索引都从这张统一试次表派生，避免各脚本各自拼表造成口径不一致。

order1/order2/neworder2 的权威场景顺序、简化眼动文件夹语义、`C0/C1` 复杂度编码及问卷跨模态共同剔除规则，统一见 [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md)。

## 老师规范的一键运行

```powershell
python scripts/run_teacher_analysis.py all-results `
  --config configs/teacher_analysis.local.json `
  --outputs-root "E:\26\补\数据分析结果" `
  --self-review --reuse-valid --resume --promote
```

该命令先写入 `teacher_runs/<run-id>`，只有完整性和数字一致性 QA
通过后才提升到 `12_teacher_analysis`。`--self-review` 允许代码记录并
跳过中间人工审批等待；`--reuse-valid` 只复用输入、配置、方法契约和
关键表结构均验证通过的历史结果；`--resume` 支持断点续跑；
`--promote` 控制正式结果更新。旧的 `scripts/run_all.py` 仅作为兼容
工作流保留。

眼动真实设计固定为 12 个场景和 12 套 AOI，不压缩成 9 套。眼动主
样本按眼动 Stage 1+2 QC 独立确定；EEG 有效样本只用于精确
`Participant × GlobalTrialOrder` 共同试次敏感性。高低经验组统一使用
仓库 Q1.4 规则生成的 `ExperienceGroup`，不使用 ExerciseFrequency。
问卷正式分析使用 EEG 有效的 42 人/504试次，并从最新总问卷工作簿
重新准备；眼动主分析仍使用独立 QC 得到的 A 人样本。
各阶段正式叙述报告采用 Markdown，机器可读表、图片、日志和代码副本
仍按老师清单交付。

老师流程同时生成 `09_eye_figures/`：固定语义配色的六场景 AOI 图、通过
60% tracking QC 的 Fig. 6 fixation 事件密度图、时长加权敏感性图，以及
Block 2 到 Block 1 的配准矩阵、配准 QA 和去标识 source data。已有完整
老师运行也可用 `scripts/build_eye_scene_figures.py` 单独补图，无需重跑
眼动 Stage 2/3 或 EEG 模型。

## 输出结构

- `outputs/01_sample_qc/`：样本流向、补招前后组别平衡、场景/条件平衡。
- `outputs/02_questionnaire/`：S1-S5、B1-B3、IPQ 长表、扩展描述统计、信度诊断、B 题 C1-only QC、IPQ 被试层结果、逐题 LMM 诊断和 WWR 趋势对比。
- `outputs/03_eye_tracking/`：AOI visited、FCR、TFD、TTFF、attention share、AOI 有效性和眼动 QC。
- `outputs/04_eeg/`：EEG trial 级指标和频段 QC。
- `outputs/05_multimodal_fusion/`：论文主分析长表、EEG+眼动场景级融合表、time-bin 融合表、同步 QC、精细对齐 QC、claim support matrix。
- `outputs/06_models/`：注册模型结果、WWR planned contrasts、模型诊断。
- `outputs/06_robustness/`：顺序/疲劳、性别、补招批次、WWR 非线性、功效敏感性分析。
- `outputs/07_paper_tables/`：论文表格、claim strength、figure contracts、source data index。
- `outputs/08_reviewer_response/`：审稿意见到证据文件的回应索引和 reviewer issue matrix。
- `outputs/09_data_package/`：Nature-style 数据可用性索引和 Data Availability 草稿。
- `outputs/10_figures/`：Nature-style SVG/PDF/TIFF/PNG 成图、panel source CSV、figure manifest、图注说明和 QA。
- `teacher_runs/<run-id>/`：每次老师流程的完整可追溯运行归档。
- `12_teacher_analysis/`：最近一次通过 QA 并正式提升的老师流程结果。
- 顶层 `论文数据分析结果报告.md`、`结果文件总索引.xlsx`、
  `老师任务完成矩阵.xlsx` 和 `artifact_reuse_manifest.xlsx`：完整成果包
  的阅读入口与审计入口。

## 问卷方法口径

问卷模块吸收了 `wannaqueen66-create/spss` 的合理思路，但没有整仓照搬。保留的是扩展描述统计、Cronbach alpha 作为内部一致性诊断、B 题 C1-only 处理、IPQ 被试层分析、逐题混合模型、WWR 线性/二次趋势对比。

修正后的口径是：S1-S5 仍是主要逐题结果；`Afford4` 只是 S1-S4 的补充候选构念；`Bmean` 只用于 C1 补充分析；`IPQ_mean` 只在被试层解释，不作为场景级 WWR/Complexity 证据；Shapiro、偏度、峰度只作诊断；三档 WWR 只能支持 trend/planned contrast 表述，不能强称最优点。

## 面向拒稿意见的设计

- 低经验组不平衡：通过 `RecruitmentBatch`、`SupplementFlag` 和 `group_balance_before_after.csv` 明确记录补招与组别平衡。
- 性别未纳入：模型配置中保留 `Gender`，并输出 `gender_sensitivity.csv`。
- 顺序/疲劳效应：所有 trial 保留 `block`、`position`、`round`，并输出 order/fatigue 诊断。
- 三档 WWR 不能强称最优：只输出 trend/planned contrast 证据，claim strength 自动限制为探索性或有界表述。
- AOI 有效性不足：输出 AOI 面积、visited rate、样本覆盖和眼动 QC。
- EEG 解释过强：EEG 结论必须经过同步 QC、眼动/问卷收敛和 `claim_support_matrix.csv` 约束。

## EEG+眼动融合

融合层不是补丁脚本，而是根仓库的数据模型层：

- `analysis_master_long.csv`：问卷 + EEG + 眼动的论文统计主表。
- `aligned_scene_table.csv`：保留原仓库 EEG+AOI 场景级融合逻辑。
- `aligned_timebin_table.csv`：旧兼容表；场景级 EEG 值在眼动 bin 中重复，
  明确标记为 `legacy_not_for_temporal_inference`，不能作为真正时序同步证据。
- `aligned_synchronized_timebin_table.csv`：使用同一绝对时钟窗口计算的
  window-specific EEG 与眼动指标，是正式时序同步分析输入。

历史 EEG 时钟缓存若含有旧版 latin1/GBK 乱码，运行
`scripts/repair_eeg_clock_cache_encoding.py` 在本次运行目录生成 UTF-8
副本；原缓存不会被修改。没有通过时钟缓存验证的参与者会写明排除原因，
不会阻断其他已验证参与者的同步样本导出。
- `sync_qc.csv`：眼动时长、EEG 时长、差值、mismatch、场景数量检查。
- `alignment_scene_qc.csv`、`alignment_landmarks.csv`、`time_sync_map.csv`：眼动时间到 EEG 时间的精细仿射映射诊断。

## Nature-skills 对齐

- `nature-response`：`configs/reviewer_response_map.json`、`docs/REVIEWER_ISSUE_MATRIX.md` 和 `outputs/08_reviewer_response/` 保证每条审稿质疑都有行动和证据文件。
- `nature-writing`：`claim_strength_table.csv` 把论文主张限制在当前证据强度内，避免过度结论。
- `nature-data`：`configs/data_availability.json`、`docs/DATA_AVAILABILITY_DRAFT.md` 和 `outputs/09_data_package/` 明确 raw/processed/source data 的开放或受限路径。
- `nature-figure`：`configs/figure_contracts.json`、`docs/FIGURE_CONTRACTS.md`、`outputs/10_figures/` 为每个论文图建立结论、证据链、panel 源数据、SVG/PDF/TIFF/PNG 成图、图注说明和 QA。
