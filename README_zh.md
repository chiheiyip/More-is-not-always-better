# More is not always better 论文级数据分析仓库

本仓库是对原 `More-is-not-always-better` 项目的深度改造，不是另建子项目。现在的目标是支撑论文返修与重投：把问卷、眼动、EEG、EEG+眼动融合、稳健性诊断、图表源数据和审稿意见回应证据整合到同一个可复现分析流水线中。

底层统一使用 `participant_id + scene_id` 作为 canonical trial index。问卷、眼动、EEG、同步 QC、time-bin 融合、统计模型和审稿回应索引都从这张统一试次表派生，避免各脚本各自拼表造成口径不一致。

## 一键运行

```bash
python scripts/run_all.py --config configs/paths.example.json
```

真实数据运行时，复制 `configs/paths.example.json` 为 `configs/paths.local.json`，并把问卷、眼动、EEG 和输出路径改成本地真实路径。`configs/local_paths.example.json` 只保留安全占位路径，不提交可识别原始数据。

## 输出结构

- `outputs/01_sample_qc/`：样本流向、补招前后组别平衡、场景/条件平衡。
- `outputs/02_questionnaire/`：S1-S5、B1-B3、IPQ 长表、质控和描述统计。
- `outputs/03_eye_tracking/`：AOI visited、FCR、TFD、TTFF、attention share、AOI 有效性和眼动 QC。
- `outputs/04_eeg/`：EEG trial 级指标和频段 QC。
- `outputs/05_multimodal_fusion/`：论文主分析长表、EEG+眼动场景级融合表、time-bin 融合表、同步 QC、精细对齐 QC、claim support matrix。
- `outputs/06_models/`：注册模型结果、WWR planned contrasts、模型诊断。
- `outputs/06_robustness/`：顺序/疲劳、性别、补招批次、WWR 非线性、功效敏感性分析。
- `outputs/07_paper_tables/`：论文表格、claim strength、figure contracts、source data index。
- `outputs/08_reviewer_response/`：审稿意见到证据文件的回应索引和 reviewer issue matrix。
- `outputs/09_data_package/`：Nature-style 数据可用性索引和 Data Availability 草稿。

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
- `aligned_timebin_table.csv`：time-bin 级眼动 AOI 指标，并附加对应场景 EEG 指标。
- `sync_qc.csv`：眼动时长、EEG 时长、差值、mismatch、场景数量检查。
- `alignment_scene_qc.csv`、`alignment_landmarks.csv`、`time_sync_map.csv`：眼动时间到 EEG 时间的精细仿射映射诊断。

## Nature-skills 对齐

- `nature-response`：`configs/reviewer_response_map.json`、`docs/REVIEWER_ISSUE_MATRIX.md` 和 `outputs/08_reviewer_response/` 保证每条审稿质疑都有行动和证据文件。
- `nature-writing`：`claim_strength_table.csv` 把论文主张限制在当前证据强度内，避免过度结论。
- `nature-data`：`configs/data_availability.json`、`docs/DATA_AVAILABILITY_DRAFT.md` 和 `outputs/09_data_package/` 明确 raw/processed/source data 的开放或受限路径。
- `nature-figure`：`configs/figure_contracts.json`、`docs/FIGURE_CONTRACTS.md` 和 `source_data_index.csv` 为每个论文图建立结论、证据链、源数据和导出格式契约。
