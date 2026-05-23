# More Is Not Always Better：EEG + 眼动融合项目

本仓库用于把 EEG 场景级频段功率结果与眼动场景 CSV/AOI 指标合并成一个可直接写论文、建模和出图的独立项目。下面两个仓库只是历史参考，不是运行依赖：

- 眼动：https://github.com/wannaqueen66-create/eyetrack
- EEG：https://github.com/wannaqueen66-create/eeg

本仓库不依赖也不修改上述两个仓库。核心合并键是 `participant_id + scene_id`。

## 对齐原则

每个眼动 CSV 已经是单个场景，因此默认：

```text
eye_aligned_ms = 眼动时间戳 - 该 CSV 最小时间戳 + eye_offset_ms
```

也就是说，每个眼动场景 CSV 的起点对齐到 EEG 中该场景 `view` 段的起点，即 marker `7 -> 8`。

## 主要输出

- `outputs/fusion/aligned_scene_table.csv`：场景级 EEG + 眼动融合主表。
- `outputs/fusion/aligned_timebin_table.csv`：默认 2 秒非重叠 time-bin 的眼动 AOI 指标，并附加对应场景 EEG 指标。
- `outputs/fusion/sync_qc.csv`：同步质量检查，包括 EEG view 时长、眼动 CSV 时长、差值、样本数、bin 数、缺失和 mismatch 标记。

## 端到端原始输入

现在 EEG 原始输入是 `.set/.fdt` 成对文件，默认目录：

```text
E:\eeg原始文件
```

眼动输入是已经按场景拆分的 CSV，默认目录：

```text
E:\2.7眼动数据\映射
```

问卷输入默认目录：

```text
E:\VR+EEG实验问卷-文本版-2026-02-17.xlsx
```

先只检查目录，不跑全量：

```bash
python scripts/build_manifests.py --dry_run
python scripts/run_end_to_end.py --dry_run
```

生成 manifests：

```bash
python scripts/build_manifests.py
```

`build_manifests.py` 会读取问卷里的 `Q1.8_场景顺序编号`，写入 `participants.csv` 的 `Order` 字段，因此不再猜测 counterbalance 顺序。

EEG 端到端导出场景级频段功率：

```bash
matlab -batch "addpath('matlab'); run_eeg_bandpower_from_set('E:/eeg原始文件', 'outputs/eeg'); exit"
```

如果后续发现眼动导出使用了错误被试名，可以用 `--eye_alias_csv` 提供手动别名表。脚本也保留了基于 record id 的通用自动别名机制，用于处理 `User1` 这类泛化标签，但没有任何针对某个被试的硬编码。

如果 `aoi_json_path` 为空，眼动脚本不会中断，会输出 `whole_scene` 级别的基础眼动指标；之后补上 AOI JSON 后会自动输出 AOI class 指标。

## 精准对齐 QC

眼动 CSV 保留的是连续眼动记录的 `Recording Time Stamp[ms]`，不是每个场景重新从 0 开始。因此可以按被试拟合：

```text
eeg_time_ms = time_sync_slope * eye_time_ms + time_sync_offset_ms
```

对齐锚点是每个场景的眼动开始/结束时间，以及 EEG 中 marker `7 -> next 8` 导出的 `view_start_s/view_end_s`。

```bash
python scripts/run_alignment_qc.py --participants manifests/generated/participants.csv --scene_manifest manifests/generated/scene_manifest.csv --eeg_scene_csv outputs/eeg/summary/all_subjects_scene_level.csv --outdir outputs/fusion
```

输出：

- `outputs/fusion/time_sync_map.csv`
- `outputs/fusion/alignment_landmarks.csv`
- `outputs/fusion/alignment_scene_qc.csv`

## 推荐运行

```bash
python -m pip install -r requirements.txt
python scripts/run_eye_aoi_batch.py --participants manifests/participants.csv --scene_manifest manifests/scene_manifest.csv --outdir outputs/eye --dwell_mode fixation
python scripts/run_fusion.py --participants manifests/participants.csv --scene_manifest manifests/scene_manifest.csv --eeg_scene_csv outputs/eeg/summary/all_subjects_scene_level.csv --eye_aoi_class_csv outputs/eye/batch_aoi_metrics_by_class.csv --outdir outputs/fusion
```

## 数据表要求

`manifests/participants.csv`：

```csv
participant_id,eeg_subject_id,eye_subject_id,SportFreq,Experience,Order,exclude
P001,P001,P001,High,Low,1,false
```

`manifests/scene_manifest.csv`：

```csv
participant_id,scene_id,block,position,scene_name,eye_csv_path,aoi_json_path,WWR,Cond,Complexity,eye_offset_ms
P001,1,1,1,scene_01,data/raw/eye/P001_scene01.csv,data/raw/aoi/scene_01_aoi.json,0.2,A,1,0
```

`eye_offset_ms` 用于修正固定同步延迟；没有延迟时填 `0`。
