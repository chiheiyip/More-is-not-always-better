# More Is Not Always Better

Independent fusion pipeline for scene-level and time-bin aligned EEG + eye-tracking analysis.

This repository runs independently. The earlier repositories are historical references only, not runtime dependencies:

- Eye-tracking source reference: https://github.com/wannaqueen66-create/eyetrack
- EEG source reference: https://github.com/wannaqueen66-create/eeg

The main fusion key is `participant_id + scene_id`. Each eye-tracking CSV is assumed to contain one scene only. Its minimum recording timestamp is treated as scene `t=0` and aligned to the EEG `view` segment start, corresponding to marker transition `7 -> 8` in the EEG pipeline.

## Project Layout

```text
configs/
  columns_default.json          Eye-tracking column aliases.
  fusion_config.json            Default fusion parameters.
manifests/
  participants.csv              Participant/group mapping template.
  scene_manifest.csv            Scene/order/file mapping template.
scripts/
  build_manifests.py            Scan raw roots and create manifests.
  run_alignment_qc.py           Estimate eye-to-EEG time mapping residuals.
  run_eye_aoi_batch.py          Compute eye AOI metrics from scene CSVs.
  run_end_to_end.py             Orchestrate raw-root to fusion outputs.
  run_fusion.py                 Build aligned scene, time-bin, and sync-QC tables.
src/more_is_not_always_better/
  aoi.py                        AOI loading and polygon metrics.
  discovery.py                  Raw-root scanning and manifest builders.
  eye_batch.py                  Batch eye-tracking AOI runner.
  fusion.py                     EEG + eye fusion and synchronization QC.
matlab/
  README.md                     EEG export contract for MATLAB/EEGLAB runs.
tests/
  fixtures/                     Minimal 1-subject, 2-scene smoke data.
```

## Inputs

`manifests/participants.csv`

```csv
participant_id,eeg_subject_id,eye_subject_id,SportFreq,Experience,Order,exclude
P001,P001,P001,High,Low,1,false
```

`manifests/scene_manifest.csv`

```csv
participant_id,scene_id,block,position,scene_name,eye_csv_path,aoi_json_path,WWR,Cond,Complexity,eye_offset_ms
P001,1,1,1,scene_01,data/raw/eye/P001_scene01.csv,data/raw/aoi/scene_01_aoi.json,0.2,A,1,0
```

EEG summary input:

```text
outputs/eeg/summary/all_subjects_scene_level.csv
```

Eye AOI batch input to fusion:

```text
outputs/eye/batch_aoi_metrics_by_class.csv
```

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## End-To-End Raw Inputs

The repository can now run from the real raw input roots:

```text
Eye-tracking scenes: E:\2.7眼动数据\映射
EEG EEGLAB files:   E:\eeg原始文件
Questionnaire:       E:\VR+EEG实验问卷-文本版-2026-02-17.xlsx
```

EEG raw input is `.set/.fdt` pairs. The MATLAB step reads those files with EEGLAB and exports `outputs/eeg/summary/all_subjects_scene_level.csv`:

```bash
matlab -batch "addpath('matlab'); run_eeg_bandpower_from_set('E:/eeg原始文件', 'outputs/eeg'); exit"
```

Inspect the real roots without running the full dataset:

```bash
python scripts/build_manifests.py --dry_run
python scripts/run_end_to_end.py --dry_run
```

Generate manifests when ready:

```bash
python scripts/build_manifests.py
```

`build_manifests.py` reads `Q1.8_场景顺序编号` from the questionnaire to populate `participants.csv:Order`, so counterbalanced scene order is not guessed. The eye-recording timestamps provide an independent QC check for this order.

If an eye export uses the wrong subject label, provide a manual alias table with `--eye_alias_csv`. Automatic record-id based aliasing is also available for generic labels such as `User1`, but no dataset-specific hard-coded alias is required.

Compute eye-tracking AOI metrics:

```bash
python scripts/run_eye_aoi_batch.py \
  --participants manifests/participants.csv \
  --scene_manifest manifests/scene_manifest.csv \
  --outdir outputs/eye \
  --dwell_mode fixation
```

Build fusion outputs:

```bash
python scripts/run_fusion.py \
  --participants manifests/participants.csv \
  --scene_manifest manifests/scene_manifest.csv \
  --eeg_scene_csv outputs/eeg/summary/all_subjects_scene_level.csv \
  --eye_aoi_class_csv outputs/eye/batch_aoi_metrics_by_class.csv \
  --outdir outputs/fusion
```

Generated outputs:

- `outputs/fusion/aligned_scene_table.csv`
- `outputs/fusion/aligned_timebin_table.csv`
- `outputs/fusion/sync_qc.csv`

## Alignment Rule

For each row in `scene_manifest.csv`:

```text
eye_aligned_ms = eye_timestamp_ms - min(eye_timestamp_ms) + eye_offset_ms
```

The aligned `0 ms` point is interpreted as the EEG scene-viewing start, i.e. the EEG marker transition `7 -> 8`.

Default time bins are non-overlapping `2000 ms` bins, matching the EEG pipeline's 2-second Welch window convention. The current time-bin output computes eye AOI metrics per bin and attaches the corresponding scene-level EEG columns to every bin. If a future EEG export provides true per-bin bandpower, that file can be joined on the same `participant_id + scene_id + bin_start_ms`.

If `aoi_json_path` is blank or missing, the eye pipeline still emits a `whole_scene` class row with dwell, fixation count, TTFF, sample count, and per-bin whole-scene metrics. When AOI JSON files are added later, the same manifest column enables AOI class metrics automatically.

## Precise Alignment QC

Eye CSV files preserve the continuous eye-recorder `Recording Time Stamp[ms]`, not just per-scene relative time. The precise-alignment QC fits one affine map per participant:

```text
eeg_time_ms = time_sync_slope * eye_time_ms + time_sync_offset_ms
```

The landmarks are each scene's eye start/end timestamp and the EEG `view_start_s/view_end_s` from marker `7 -> next 8`. Run after EEG scene export:

```bash
python scripts/run_alignment_qc.py \
  --participants manifests/generated/participants.csv \
  --scene_manifest manifests/generated/scene_manifest.csv \
  --eeg_scene_csv outputs/eeg/summary/all_subjects_scene_level.csv \
  --outdir outputs/fusion
```

Generated QC outputs:

- `outputs/fusion/time_sync_map.csv`
- `outputs/fusion/alignment_landmarks.csv`
- `outputs/fusion/alignment_scene_qc.csv`

## Synchronization QC

`sync_qc.csv` reports:

- EEG view duration, inferred from `view_dur_s`, `dur_s`, `duration_s`, or start/end columns.
- Eye CSV duration from canonical eye timestamp columns.
- Duration delta and `duration_mismatch`, using a default tolerance of `2 s`.
- Eye sample count, time-bin count, missing EEG/eye flags, and scene-count checks.

## Tests

```bash
python -m pytest
```

The smoke test uses one subject and two scenes to validate manifest loading, eye AOI batch output, scene fusion, sync QC, and time-bin output.
