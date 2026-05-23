# Provenance

This repository tracks analysis provenance from raw modality inputs to manuscript-ready outputs.

## Input Layer

| Input | Config key | Main use |
|---|---|---|
| Participant table | `participants` | Demographics, exclusion, experience group, recruitment batch. |
| Scene manifest | `scene_manifest` | `participant_id + scene_id` trial index, WWR, complexity, order, eye/AOI paths. |
| Questionnaire export | `questionnaire_wide` or `questionnaire_long` | S1-S5, B1-B3, IPQ, subjective outcomes. |
| Eye-tracking CSV and AOI JSON | `scene_manifest.eye_csv_path`, `scene_manifest.aoi_json_path` | AOI metrics, time-bin metrics, AOI validity. |
| EEG scene export | `eeg_scene_csv` | EEG theta/alpha metrics and viewing-duration landmarks. |

## Standardization Layer

`scripts/00_build_manifests.py` standardizes participants and scenes. The canonical unit is:

```text
participant_id + scene_id
```

All later tables must preserve these keys.

## Modality Layer

| Step | Output | Provenance role |
|---|---|---|
| Questionnaire | `outputs/02_questionnaire/questionnaire_long.csv` | Subjective outcomes attached to trial index. |
| Eye tracking | `outputs/03_eye_tracking/eye_aoi_trial_long.csv` | AOI metrics attached to trial index. |
| EEG | `outputs/04_eeg/eeg_trial_long.csv` | EEG scene metrics attached to trial index. |

## Fusion Layer

`scripts/04_fusion_pipeline.py` creates all EEG+eye+questionnaire projections from the same trial index:

- `analysis_master_long.csv` for statistical models.
- `aligned_scene_table.csv` for scene-level EEG+AOI fusion.
- `aligned_timebin_table.csv` for time-bin eye metrics with EEG attached.
- `sync_qc.csv`, `alignment_scene_qc.csv`, `alignment_landmarks.csv`, `time_sync_map.csv` for synchronization evidence.
- `modality_convergence_table.csv` and `claim_support_matrix.csv` for bounded EEG/multimodal interpretation.

## Manuscript Layer

| Manuscript need | Output |
|---|---|
| Statistical result tables | `outputs/06_models/model_results.csv`, `outputs/07_paper_tables/table_model_results.csv` |
| Robustness and reviewer concerns | `outputs/06_robustness/*.csv` |
| Bounded discussion claims | `outputs/07_paper_tables/claim_strength_table.csv` |
| Figure source data | `outputs/07_paper_tables/source_data_index.csv` |
| Reviewer response evidence | `outputs/08_reviewer_response/*.csv` |
| Data availability package | `outputs/09_data_package/*.csv`, `outputs/09_data_package/*.md` |
