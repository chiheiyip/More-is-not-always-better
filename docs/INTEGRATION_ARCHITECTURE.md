# Integration Architecture

This repository uses a bottom-layer integration design rather than patching old scripts into a new folder. The root `More-is-not-always-better` repository now contains both the legacy compatibility package and the paper-level analysis package.

## Canonical Trial Index

The canonical unit is:

```text
participant_id + scene_id
```

The trial index is built from standardized participants and scene manifests. It carries:

- Participant fields: `ExperienceGroup`, `Gender`, `Age`, `RecruitmentBatch`, `SupplementFlag`.
- Design fields: `WWR`, `Complexity`, `Cond`, `block`, `position`, `round`, `condition_id`.
- Data paths: `eye_csv_path`, `aoi_json_path`.

All modality tables must attach to this index. This prevents questionnaire, eye-tracking, EEG, time-bin QC, and paper statistics from using incompatible definitions of a trial.

## Fusion Core Outputs

`paper_analysis.fusion.pipeline.run_fusion_pipeline()` is the single fusion core. It emits:

| Output | Purpose |
|---|---|
| `analysis_master_long.csv` | Paper-level table used by statistical models. |
| `aligned_scene_table.csv` | Scene-level EEG + AOI metric table, preserving the original fusion pipeline concept. |
| `aligned_timebin_table.csv` | Time-bin eye AOI table with scene EEG attached for temporal/order diagnostics. |
| `sync_qc.csv` | Duration and scene-count QC for EEG-eye synchronization. |
| `alignment_scene_qc.csv` | Per-scene precise alignment QC. |
| `alignment_landmarks.csv` | Start/end eye and EEG landmarks used for affine time mapping. |
| `time_sync_map.csv` | Participant-level eye-to-EEG time mapping and residual diagnostics. |

## Relationship To Original Repository

The original `More-is-not-always-better` repository provided EEG + eye-tracking fusion logic. That logic is still present for compatibility under `src/more_is_not_always_better/`, while the paper-facing architecture is integrated at the data-model layer under `src/paper_analysis/`:

- Original scene-level fusion maps to `aligned_scene_table.csv`.
- Original time-bin eye/EEG table maps to `aligned_timebin_table.csv`.
- Original synchronization QC maps to `sync_qc.csv`.
- Original precise alignment QC maps to `alignment_scene_qc.csv`, `alignment_landmarks.csv`, and `time_sync_map.csv`.
- Paper-specific statistics use `analysis_master_long.csv`, which is generated from the same trial index.

This means the paper statistics and the EEG-eye QC are no longer separate products; they are different projections of the same canonical multimodal dataset.

## Nature-Level Evidence Layers

The reporting layer adds three manuscript-facing projections:

- Reviewer response evidence: `response_evidence_index.csv` and `reviewer_issue_matrix.csv`.
- Figure/source-data contracts: `figure_contracts_index.csv` and `source_data_index.csv`.
- Data availability package: `data_availability_index.csv` and `data_availability_statement.md`.

These outputs make the repository auditable from reviewer concern to analysis action, from figure claim to source data, and from manuscript claim to data availability route.
