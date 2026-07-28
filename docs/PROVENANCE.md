# Provenance

This repository tracks analysis provenance from raw modality inputs to manuscript-ready outputs.

## Teacher-priority run provenance

`scripts/run_teacher_analysis.py` is the authoritative formal entry point.
The complete entry point is `all-results --self-review --reuse-valid --resume
--promote`; stage-specific commands remain available for diagnosis.
Each stage writes to
`<outputs-root>/teacher_runs/<run-id>/<numbered-stage>/` and never overwrites
historical outputs. Every `run_manifest.json` records the stage fingerprint,
resolved config, arguments, Git commit, Python/platform identity, status, and
blocking reasons. Approval files must contain the exact stage name and
fingerprint; changed inputs make old approval unusable.
Every manifest also records a method-contract hash calculated from the teacher
Python orchestration, R model scripts and command-line entry point. Resume and
reuse are invalidated when that code contract changes, even if source data are
unchanged. In self-review mode, skipped manual gates are recorded as explicit
machine review decisions rather than silently omitted.

Formal narrative reports are Markdown. Excel/CSV tables remain the
machine-readable source of numerical claims; figures, logs and code snapshots
remain separate auditable artifacts. DOCX is optional and is not a completion
criterion.

The formal key is `Participant + GlobalTrialOrder`. Modality registries are
unions. Eye, EEG, and questionnaire samples are recorded separately, while
cross-modal work must materialize and report the exact participant–trial
intersection. R dependency versions are locked in `analysis/r/renv.lock`;
formal stages stop when R or its packages are unavailable.

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
| Questionnaire analysis | `outputs/02_questionnaire/questionnaire_long.csv` | Canonical questionnaire-specific source using all available questionnaire scene trials; EEG QC is not applied. |
| Eye tracking | `outputs/03_eye_tracking/eye_aoi_trial_long.csv` | AOI metrics attached to trial index. |
| EEG | `outputs/04_eeg/eeg_trial_long.csv` | EEG scene metrics attached to trial index. |

## Fusion Layer

`scripts/04_fusion_pipeline.py` creates all EEG+eye+questionnaire projections from the same trial index:

- `analysis_master_long.csv` for statistical models.
- `aligned_scene_table.csv` for scene-level EEG+AOI fusion.
- `aligned_timebin_table.csv` for legacy compatibility only; EEG is scene-repeated and the table is not used for temporal inference.
- `aligned_synchronized_timebin_table.csv` for window-specific eye/EEG features from identical absolute-clock intervals.
- `clock_alignment_scene_qc.csv` and `clock_alignment_participant_qc.csv` for clock eligibility.
- `aligned_pointwise/` for the auditable nearest-sample match at ≤2 ms.

Canonical unimodal inference is generated from modality-specific scene tables:
questionnaire uses its complete available table, eye tracking uses
metric-specific availability, and EEG uses the EEG-QC-passed table. The
trimodal synchronized keep set is used only by aligned fusion outputs.

The absolute EEG clock cache is built once from the acquisition archive and
stored with the E-drive EEG data. Routine processing thereafter uses only that
cache and the preprocessed `.set/.fdt` waveform. Source identity follows the
participant name/history mapping; sample count and sample rate verify that
recording sample indices remain compatible.
- `sync_qc.csv`, `alignment_scene_qc.csv`, `alignment_landmarks.csv`, `time_sync_map.csv` for synchronization evidence.
- `modality_convergence_table.csv` and `claim_support_matrix.csv` for bounded EEG/multimodal interpretation.

## Manuscript Layer

| Manuscript need | Output |
|---|---|
| Statistical result tables | `outputs/06_models/model_results.csv`, `outputs/07_paper_tables/table_model_results.csv` |
| Robustness and reviewer concerns | `outputs/06_robustness/*.csv` |
| Bounded discussion claims | `outputs/07_paper_tables/claim_strength_table.csv` |
| Figure source data | `outputs/07_paper_tables/source_data_index.csv` |
| Nature-style figure exports and QA | `outputs/10_figures/figure_manifest.csv`, `outputs/10_figures/figure_qa.csv`, `outputs/10_figures/source_data/*_source.csv` |
| Reviewer response evidence | `outputs/08_reviewer_response/*.csv` |
| Data availability package | `outputs/09_data_package/*.csv`, `outputs/09_data_package/*.md` |
