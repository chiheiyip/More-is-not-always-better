# More is not always better: Paper-Level Multimodal Analysis

This repository is the root-level reconstruction of `More-is-not-always-better` into a paper-aligned multimodal analysis repository. It consolidates questionnaire, eye-tracking, EEG, EEG-eye fusion, robustness diagnostics, Nature-style data availability, figure source-data contracts, and reviewer-response evidence into one reproducible pipeline.

The repository is organized around explicit grains. Questionnaire, eye-tracking, and EEG use their own metric-eligible samples; `participant_id + scene_id` is the scene-trial key, AOI models add `class_name`, and the trimodal intersection is reserved for synchronized fusion rather than imposed on unimodal inference.

The authoritative order1/order2/neworder2 sequences, simple eye-folder semantics, `C0/C1` complexity coding, and shared cross-modal questionnaire exclusion policy are documented in [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

## Core Workflow

```bash
python scripts/run_all.py --config configs/paths.example.json
```

For real data, copy `configs/paths.example.json` to `configs/paths.local.json` and point it to local questionnaire, eye-tracking, and EEG inputs.

## Output Map

- `outputs/01_sample_qc/`: participant flow, group balance, scene/design balance.
- `outputs/02_questionnaire/`: S1-S5, B1-B3, IPQ long tables, extended descriptives, reliability diagnostics, C1-only B-item QC, subject-level IPQ summaries, item-level LMM diagnostics, and WWR trend contrasts.
- `outputs/03_eye_tracking/`: fixation sequences, two-scope AOI transitions and matrices, dynamic scanpath/saccade/pupil/blink metrics, AOI metrics, structural validation, and 50%–80% QC sensitivity flow.
- `outputs/04_eeg/`: EEG trial-level table and frequency-band QC.
- `outputs/05_multimodal_fusion/`: canonical analysis master table, original-style EEG-eye aligned scene table, time-bin table, sync QC, precise alignment QC, and multimodal claim support.
- `outputs/06_models/`: the canonical participant-clustered GEE result table, fit diagnostics, modality sample flow, QC sensitivity models, metric availability, and Monte Carlo MDE. Superseded models are written only with `--legacy-models`, under `outputs/06_models/legacy/`.
- `outputs/06_robustness/`: formal block/position order estimates, scene-level order descriptives, condition-by-trial-index stability, previous-condition carryover, gender, batch, nonlinear WWR, and power sensitivity.
- `outputs/07_paper_tables/`: paper-facing tables, claim strength table, result summary.
- `outputs/08_reviewer_response/`: reviewer issue to evidence index and reviewer issue matrix.
- `outputs/09_data_package/`: Data Availability draft and dataset/source-data availability index.
- `outputs/10_figures/`: Nature-style SVG/PDF/TIFF/PNG figure exports, panel source CSV files, figure manifest, legends, and QA.

## Questionnaire Method Policy

Questionnaire logic draws on the public `wannaqueen66-create/spss` workflow, but it is not copied wholesale. The adopted pieces are the robust parts: extended descriptive diagnostics, Cronbach alpha as internal-consistency evidence, B-item C1-only handling, subject-level IPQ analysis, item-level mixed models, and WWR linear/quadratic trend contrasts.

The corrections are deliberate: S1-S5 stay as primary item-level outcomes; `Afford4` is only a supplementary S1-S4 candidate construct; `Bmean` is C1-only; `IPQ_mean` is participant-level and is not interpreted as scene-level WWR/Complexity evidence; Shapiro/skew/kurtosis are diagnostics only; three WWR levels support trend language, not a definitive optimum claim.

## Reviewer-Driven Principles

- Supplementary recruitment is encoded explicitly using `RecruitmentBatch` and `SupplementFlag`.
- Gender, age, block, position, and recruitment batch are available as covariates in registered models.
- Three WWR levels are treated as supporting trend or planned-contrast language only; the pipeline does not encode a strong optimality claim.
- EEG interpretations are claim-gated through multimodal convergence with questionnaire and/or eye-tracking evidence.
- AOI validity is documented via AOI area, visited rate, and per-AOI sample coverage.
- Eye-tracking does not inherit EEG QC exclusions. Coordinate-validity thresholds are sensitivity analyses, not primary hard gates.
- Pupil change is referenced to the first two seconds of the already-presented scene, not a pre-stimulus baseline, and is labelled exploratory/luminance-confounded.

## Integrated EEG + Eye Fusion

The fusion layer is not a bolt-on script. It builds a canonical trial index from standardized participants and scene manifests, then emits all downstream views from that shared base:

- `analysis_master_long.csv`: paper-level questionnaire + EEG + eye table for statistical modeling.
- `aligned_scene_table.csv`: scene-level EEG + AOI metrics, compatible with the original fusion concept.
- `aligned_timebin_table.csv`: time-bin eye AOI metrics with scene-level EEG attached.
- `sync_qc.csv`: eye duration, EEG duration, mismatch flags, scene count checks.
- `alignment_scene_qc.csv`, `alignment_landmarks.csv`, `time_sync_map.csv`: precise eye-to-EEG affine alignment diagnostics.

## Historical Logic Sources

- Questionnaire logic source: `https://github.com/wannaqueen66-create/spss`
- EEG + eye-tracking fusion logic source: the original logic in this `More-is-not-always-better` repository, preserved in `src/more_is_not_always_better/` and integrated into the paper-level `src/paper_analysis/` architecture.

These are logic sources, not runtime dependencies.

## Nature-Skills Alignment

- `nature-response`: reviewer concerns are mapped through `configs/reviewer_response_map.json`, `docs/REVIEWER_ISSUE_MATRIX.md`, and `outputs/08_reviewer_response/`.
- `nature-writing`: claim strength is explicitly constrained by `outputs/07_paper_tables/claim_strength_table.csv`.
- `nature-data`: dataset access routes and unresolved repository identifiers are tracked by `configs/data_availability.json` and `outputs/09_data_package/`.
- `nature-figure`: each paper figure has a claim/evidence/source-data/export contract in `configs/figure_contracts.json`, plus Python/matplotlib SVG/PDF/TIFF/PNG exports, panel source CSV files, legends, and QA under `outputs/10_figures/`.
