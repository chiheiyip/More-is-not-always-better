# More is not always better: Paper-Level Multimodal Analysis

This repository is the root-level reconstruction of `More-is-not-always-better` into a paper-aligned multimodal analysis repository. It consolidates questionnaire, eye-tracking, EEG, EEG-eye fusion, robustness diagnostics, Nature-style data availability, figure source-data contracts, and reviewer-response evidence into one reproducible pipeline.

The repository is organized around explicit grains. Questionnaire, eye-tracking, and EEG use their own metric-eligible samples; `participant_id + scene_id` is the scene-trial key, AOI models add `class_name`, and the trimodal intersection is reserved for synchronized fusion rather than imposed on unimodal inference.

The authoritative order1/order2/neworder2 sequences, simple eye-folder semantics, `C0/C1` complexity coding, and shared cross-modal questionnaire exclusion policy are documented in [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

## Authoritative Teacher Workflow

The default formal workflow is the staged teacher specification. Python owns
input scanning, QC, fingerprints, approval gates, data contracts, and output
packaging; locked R code owns LMM/GLMM/ordered-beta mixed models, CR2,
`emmeans`, and participant-cluster bootstrap inference.

```powershell
Copy-Item configs/teacher_analysis.example.json configs/teacher_analysis.local.json
python scripts/run_teacher_analysis.py eye-stage1 --config configs/teacher_analysis.local.json --run-id review01
```

Then use the same `--run-id` for `eye-stage2`, `eye-stage3-plan`,
`eye-stage3-run`, `eeg-order`, and `eeg-primary`. Every command also supports
`--outputs-root` and `--dry-run`. Stage 2 requires a current
`AOI_masks_approved.txt`; Stage 3 requires a current
`stage3_plan_approved.txt`; EEG primary requires the matching successful order
analysis marker. Approvals are JSON stored in `.txt` files and are bound to the
stage input fingerprint.

R is mandatory for formal inference. Install R 4.4.x and run
`renv::restore(lockfile = "analysis/r/renv.lock")` before Stage 2 or either EEG
stage. Missing R or a failed mixed model stops or records a diagnostic; no OLS
fallback is permitted.

## Compatibility Workflow

```bash
python scripts/run_all.py --config configs/paths.example.json
```

This older one-command pipeline remains available for synchronization,
time-bin, questionnaire, fusion, figure, and historical-result compatibility.
Its GEE and earlier model outputs are supplementary and are not the
teacher-specified primary inference.

## Output Map

- `outputs/01_sample_qc/`: participant flow, group balance, scene/design balance.
- `outputs/02_questionnaire/`: S1-S5, B1-B3, IPQ long tables, extended descriptives, reliability diagnostics, C1-only B-item QC, subject-level IPQ summaries, item-level LMM diagnostics, and WWR trend contrasts.
- `outputs/03_eye_tracking/`: fixation sequences, two-scope AOI transitions and matrices, dynamic scanpath/saccade/pupil/blink metrics, AOI metrics, structural validation, and 50%–80% QC sensitivity flow.
- `outputs/04_eeg/`: EEG trial-level table and frequency-band QC.
- `outputs/05_multimodal_fusion/`: canonical analysis master table, original-style EEG-eye aligned scene table, time-bin table, sync QC, precise alignment QC, and multimodal claim support.
- `outputs/06_models/`: the two co-primary participant-clustered GEE layers (scene level and clock-synchronized 2-second time bins), fit diagnostics, temporal scene summaries, multiscale claim support, modality sample flow, QC sensitivity models, metric availability, and Monte Carlo MDE. Superseded models are written only with `--legacy-models`, under `outputs/06_models/legacy/`.
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
- `aligned_timebin_table.csv`: legacy compatibility table whose scene-level EEG values are repeated across eye bins; it is marked `legacy_not_for_temporal_inference`.
- `aligned_synchronized_timebin_table.csv`: co-primary 2-second windows computed from the same absolute eye/EEG clock interval, with window-specific EEG power and eye metrics.
- `aligned_pointwise/<participant>/scene_XX_eye_eeg_aligned.csv`: nearest-clock EEG sample matched to each eye row at a maximum absolute difference of 2 ms.

EEG clock time is cached once from the acquisition archive and then reused:

```powershell
python scripts/build_eeg_clock_cache.py --acquisition-root "D:\AAA所有应用\暂存\1.31" --eeg-root "E:\26\补\脑电数据" --cache-root "E:\26\补\脑电数据\eeg_clock_cache"
```

Routine analysis reads the preprocessed `.set/.fdt` waveforms and this E-drive
clock cache only. The `.easy` archive is not copied and is not silently reopened
when the cache is missing. Per-sample columns are explicitly named
`preproc_<channel>_uV`.
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
