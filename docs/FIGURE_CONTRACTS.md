# Figure Contracts

Figures are treated as claim-evidence objects, not only visual styling tasks. The machine-readable source is `configs/figure_contracts.json`; `scripts/run_all.py` exports `outputs/07_paper_tables/figure_contracts_index.csv`, `outputs/07_paper_tables/source_data_index.csv`, and complete figure artifacts under `outputs/10_figures/`.

| Figure | Core conclusion | Primary source data | Review risk |
|---|---|---|---|
| Fig1 design and sample | Sample and scene design are traceable by group, batch, WWR, complexity, and order. | `participant_flow.csv`, `group_balance_before_after.csv`, `scene_design_balance.csv` | Do not hide low-experience supplementation or imbalance. |
| Fig2 questionnaire effects | S1-S5 are evaluated as item-level outcomes. | `s_items_descriptives.csv`, `model_results.csv`, `emmeans_contrasts.csv` | Do not claim a definitive WWR optimum from three WWR levels. |
| Fig3 eye AOI validity | AOI gaze metrics require AOI area, visit rate, and coverage evidence. | `aoi_validation_summary.csv`, `eye_qc.csv`, `eye_aoi_trial_long.csv` | Establish AOI validity before interpreting gaze allocation. |
| Fig4 EEG-eye fusion | EEG interpretation is bounded by synchronization QC and multimodal convergence. | `aligned_scene_table.csv`, `aligned_timebin_table.csv`, `sync_qc.csv`, `time_sync_map.csv`, `claim_support_matrix.csv` | Avoid EEG-only cognitive-load overclaims. |
| Fig5 robustness and claims | Formal order/fatigue-proxy estimates and sensitivity analyses determine final claim strength. | `order_fatigue_effects.csv`, `order_condition_stability.csv`, `carryover_sensitivity.csv`, `nonlinear_wwr_sensitivity.csv`, `claim_strength_table.csv` | Panel A uses modality-faceted participant-clustered GEE estimates with 95% CIs and actual subject/trial counts; do not label proxies as direct fatigue. |
| Fig6 eye fixation density | Six WWR × Complexity fixation distributions are compared after 60% tracking QC and Block 2-to-Block 1 registration. | `Figure6_fixation_source_data.csv`, `Figure6_condition_counts.csv`, `Figure6_registration_qc.csv`, `Figure6_registration_transforms.json` | Do not pool raw Block coordinates; require all six registration and post-warp retention checks to pass. Use one shared blue-green-yellow-red scale with red reserved for the density peak. |

## Export Contract

`scripts/08_build_figures.py` and `scripts/build_eye_scene_figures.py` use the Python/matplotlib backend only. They export editable `svg`, vector `pdf`, high-resolution `tiff`, QA `png`, and the exact source data used by every figure. The eye-scene workflow additionally records source-image hashes, registration matrices, image-mask policy, and registration QA. AOIs use fixed semantic colors, thick outlines with a white contrast halo, and a subtle 10% tint that preserves background-scene detail.

## QA Outputs

- `outputs/10_figures/figure_manifest.csv`: one row per figure with export paths and source CSV.
- `outputs/10_figures/figure_qa.csv`: backend, export, source-data, `n`, error-bar, and image-integrity checks.
- `outputs/10_figures/figure_legends.md`: figure conclusions, panel roles, statistics notes, source data, and review risks.
- `outputs/10_figures/source_data/*_source.csv`: panel-level source data with `n` and `error_bar_definition`.
