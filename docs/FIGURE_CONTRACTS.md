# Figure Contracts

Figures are treated as claim-evidence objects, not only visual styling tasks. The machine-readable source is `configs/figure_contracts.json`; `scripts/run_all.py` exports `outputs/07_paper_tables/figure_contracts_index.csv` and `outputs/07_paper_tables/source_data_index.csv`.

| Figure | Core conclusion | Primary source data | Review risk |
|---|---|---|---|
| Fig1 design and sample | Sample and scene design are traceable by group, batch, WWR, complexity, and order. | `participant_flow.csv`, `group_balance_before_after.csv`, `scene_design_balance.csv` | Do not hide low-experience supplementation or imbalance. |
| Fig2 questionnaire effects | S1-S5 are evaluated as item-level outcomes. | `s_items_descriptives.csv`, `model_results.csv`, `emmeans_contrasts.csv` | Do not claim a definitive WWR optimum from three WWR levels. |
| Fig3 eye AOI validity | AOI gaze metrics require AOI area, visit rate, and coverage evidence. | `aoi_validation_summary.csv`, `eye_qc.csv`, `eye_aoi_trial_long.csv` | Establish AOI validity before interpreting gaze allocation. |
| Fig4 EEG-eye fusion | EEG interpretation is bounded by synchronization QC and multimodal convergence. | `aligned_scene_table.csv`, `aligned_timebin_table.csv`, `sync_qc.csv`, `time_sync_map.csv`, `claim_support_matrix.csv` | Avoid EEG-only cognitive-load overclaims. |
| Fig5 robustness and claims | Sensitivity analyses determine final claim strength. | `gender_sensitivity.csv`, `order_fatigue_effects.csv`, `nonlinear_wwr_sensitivity.csv`, `claim_strength_table.csv` | Downgrade claims when diagnostics are weak or incomplete. |

## Export Contract

Final figure scripts should export editable `svg`, vector `pdf`, high-resolution `tiff`, and the exact `source_csv` used by every panel.
