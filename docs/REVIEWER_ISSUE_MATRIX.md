# Reviewer Issue Matrix

This follows the `nature-response` rule that every reviewer concern must map to a concrete action, evidence output, or explicit author-input placeholder.

| Issue | Repository response | Evidence output | Response posture |
|---|---|---|---|
| Sample size and low-experience imbalance | Supplement low-experience participants and report balance | `outputs/01_sample_qc/group_balance_before_after.csv` | Answer with before/after counts and bounded moderation language. |
| Gender recorded but omitted | Include Gender in models and sensitivity summaries | `outputs/06_robustness/gender_sensitivity.csv` | Treat Gender as covariate/sensitivity factor, not as a new primary claim. |
| Fatigue/order/carryover | Model and diagnose block/position/round trends | `outputs/06_robustness/order_fatigue_effects.csv` | Report whether order variables materially alter conclusions. |
| WWR nonlinear overclaim | Use planned contrasts and soften optimality language | `outputs/06_robustness/nonlinear_wwr_sensitivity.csv` | State exploratory trend only; do not claim definitive optimum. |
| AOI validity | Report AOI area, visited rate, coverage | `outputs/03_eye_tracking/aoi_validation_summary.csv` | Establish measurement validity before interpreting AOI effects. |
| EEG interpretation overclaim | Gate claims through multimodal convergence and claim strength | `outputs/05_multimodal_fusion/claim_support_matrix.csv`, `outputs/07_paper_tables/claim_strength_table.csv` | Require EEG-eye-questionnaire convergence before cognitive-load interpretation. |
| Data/source-data availability | Provide raw/processed/source-data inventory and repository placeholders | `outputs/09_data_package/data_availability_index.csv` | Keep repository DOI/access restrictions as `AUTHOR_INPUT_NEEDED` until confirmed. |
