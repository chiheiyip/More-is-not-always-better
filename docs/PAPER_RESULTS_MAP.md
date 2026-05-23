# Paper Results Map

| Manuscript section | Pipeline step | Output |
|---|---|---|
| 2.2 Participants | `00_build_manifests.py` | `outputs/01_sample_qc/participant_flow.csv`, `group_balance_before_after.csv` |
| 2.3 Stimuli and design | `00_build_manifests.py` | `outputs/01_sample_qc/scene_design_balance.csv` |
| 2.6.1 EEG | `03_eeg_pipeline.py` | `outputs/04_eeg/eeg_trial_long.csv`, `eeg_qc_summary.csv` |
| 2.6.2 Eye tracking | `02_eye_pipeline.py` | `outputs/03_eye_tracking/eye_aoi_trial_long.csv`, `aoi_validation_summary.csv` |
| 2.6.3 Questionnaire | `01_questionnaire_pipeline.py` | `outputs/02_questionnaire/questionnaire_long.csv`, `questionnaire_paper_tables.md` |
| 2.7 Statistical analysis | `05_statistical_models.py` | `outputs/06_models/model_results.csv`, `emmeans_contrasts.csv` |
| EEG-eye synchronization | `04_fusion_pipeline.py` | `outputs/05_multimodal_fusion/sync_qc.csv`, `alignment_scene_qc.csv`, `time_sync_map.csv` |
| Time-bin diagnostics | `04_fusion_pipeline.py` | `outputs/05_multimodal_fusion/aligned_timebin_table.csv` |
| 3 Results | `07_build_paper_outputs.py` | `outputs/07_paper_tables/table_model_results.csv` |
| Discussion and limitations | `06_diagnostics_and_sensitivity.py` | `outputs/06_robustness/*.csv`, `outputs/07_paper_tables/claim_strength_table.csv` |
| Figure source data | `07_build_paper_outputs.py` | `outputs/07_paper_tables/figure_contracts_index.csv`, `outputs/07_paper_tables/source_data_index.csv` |
| Nature-style figures | `08_build_figures.py` | `outputs/10_figures/figures/*.svg`, `outputs/10_figures/figures/*.pdf`, `outputs/10_figures/figures/*.tiff`, `outputs/10_figures/source_data/*_source.csv`, `outputs/10_figures/figure_qa.csv` |
| Response to reviewers | `07_build_paper_outputs.py` | `outputs/08_reviewer_response/response_evidence_index.csv`, `outputs/08_reviewer_response/reviewer_issue_matrix.csv` |
| Data Availability | `07_build_paper_outputs.py` | `outputs/09_data_package/data_availability_index.csv`, `outputs/09_data_package/data_availability_statement.md` |
