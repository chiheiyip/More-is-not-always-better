# Analysis Strengthening Report

本报告记录本轮补强分析是否覆盖前次审查指出的缺口。

## 新增补强输出

- 06_robustness/datebatch_adjusted_core_models.csv: present
- 06_robustness/experience_split_questionnaire.csv: present
- 06_robustness/effect_size_summary.csv: present
- 07_paper_tables/main_claim_evidence_chain.csv: present
- 07_paper_tables/analysis_strengthening_report.md: present

## 关键读数

- DateBatch 控制模型：fit rows=70, WWR p<0.05 rows=12, DateBatch-related p<0.05 rows=2.
- 经验组拆分：Low leaders q_S1:WWR15.0, q_S2:WWR15.0, q_S3:WWR15.0, q_S4:WWR15.0, q_S5:WWR15.0; High leaders q_S1:WWR15.0, q_S2:WWR15.0, q_S3:WWR45.0, q_S4:WWR15.0, q_S5:WWR15.0
- FDR 修正：raw p<0.05=99; BH-FDR all<0.05=88; BH-FDR family<0.05=88.
- 模型警告：fallback rows=81; warning rows=151; read p-values with these flags.

## 研究问题证据链

| research_question | key_data_result | model_or_sensitivity_result | claim_strength | caveat | recommended_use |
| --- | --- | --- | --- | --- | --- |
| RQ1_WWR_subjective | WWR15 is highest for 5/5 S items; leaders: S1:15=5.54; S2:15=5.78; S3:15=5.4; S4:15=5.49; S5:15=6.78 | rows=60, p<0.05=19, FDR-family<0.05=18, warning=10. | moderate_for_three_tested_levels | WWR has only 15/45/75 levels; do not claim a continuous optimum. | Primary result, with trend/three-level wording. |
| RQ2_supplement_batch | First_before_2026-05-01: S1:WWR15, S2:WWR15, S3:WWR15, S4:WWR15, S5:WWR15; Second_2026-05-01_or_later: S1:WWR75, S2:WWR15, S3:WWR45, S4:WWR75, S5:WWR15 | fit rows=70, WWR p<0.05 rows=12, DateBatch-related p<0.05 rows=2. | sensitivity_evidence | DateBatch is confounded with ExperienceGroup balance; interpret as sensitivity, not causal batch effect. | Use to state that supplementation attenuated but did not overturn the overall WWR pattern. |
| RQ3_experience_group | Low leaders q_S1:WWR15.0, q_S2:WWR15.0, q_S3:WWR15.0, q_S4:WWR15.0, q_S5:WWR15.0; High leaders q_S1:WWR15.0, q_S2:WWR15.0, q_S3:WWR45.0, q_S4:WWR15.0, q_S5:WWR15.0 | rows=20, p<0.05=4, FDR-family<0.05=2, warning=4. | exploratory_moderation | ExperienceGroup and DateBatch are imbalanced; interaction language requires caution. | Use as subgroup exploration, not as a settled mechanism. |
| RQ4_complexity | S1:C1-C0=0.22; S2:C1-C0=0.0685; S3:C1-C0=0.0476; S4:C1-C0=0.101; S5:C1-C0=0.0357 | rows=10, p<0.05=9, FDR-family<0.05=9, warning=5. | cautious_moderate | Raw questionnaire differences are small and batch-dependent; mechanisms need multimodal convergence. | Secondary result with bounded wording. |
| RQ5_eye_tracking | visited:WWR75; FCR:WWR75; TFD_ms:WWR75; TTFF_ms:WWR15; attention_share:WWR75 | rows=35, p<0.05=0, FDR-family<0.05=0, warning=9. | auxiliary_partial | AOI-expanded rows are not scene trials; several continuous eye models need stability checks. | Auxiliary evidence for attention allocation. |
| RQ6_EEG | F_theta:WWR75; O_theta:WWR75; O_alpha:WWR45 | rows=256, p<0.05=42, FDR-family<0.05=37, warning=75. | bounded_auxiliary | EEG exclusions and model warnings prevent standalone cognitive-mechanism claims. | Use only after QC and convergence caveats. |
| RQ7_reporting_integrity | raw p<0.05=99; BH-FDR all<0.05=88; BH-FDR family<0.05=88. | fallback rows=81; warning rows=151; read p-values with these flags. | audit_ready_with_warnings | Data Availability placeholders remain author input; FDR should be read with model-warning flags. | Use for teacher/reviewer-facing transparency. |

## 仍需谨慎

- DateBatch 与 ExperienceGroup 结构不均衡，批次差异不能直接写成补样因果效应。
- WWR 只有 15/45/75 三档，不能写连续最优点。
- EEG 和部分眼动模型仍需与 QC、fallback/warning 标记一起解读。
- FDR 后未显著的结果只能作为方向性或探索性材料。
