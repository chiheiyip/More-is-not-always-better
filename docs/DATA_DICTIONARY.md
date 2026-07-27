# Data Dictionary

## Core Keys

| Field | Meaning |
|---|---|
| `participant_id` | Canonical participant identifier across questionnaire, eye, and EEG data. |
| `scene_id` | Presentation slot 1–12 within participant, resolved from questionnaire Order, collection batch, block, and condition; see `EXPERIMENT_DESIGN.md`. |
| `condition_id` | Combined condition label, usually derived from WWR and Complexity. |
| `WWR` | Window-to-wall ratio condition. |
| `Complexity` | Visual complexity parsed only from the condition label: `C0=0` (low), `C1=1` (high). |
| `block` | Experimental block (1 or 2); the numeric prefix in a simple eye folder is block, not Order. |
| `position` | Within-block presentation position; used for order/fatigue diagnostics. |
| `round` | Viewing round when available. |
| `trial_index` | Canonical experiment progression index; equals `scene_id = (block - 1) * 6 + position`. |
| `previous_WWR`, `previous_Complexity` | Previous scene condition within `participant_id + block`; missing for position 1 of both blocks. |
| `order_scheme` | Counterbalancing/order sequence identifier attached from the scene manifest. |
| `break_before_trial` | Descriptive indicator for block 2, position 1, which follows the registered 120-second break; it is not used as a lagged-condition model term because both block starts have undefined within-block lags. |

## Teacher workflow canonical fields

| Field | Meaning |
|---|---|
| `Participant` | Modality-union participant identifier; absence from one modality does not remove another. |
| `ExperienceRaw` | Unmodified Q1.4 experience response used by the repository-wide grouping rule. |
| `ExperienceGroup` | Registered `High`/`Low` experience classification. Teacher-priority eye and EEG models use this field; Q1.5 exercise frequency is not a substitute. |
| `OrderGroup` | One of `order1`, `order2`, or canonical `new order2`; `neworder2` is compatibility input only. |
| `IncludeEyeCandidate` | Candidate for eye QC, independent of EEG and questionnaire status. |
| `IncludeEEGValid` | EEG-valid flag; used for EEG and explicitly named common-sample sensitivities only. |
| `IncludeQuestionnaireValid` | Questionnaire-valid flag. |
| `GlobalTrialOrder` | Formal trial key 1–12 within participant. |
| `PositionWithinBlockCentered` | Within-block position centered at 3.5. |
| `ValidTrackingRatio` | Recomputed ratio using configured binocular validity and finite coordinates. |
| `SoftwareTrackingRatio` | Vendor-reported ratio retained for audit. |
| `TrackingRatioDifference` | Recomputed minus software tracking ratio. |
| `ValidScene` | Confirmed stimulus region used in the attention-share denominator. |
| `OffStimulus` | Fixation outside ValidScene; excluded from ValidScene TFD. |
| `StructuralNA` | True for Equipment in C0; not a measured zero. |
| `stage_fingerprint` | SHA-256 binding stage config and input identities to manifests and approvals. |

## Participant Fields

| Field | Meaning |
|---|---|
| `ExperienceRaw` | Original Q1.4 table-tennis-experience response used for experience grouping. |
| `ExperienceGroup` | Latest two-by-two Q1.4 experience grouping: `Low` = never/rarely or occasional (monthly <1 or 1-2 times); `High` = sometimes/often (monthly 3-4 or >=5 times). `SportFreq`/Q1.5 is never substituted for this field. |
| `ExperienceGroupInput` | Optional audit copy of any pre-existing `ExperienceGroup` column before recomputing the latest two-by-two grouping. |
| `ExperienceGroupSource` | Source column used to recompute `ExperienceGroup`, normally `Experience` from Q1.4. |
| `ExperienceGroupRule` | Experience grouping rule identifier, normally `q1_4_table_tennis_experience_2_by_2`. |
| `DateBatch` | Collection-date batch inferred from `eye_record_id`: before 2026-05-01 = first batch; 2026-05-01 or later = second/supplement batch. |
| `collection_date` | Parsed collection date from the leading YYMMDD portion of `eye_record_id`. |
| `Gender` | Participant gender, included as a model covariate. |
| `Age` | Participant age, included as a model covariate when available. |
| `RecruitmentBatch` | Original or supplementary recruitment batch. |
| `SupplementFlag` | Whether the participant belongs to the supplementary recruitment batch. |
| `exclude` | Participant-level exclusion flag. |
| `ExcludeReason` | Participant-level exclusion reason. |

## Questionnaire Fields

`S1`-`S5` are primary scene-level subjective items and are analyzed separately by default; the pipeline does not force them into a total score. `Afford4` is a supplementary composite from `S1`-`S4` when enough valid items are present and reliability diagnostics support that interpretation. `S5_7` rescales a detected 1-9 `S5` item to 1-7 for comparability, but `S5` is not merged into `Afford4`. `B1`-`B3` and `Bmean` are supplementary C1-only outcomes. `IPQ1`-`IPQ6` and `IPQ_mean` are subject-level scale fields; IPQ outputs should not be interpreted as repeated scene-level trial effects.

| Field or file | Meaning |
|---|---|
| `questionnaire_long.csv` | Complete prepared questionnaire trial table and canonical questionnaire-specific analysis source. EEG QC is not applied to it. |
| `questionnaire_analysis_long.csv` | Legacy/shared-intersection compatibility view. It is not the canonical questionnaire model source. |
| `questionnaire_analysis_sample.csv` | Audit row recording the raw, retained, and excluded questionnaire trial/participant counts and the applied QC policy. |
| `questionnaire_descriptives.csv` | Item/composite descriptives with observation count, subject count, mean, SD, median, range, 95% CI, skewness, kurtosis, and Shapiro diagnostic p value when estimable. |
| `questionnaire_reliability.csv` | Cronbach alpha diagnostics for configured scales such as `S1`-`S4`, `S1`-`S5`, `B1`-`B3`, and `IPQ1`-`IPQ6`; statuses distinguish acceptable, check, insufficient rows, and missing items. |
| `questionnaire_scale_qc.csv` | Scale and composite warnings, including `S5` scale conversion notes and composite interpretation limits. |
| `questionnaire_b_item_qc.csv` | C1-only quality-control table for `B1`-`B3`, including warnings if C0 rows contain B-item values. |
| `questionnaire_item_model_results.csv` | Reproducible item-level mixed-model or fallback regression coefficients, confidence intervals, p values, and model metadata; this is not labelled as SPSS Type III output. |
| `questionnaire_wwr_polynomial_contrasts.csv` | Participant-aggregated three-level WWR linear and quadratic planned contrasts with `claim_strength=trend_only`. |
| `ipq_subject_level.csv` | One row per participant for IPQ subject-level analysis. |
| `ipq_descriptives.csv`, `ipq_reliability.csv`, `ipq_group_comparisons.csv` | Subject-level IPQ descriptives, reliability diagnostics, and group comparisons when IPQ item data are available. |

## Eye-Tracking Fields

| Field | Meaning |
|---|---|
| `visited` | Whether the AOI was visited. |
| `FC` | Number of unique fixations in the AOI. |
| `FCR` / `FC_rate` | Fixation count per valid trial second. |
| `FC_share` / `FC_prop` | AOI fixation count divided by total valid trial fixation count. |
| `TFD_ms` | Total fixation duration in milliseconds. |
| `TTFF_ms` | Time to first fixation in milliseconds. |
| `FFD_ms` | First fixation duration in the AOI. Exploratory by default. |
| `MFD_ms` | Mean fixation duration in the AOI. Exploratory by default. |
| `RFF` | Re-fixation frequency, counted as returns to an AOI after leaving it. Exploratory by default. |
| `MPD` | Mean pupil diameter over AOI samples when exported by the eye tracker. Exploratory by default. |
| `attention_share` / `share` / `share_pct` | AOI TFD divided by total valid trial TFD, with `share_pct` expressed as percent. |
| `point_source_used` | Coordinate source used for AOI hit testing: `fixation` or `gaze`. Formal analyses should use fixation points when exported; `auto` falls back to gaze and records the fallback. |
| `analysis_valid_ratio` | Fraction of rows retained after finite-coordinate, optional screen-bounds, and optional validity-code checks. |
| `screen_valid_ratio` | Fraction of rows with coordinates inside configured screen bounds; audit-only unless screen dimensions are supplied. |
| `validity_valid_ratio` | Fraction of rows whose vendor validity codes match the configured accepted values; audit-only unless accepted values are supplied. |
| `time_segment_count` | Number of timestamp segments detected after resets or large gaps. |
| `timestamp_gap_count` | Number of timestamp gaps above the configured threshold. |
| `aoi_overlap_summary.csv` | Per-trial overlap of AOI class hit masks. Non-zero overlap means AOI shares can legitimately sum above 1 and should be interpreted with care. |
| `eye_fixation_sequence_long.csv` | One row per unique fixation, including time segment, centroid, binocular mean gaze direction, assigned AOI, all candidate AOIs, ambiguity, and canvas status. |
| `eye_transition_long.csv` | One row per compressed transition, with `state` scope including `outside` and `named_aoi` scope omitting outside while recording `passed_outside`. |
| `eye_transition_matrix.csv` | Directed transition counts and conditional probabilities by from-AOI, to-AOI, and transition scope. |
| `eye_trial_dynamic_metrics.csv` | One row per participant-scene containing transitions, entropy, 2D/angular scanpath, vendor-event saccades/blinks, and scene-early pupil change. |
| `transition_entropy_normalized` | Transition entropy divided by the maximum implied by the number of AOIs available in that scene. |
| `angular_scanpath_deg_per_s` | Sum of angles between adjacent binocular mean unit gaze vectors, divided by valid scene duration; timestamp gaps >5 s are not bridged. |
| `median_revisit_latency_ms` | Median time between leaving and subsequently revisiting an AOI, calculated within timestamp segments. |
| `pupil_post_early_delta_mm` | Median post-2-second pupil diameter minus the first-2-second scene-early reference. Not a pre-stimulus baseline; exploratory and luminance-confounded. |
| `coordinate_contract_status` | Whether source/target canvases were explicitly verified, explicitly scaled, incomplete, or unverified. |
| `eye_qc_sensitivity.csv` | Retained trial/subject counts and condition balance at 50%, 60%, 70%, and 80% valid-coordinate thresholds. |

## Order, Fatigue-Proxy, and Carryover Outputs

| File | Meaning |
|---|---|
| `order_fatigue_effects.csv` | Formal participant-clustered GEE block/position estimates with SE, 95% CI, raw p, BH-FDR q, actual subjects/trials, effect scale, and fit status. |
| `order_fatigue_descriptives.csv` | Scene-grain questionnaire, eye and EEG summaries by block and position; never AOI-expanded for scene outcomes. |
| `order_condition_stability.csv` | Controlled-versus-unadjusted WWR/complexity coefficients plus WWR × trial-index and Complexity × trial-index sensitivity terms. |
| `carryover_sensitivity.csv` | Previous-WWR, previous-complexity, order-scheme and break-marker sensitivity results, including explicit failed/unstable status rows. |
| `reviewer_order_fatigue_evidence.md` | Reviewer-facing R1.11/R2.5 evidence package with methods, numeric results, limitations and readiness gate. |

## Fusion And Synchronization Fields

| Field or file | Meaning |
|---|---|
| `analysis_master_long_pre_qc.csv` | AOI-expanded multimodal table before analysis-level QC exclusions. |
| `analysis_master_long.csv` | Main AOI-expanded analysis table after analysis-level QC. Do not treat its row count as the number of participant-scene trials. |
| `analysis_qc_exclusions.csv` | Participant-scene QC table listing retained/excluded trials and exclusion reasons. |
| `aligned_scene_table.csv` | Scene-level EEG + eye AOI projection from the canonical trial index. |
| `aligned_timebin_table.csv` | Legacy compatibility projection with scene-level EEG repeated across eye bins; marked `eeg_temporal_resolution=scene_repeated` and `analysis_status=legacy_not_for_temporal_inference`. |
| `eeg_recording_clock.csv` | One-time recording-level EEG epoch cache, source checksum, sampling structure, timezone, and cache status. |
| `eeg_trigger_events.csv` | Validated EEG trigger code, recording sample index, Unix epoch ms, and local datetime. |
| `eeg_source_resolution.csv` | Candidate EASY source audit and deterministic selection reason. |
| `eeg_sample_file_manifest.csv` | One row per participant-scene preprocessed EEG sample CSV, with `[7,8)` latencies, timing bounds, rate/cache checks, and export status. |
| `clock_alignment_scene_qc.csv` | Scene-level absolute-clock coverage, nearest-match rate/delta, monotonicity, and source/cache QC. |
| `clock_alignment_participant_qc.csv` | Participant-level 12-scene clock-alignment eligibility. |
| `aligned_synchronized_timebin_table.csv` | Co-primary non-overlapping 2-second windows with window-specific F/P/O theta/alpha/beta and eye AOI metrics from the identical epoch interval. |
| `timebin_model_results.csv` | Co-primary synchronized-time GEE estimates and separate BH-FDR families. |
| `temporal_scene_summaries.csv` | Early/middle/late means, late-minus-early difference, and per-scene slope linking the two primary resolutions. |
| `multiscale_claim_support.csv` | Separate overall-level and time-dynamic support with a scale-dependence interpretation rule. |
| `sync_qc.csv` | Per-trial duration and scene-count synchronization QC. |
| `duration_delta_s` | Eye duration minus EEG viewing duration. |
| `duration_mismatch` | Whether duration delta exceeds the configured tolerance. |
| `time_sync_slope` | Participant-level affine eye-to-EEG time mapping slope. |
| `time_sync_offset_ms` | Participant-level affine eye-to-EEG time mapping offset. |
| `median_abs_residual_ms` | Median absolute residual of alignment landmarks. |

## EEG Fields

EEG columns follow ROI + band naming such as `F_theta`, `P_alpha`, and `O_beta`. In the fused table they are prefixed as `eeg_F_theta`, etc. The primary EEG model set analyzes the nine F/P/O x theta/alpha/beta ROI-band metrics symmetrically; recovery contrasts such as `delta_O_alpha` are supplementary.

| Field | Meaning |
|---|---|
| `hf_ratio_20_40Hz` | Segment high-frequency audit ratio, used as a data-quality indicator. |
| `rms_mean_uV` | Mean channel RMS amplitude for the EEG segment. |
| `peak_to_peak_uV` | Mean channel peak-to-peak amplitude for the EEG segment. |
| `nan_fraction` | Fraction of non-finite EEG samples in the segment. |
| `flat_fraction` | Fraction of flat channels in the segment. |
| `segment_valid_duration` | Whether the EEG segment duration passes the configured minimum duration. |
| `eeg_legacy_hf_flag` | Legacy audit flag for `hf_ratio_20_40Hz > 0.4`; not treated as a universal standard by default. |
| `bad_eeg_quality` | Formal Python-side EEG quality exclusion flag under the configured QC policy. |
| `eeg_qc_reasons` | Semicolon-delimited formal EEG QC reasons. |
| `eeg_qc_policy` | EEG QC policy used, such as `robust`, `legacy_0_4`, `audit_only`, `off`, or `unavailable`. |
| `eeg_subject_quality_exclusion` | Whether the participant crossed the configured bad-scene fraction threshold. |

## Reporting And Nature-Style Metadata

The authoritative inferential table is `outputs/06_models/model_results.csv`. It records grain, GEE family, scope, sample counts, formula, effect scale, confidence interval, raw p value, four-block BH-FDR result, and fit status. Compatibility models, when explicitly requested, live under `outputs/06_models/legacy/` and are excluded from the report evidence chain.

| Field | Meaning |
|---|---|
| `claim_id` | Manuscript claim identifier used by claim strength, claim support, and reviewer response outputs. |
| `support_level` | Evidence strength label. Exploratory labels require bounded manuscript wording. |
| `issue_id` | Reviewer issue identifier in `configs/reviewer_response_map.json`. |
| `response_readiness` | Whether a reviewer response is ready, bounded, or still needs author input. |
| `dataset_id` | Data Availability dataset identifier in `configs/data_availability.json`. |
| `access_route` | Public, controlled, restricted, reused, or request-based availability route. |
| `repository_target` | Repository destination or `AUTHOR_INPUT_NEEDED` placeholder. |
| `identifier` | DOI/accession/record identifier or `AUTHOR_INPUT_NEEDED` placeholder. |
| `figure_id` | Figure contract identifier in `configs/figure_contracts.json`. |
| `source_file` | CSV or table that should be deposited as source data for a manuscript figure. |
| `panel_id` | Figure panel identifier used in panel-level source data. |
| `error_bar_definition` | Explicit definition for plotted uncertainty or a note that no error bar is applicable. |
| `qa_status` | Figure QA status, expected to be `pass` for submission-ready generated figures. |
| `editable_text_policy` | SVG/PDF text-editability policy used by the plotting backend. |
| `image_integrity_note` | Deterministic plotting and image-integrity note linked to each figure contract. |
