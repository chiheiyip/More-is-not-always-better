# Data Dictionary

## Core Keys

| Field | Meaning |
|---|---|
| `participant_id` | Canonical participant identifier across questionnaire, eye, and EEG data. |
| `scene_id` | Trial/scene identifier within participant. |
| `condition_id` | Combined condition label, usually derived from WWR and Complexity. |
| `WWR` | Window-to-wall ratio condition. |
| `Complexity` | Visual complexity condition. |
| `block` | Experimental block or round block. |
| `position` | Within-block presentation position; used for order/fatigue diagnostics. |
| `round` | Viewing round when available. |

## Participant Fields

| Field | Meaning |
|---|---|
| `ExperienceRaw` | Original questionnaire response for sports/table-tennis experience. |
| `ExperienceGroup` | Analysis grouping, usually `Low` or `High`. |
| `Gender` | Participant gender, included as a model covariate. |
| `Age` | Participant age, included as a model covariate when available. |
| `RecruitmentBatch` | Original or supplementary recruitment batch. |
| `SupplementFlag` | Whether the participant belongs to the supplementary recruitment batch. |
| `exclude` | Participant-level exclusion flag. |
| `ExcludeReason` | Participant-level exclusion reason. |

## Questionnaire Fields

`S1`-`S5` are scene-level subjective items and are analyzed separately. `B1`-`B3` are supplementary C1-related items. `IPQ` is treated descriptively unless explicitly modeled.

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

## Fusion And Synchronization Fields

| Field | Meaning |
|---|---|
| `aligned_scene_table.csv` | Scene-level EEG + eye AOI projection from the canonical trial index. |
| `aligned_timebin_table.csv` | Time-bin eye AOI projection with scene-level EEG columns attached. |
| `sync_qc.csv` | Per-trial duration and scene-count synchronization QC. |
| `duration_delta_s` | Eye duration minus EEG viewing duration. |
| `duration_mismatch` | Whether duration delta exceeds the configured tolerance. |
| `time_sync_slope` | Participant-level affine eye-to-EEG time mapping slope. |
| `time_sync_offset_ms` | Participant-level affine eye-to-EEG time mapping offset. |
| `median_abs_residual_ms` | Median absolute residual of alignment landmarks. |

## EEG Fields

EEG columns follow ROI + band naming such as `O_theta`, `F_theta`, `O_alpha`. In the fused table they are prefixed as `eeg_O_theta`, etc.

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
