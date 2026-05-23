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
| `FCR` | Fixation count rate. |
| `TFD_ms` | Total fixation duration in milliseconds. |
| `TTFF_ms` | Time to first fixation in milliseconds. |
| `attention_share` | AOI TFD divided by trial total fixation duration. |

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
