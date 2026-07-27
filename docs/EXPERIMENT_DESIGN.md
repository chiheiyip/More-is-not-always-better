# Experiment design and scene-order contract

The canonical trial key is `participant_id + scene_id`. Here `scene_id` is the
presentation slot (1–12), not a fixed condition number.

## Authoritative presentation sequences

| Scheme | Questionnaire `Order` | Collection rule | Block | Positions 1–6 |
| --- | --- | --- | --- | --- |
| order1 | 1 | all dates | 1 | `C1W45`, `C0W15`, `C1W75`, `C0W45`, `C1W15`, `C0W75` |
| order1 | 1 | all dates | 2 | `C0W45`, `C1W45`, `C0W75`, `C1W75`, `C0W15`, `C1W15` |
| order2 | 2 | before 2026-05-01 | 1 | `C1W45`, `C0W15`, `C1W75`, `C0W75`, `C1W15`, `C0W45` |
| order2 | 2 | before 2026-05-01 | 2 | `C0W15`, `C1W15`, `C1W45`, `C0W75`, `C0W45`, `C1W75` |
| neworder2 | 2 | on/after 2026-05-01 | 1 | `C0W75`, `C1W15`, `C0W45`, `C1W75`, `C0W15`, `C1W45` |
| neworder2 | 2 | on/after 2026-05-01 | 2 | `C1W15`, `C0W15`, `C1W75`, `C0W75`, `C1W45`, `C0W45` |

Both `order2` and `neworder2` are recorded as `2` in the questionnaire. The
collection date encoded at the start of `eye_record_id` distinguishes them.
After resolving the scheme, `scene_id = (block - 1) * 6 + position`.

## Simple eye-folder names

A folder such as `1-C0W15` means only:

- `1`: block 1;
- `C0`: low complexity;
- `W15`: WWR 15.

The prefix does not encode Order or within-block position. A simple folder
cannot be mapped to `scene_id` without questionnaire `Order` and collection
date; intake fails explicitly when Order is missing.

Complexity is always parsed from the condition label: `C0 = 0` (low) and
`C1 = 1` (high). Folder group labels and WWR values never determine Complexity.

## Modality-specific analysis and order contract

The complete prepared questionnaire table in `questionnaire_long.csv` is the
canonical questionnaire model source. Questionnaire and eye-tracking models
do not inherit EEG exclusions. EEG models use the EEG-QC-passed scene set.
The shared trimodal keep set in `analysis_qc_exclusions.csv` is reserved for
synchronized cross-modal analyses.

For formal order analyses, `trial_index = scene_id = (block - 1) * 6 +
position`. The first scene of block 2 is marked descriptively by
`break_before_trial = 1` because it follows the registered 120-second break.
Lagged previous-condition variables are constructed within
`participant_id + block`; position 1 of both blocks has no lag and is excluded
from the carryover sensitivity model. No previous-condition variable crosses
the block boundary.
