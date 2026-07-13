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

## Cross-modal exclusion contract

The complete prepared questionnaire table remains in `questionnaire_long.csv`
for audit and fusion construction. Once fusion QC produces
`analysis_qc_exclusions.csv`, all questionnaire descriptives, reliability
checks, item models, contrasts, and paper-result summaries use
`questionnaire_analysis_long.csv`, filtered by the same
`participant_id + scene_id` keep set as EEG and eye tracking.
`questionnaire_analysis_sample.csv` records raw, retained, and excluded counts.
