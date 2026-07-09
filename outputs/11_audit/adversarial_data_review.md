# Adversarial Data Review

This review audits whether the generated result package presents data clearly without hiding grain, QC, model, or smoke-run caveats.

Overall status: **pass_with_warnings**

| check_id | status | detail |
| --- | --- | --- |
| plain_results_present | PASS | plain result table has rows |
| significance_results_present | PASS | significance table has rows |
| grain_is_explicit | PASS | all plain rows carry a grain |
| aoi_row_scene_trial_distinction | PASS | retained fusion has 453 scene trials and 1130 AOI-expanded rows; report both separately |
| fallback_rows_are_flagged | PASS | fallback rows flagged: 81 |
| warning_rows_are_flagged | PASS | warning rows flagged: 151 |
| significance_has_interpretation_notes | PASS | each significance row has a plain-language note |
| figure_qa | PASS | non-pass figure QA rows: 0 |
| data_availability_placeholders_explicit | WARN | AUTHOR_INPUT_NEEDED dataset rows: 5 |
| smoke_scope_marked | INFO | full or historical result package; not marked as smoke |

## Review Focus

- Scene-trial counts and AOI-expanded row counts are separated.
- Significance rows keep model type, fallback flags, warning flags, and interpretation notes.
- EEG and WWR language is bounded by design and QC caveats.
- Smoke runs are not treated as final paper results.
