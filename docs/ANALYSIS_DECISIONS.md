# Analysis Decisions

1. The repository uses Python as the primary analysis stack.
2. The main analysis preserves S1-S5 as separate questionnaire outcomes instead of forcing a single unvalidated composite score.
3. Eye-tracking uses a two-part analysis policy: AOI visited first, then continuous AOI metrics conditionally.
4. EEG primary outcomes are limited to predefined theta/alpha metrics to reduce multiple-comparison and overinterpretation risk.
5. Gender, age, block, position, and recruitment batch are included in model registries when available.
6. Three-level WWR findings are labeled as trend or planned-contrast evidence, not as definitive optimum identification.
7. Claim strength is produced explicitly so the manuscript discussion can be aligned with the evidence.
8. Every reviewer concern must map to an evidence output or an explicit `AUTHOR_INPUT_NEEDED` placeholder.
9. Figure claims must have source-data contracts before final artwork is produced.
10. Data Availability separates raw, processed, source-data, and restricted-access materials.
11. Repository-generated scientific figures use Python/matplotlib only, with editable SVG, vector PDF, high-resolution TIFF, QA PNG, panel source CSV, and a figure QA table.
