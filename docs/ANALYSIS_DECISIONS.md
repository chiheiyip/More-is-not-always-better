# Analysis Decisions

1. The repository uses Python as the primary analysis stack.
2. The main analysis preserves S1-S5 as separate questionnaire outcomes instead of forcing a single unvalidated composite score.
3. Questionnaire enhancements inspired by `wannaqueen66-create/spss` are adopted only after method correction: Afford4 is supplementary, B items are C1-only, IPQ is subject-level, and WWR polynomial contrasts are trend evidence only.
4. Cronbach's alpha is reported as an internal-consistency diagnostic, not as proof of validity; low or insufficient alpha prevents strong composite-score claims.
5. Shapiro, skewness, and kurtosis are descriptive diagnostics and do not automatically choose or reject the model family.
6. Eye-tracking uses a two-part analysis policy: AOI visited first, then continuous AOI metrics conditionally.
7. EEG primary outcomes are limited to predefined theta/alpha metrics to reduce multiple-comparison and overinterpretation risk.
8. Gender, age, block, position, and recruitment batch are included in model registries when available.
9. Three-level WWR findings are labeled as trend or planned-contrast evidence, not as definitive optimum identification.
10. Claim strength is produced explicitly so the manuscript discussion can be aligned with the evidence.
11. Every reviewer concern must map to an evidence output or an explicit `AUTHOR_INPUT_NEEDED` placeholder.
12. Figure claims must have source-data contracts before final artwork is produced.
13. Data Availability separates raw, processed, source-data, and restricted-access materials.
14. Repository-generated scientific figures use Python/matplotlib only, with editable SVG, vector PDF, high-resolution TIFF, QA PNG, panel source CSV, and a figure QA table.
15. EEG raw/preprocessed `.set` files are converted upstream by MATLAB/EEGLAB into scene-level tables; Python remains the primary analysis and fusion stack.
16. EEG quality filtering uses configurable robust thresholds by default. The legacy high-frequency ratio threshold from the upstream EEG repository is retained as an audit/sensitivity flag, not as a universal exclusion standard.
17. High-beta and low-gamma EEG metrics are exploratory because scalp high-frequency activity is more sensitive to muscle artifacts.
18. AOI hit testing accepts `gaze` or `fixation` coordinates, but formal AOI analyses should use fixation points when exported by the eye tracker; `auto` records the actual source used.
19. Eye-tracking validity, screen-bounds, timestamp-segment, image-size, and AOI-overlap checks are audit outputs by default. Hard exclusions must be configured explicitly and re-run, because validity codes and screen/image geometry are device/export-specific.
20. Eye metric expansion follows a bounded hierarchy: `visited`, `FCR`/`FC_rate`, `TFD_ms`, `TTFF_ms`, and `attention_share` remain primary; `FFD_ms`, `MFD_ms`, `RFF`, and `MPD` are exploratory unless pre-registered for a specific manuscript claim.
