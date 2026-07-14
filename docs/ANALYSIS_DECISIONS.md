# Analysis Decisions

1. The repository uses Python as the primary analysis stack.
2. The main analysis preserves S1-S5 as separate questionnaire outcomes instead of forcing a single unvalidated composite score.
3. Questionnaire enhancements inspired by `wannaqueen66-create/spss` are adopted only after method correction: Afford4 is supplementary, B items are C1-only, IPQ is subject-level, and WWR polynomial contrasts are trend evidence only.
4. Cronbach's alpha is reported as an internal-consistency diagnostic, not as proof of validity; low or insufficient alpha prevents strong composite-score claims.
5. Shapiro, skewness, and kurtosis are descriptive diagnostics and do not automatically choose or reject the model family.
6. Eye-tracking uses a two-part analysis policy: AOI visited first, then continuous AOI metrics conditionally.
7. EEG primary ROI-band outcomes symmetrically cover F/P/O x theta/alpha/beta: `F_theta`, `F_alpha`, `F_beta`, `P_theta`, `P_alpha`, `P_beta`, `O_theta`, `O_alpha`, and `O_beta`. Recovery contrasts such as `delta_O_alpha` are supplementary.
8. Gender, age, block, position, and recruitment batch are included in model registries when available.
9. Questionnaire, eye-tracking, and EEG inferential outputs use modality-specific available samples. EEG QC never excludes questionnaire or eye observations; the trimodal intersection is reserved for synchronized fusion.
10. WWR45 local-optimum evidence is evaluated within the tested levels using planned contrasts such as WWR45 minus the mean of WWR15 and WWR75; do not extrapolate to untested WWR values.
11. Claim strength is produced explicitly so the manuscript discussion can be aligned with the evidence.
12. Every reviewer concern must map to an evidence output or an explicit `AUTHOR_INPUT_NEEDED` placeholder.
13. Figure claims must have source-data contracts before final artwork is produced.
14. Data Availability separates raw, processed, source-data, and restricted-access materials.
15. Repository-generated scientific figures use Python/matplotlib only, with editable SVG, vector PDF, high-resolution TIFF, QA PNG, panel source CSV, and a figure QA table.
16. EEG raw/preprocessed `.set` files are converted upstream by MATLAB/EEGLAB into scene-level tables; Python remains the primary analysis and fusion stack.
17. EEG quality filtering uses configurable robust thresholds by default. The legacy high-frequency ratio threshold from the upstream EEG repository is retained as an audit/sensitivity flag, not as a universal exclusion standard.
18. High-beta and low-gamma EEG metrics are exploratory because scalp high-frequency activity is more sensitive to muscle artifacts.
19. AOI hit testing accepts `gaze` or `fixation` coordinates, but formal AOI analyses should use fixation points when exported by the eye tracker; `auto` records the actual source used.
20. Eye-tracking uses metric-specific eligibility without a 50%/60%/70%/80% primary hard gate. Those thresholds are automatic sensitivity analyses. Screen and validity checks are explicitly marked not applied unless configured and supported by source fields.
21. Eye dynamics prioritize AOI transitions/rate, normalized transition entropy, angular scanpath per second, saccade rate, window entry and first-window latency. Saccade amplitude/velocity, pupil, blink, and the remaining AOI duration metrics are exploratory unless tied to a registered hypothesis.
22. AOI overlaps assign a fixation to the smaller total-area AOI while preserving all candidates and ambiguity. `outside` is retained for state transitions and removed for named-AOI visit transitions.
23. Coordinate scaling occurs only when the scene manifest explicitly gives the eye-source canvas and the AOI JSON gives its target canvas. Otherwise the contract is reported as unverified; no scale is guessed.
24. Pupil preprocessing masks blink ±100 ms, uses a 3.5-MAD trial filter, interpolates only gaps ≤200 ms, and reports the first two seconds as a scene-early reference rather than a stimulus-preceding baseline. Pupil inference is exploratory and luminance-confounded.
25. The canonical model table contains participant-clustered GEE fits and visible failure diagnostics. Model failure or non-finite covariance never triggers an OLS fallback. Legacy models require an explicit compatibility flag and write only below `06_models/legacy/`.
