# Analysis Decisions

## Teacher-priority superseding decisions (2026-07-26)

These rules supersede conflicting historical entries below:

1. Formal eye and EEG analysis uses Python orchestration plus locked R
   inference. The existing Python/GEE workflow is compatibility or
   supplementary analysis.
2. Participant registration is the union of modality sources. Eye eligibility
   never depends on valid EEG or questionnaire data; exact intersections are
   used only for named cross-modal sensitivities.
3. Eye tracking uses 60% recomputed valid tracking as the sole primary
   threshold; 50% and 70% are sensitivities. Trial exclusion does not
   automatically become participant exclusion.
4. AOI overlap, unconfirmed coordinates, unreliable ValidScene, stale
   approvals, or fixation-coordinate conflicts are blocking conditions.
5. `PreviousWWR` and `PreviousComplexity` are shifted inside
   `Participant + Block`; position 1 of both blocks is missing.
6. The three WWR values are categorical levels. The teacher workflow does not
   estimate an optimum, inverted-U, continuous WWR, or nonlinear WWR claim.
7. Eye shares use ValidScene TFD as their denominator. C0 Equipment is
   structural NA, and zero shares receive no pseudoconstant.
8. EEG relative power is primary and log10 absolute power is sensitivity.
   Core, secondary, and supplementary ROI × band outcomes are explicitly
   registered; significance cannot promote an outcome or switch the measure.
9. Teacher-priority eye and EEG models use the repository-standard
   `ExperienceGroup` derived from Q1.4. They allow the three specified
   two-way condition/experience interactions plus Gender, Block, centered
   position, and three-level OrderGroup. Q1.5 exercise frequency is not a
   substitute; three-way and condition-by-OrderGroup terms are prohibited.
10. Mixed-model failure is diagnostic evidence and never triggers OLS
    fallback. Bootstrap failure counts are always reported.

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
21. Eye dynamics prioritize AOI transitions/rate, normalized transition entropy, angular scanpath per second, saccade rate, window entry and first-window latency. For the registered order/fatigue-proxy family only, blink count, blink rate, angular scanpath per second, and scene-early pupil change are prespecified; pupil remains luminance-confounded. Other condition-effect uses of pupil, blink, saccade amplitude/velocity, and remaining AOI duration metrics are exploratory.
22. AOI overlaps assign a fixation to the smaller total-area AOI while preserving all candidates and ambiguity. `outside` is retained for state transitions and removed for named-AOI visit transitions.
23. Coordinate scaling occurs only when the scene manifest explicitly gives the eye-source canvas and the AOI JSON gives its target canvas. Otherwise the contract is reported as unverified; no scale is guessed.
24. Pupil preprocessing masks blink ±100 ms, uses a 3.5-MAD trial filter, interpolates only gaps ≤200 ms, and reports the first two seconds as a scene-early reference rather than a stimulus-preceding baseline. Pupil inference is exploratory and luminance-confounded.
25. The canonical model table contains participant-clustered GEE fits and visible failure diagnostics. Model failure or non-finite covariance never triggers an OLS fallback. Legacy models require an explicit compatibility flag and write only below `06_models/legacy/`.
26. Formal order models retain `block + position`; theta/alpha EEG, prespecified eye fatigue proxies, and questionnaire outcomes receive separate BH-FDR families. EEG beta order effects are exploratory.
27. Time-stability sensitivity models test `WWR × trial_index` and `Complexity × trial_index`. Carryover models add previous WWR, previous complexity, order scheme, and the block-2 break marker. These analyses assess risk but cannot prove that fatigue, distraction, or carryover was eliminated.
28. Scene-level models and absolute-clock synchronized 2-second models are parallel, co-primary analysis layers. The former estimates whole-scene average effects; the latter estimates within-scene change, `WWR × time_norm`, and `Complexity × time_norm`. Neither layer overrides the other.
29. Synchronized models use participant-clustered GEE with an independent working correlation and robust sandwich standard errors. Scene and synchronized hypothesis families receive separate BH-FDR correction.
30. EEG absolute time is established once from validated `.easy/.info` provenance and cached beside the `.set/.fdt` data. Routine analysis fails when the cache is absent and never silently rereads the D-drive acquisition archive.
31. Per-sample EEG exports contain the preprocessed `.set/.fdt` waveform in µV. They are not described as raw nV. Scenes are half-open consecutive trigger intervals `[7, 8)`.
32. Eye and EEG clocks are aligned with `clock_offset_ms=0` because both devices used the same Windows system clock. Eye rows use the date from `eye_record_id` plus `Time of Day`; nearest EEG matching is limited to 2 ms. The earlier scene-boundary affine map remains historical diagnostics only.
33. Divergent scene-average and time-dynamic results are reported as scale-dependent evidence. Multimodal support separately records overall-level and temporal-dynamic convergence.
