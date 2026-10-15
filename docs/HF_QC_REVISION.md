# HF ratio QC audit and conditional formal revision

`scripts/run_teacher_analysis.py eeg-hf-qc-revision --config LOCAL.json --outdir NEW_RUN --dry-run`
checks the locked Python/R environment and required sources without writing outputs.
Remove `--dry-run` to execute. `--phase qc` freezes paired QC; `--phase audit`
also fits the complete EEG comparison and writes the evidence-based decision.
The default `--phase all` continues to formal reanalysis only after all five
methodological conditions pass. `--resume` requires matching code, configuration,
environment, inputs, and sealed stage output hashes. Real-data execution requires
clean committed code. Local configurations and all research results stay outside Git.

Original and No-HF start with the same unfiltered four-window features. Only the
HF entry in `robust_metrics` is removed. Remaining median + 3.5 raw-MAD thresholds
are identical, using the full feature population separately for each window.
The participant exclusion rule is strictly greater than 30%, computed before
participant exclusion propagates to every trial. Both decisions are saved.
The recorded `near_boundary` field is not an active exclusion criterion.

The original QC default remains available for historical reproduction. Revised
formal analyses use their explicitly recorded No-HF QC configuration. Fresh-source
execution accepts `eeg.qc_config`; if absent it retains the repository default.
No new HF surrogate is introduced. The formal decision never uses significance.

Model comparisons preserve the complete correction families, failure diagnostics,
factor-level CR2/HTZ marginal contrasts, and additive previous-scene specification.
Existing independent eye main results can be reused; all downstream results tied
to changed EEG membership must be rebuilt. Original outputs are never overwritten.
