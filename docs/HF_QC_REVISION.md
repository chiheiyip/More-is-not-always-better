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

The original `configs/eeg_qc.json` remains available for explicit historical
reproduction. Revised formal analyses use their recorded No-HF configuration.
New fresh-source execution accepts `eeg.qc_config`; if absent it uses the formal
`configs/eeg_qc_no_hf.json` profile. To reproduce Original QC, explicitly select
`eeg_qc.json`. Generic historical QC APIs retain their original defaults.
No new HF surrogate is introduced. The formal decision never uses significance.

Model comparisons preserve the complete correction families, failure diagnostics,
factor-level CR2/HTZ marginal contrasts, and additive previous-scene specification.
Existing independent eye main results can be reused; all downstream results tied
to changed EEG membership must be rebuilt. Original outputs are never overwritten.

After analysis, `scripts/verify_hf_qc_revision.py --config LOCAL.json --run-root RUN`
independently repeats the 1–40 models, scene-level joint models and synchronized
models, verifies unchanged eye-main evidence, checks new waveforms, and requires
the actual complete four-window bootstrap-repeat proof. The scientific calculation
SHA remains distinct from this verification and document/publication SHAs.

The existing `augment_discussion_handoff.py --action check|prepare|publish` also
accepts `mode: hf-qc-revision`. See `hf_qc_handoff.example.json`. A local writing
payload supplies sources with hashes, exact numeric claim filters and values,
paragraph-source mapping, report text, document blocks and protected manuscript
edits. Preparation validates claims, builds the nine-sheet offline source index
and Word documents, and preserves the input manuscript. Publication additionally
requires current content/visual QA hashes, actual ZIP extraction and offline-link
checks, and archives the complete existing delivery outside its root. Formal
summary pointers are updated with explicit No-HF and eye-reuse provenance;
failed publication restores both the old delivery and formal root entrypoints.
