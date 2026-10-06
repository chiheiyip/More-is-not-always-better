# Incremental independent eye and EEG audit

Update, test, commit and merge the implementation before running real data.
Use a local copy of `configs/teacher_incremental_audit.example.json`:

```powershell
python scripts/run_teacher_analysis.py incremental-audit --config configs/teacher_incremental_audit.local.json --dry-run
python scripts/run_teacher_analysis.py incremental-audit --config configs/teacher_incremental_audit.local.json --phase eye-process
python scripts/run_teacher_analysis.py incremental-audit --config configs/teacher_incremental_audit.local.json --phase eye-models
python scripts/run_teacher_analysis.py incremental-audit --config configs/teacher_incremental_audit.local.json --phase eeg-source
python scripts/run_teacher_analysis.py incremental-audit --config configs/teacher_incremental_audit.local.json --phase compare
```

The standalone `scripts/run_teacher_incremental_audit.py` exposes the same phases.
Dry-run reads source structure and approvals but writes no outputs. Actual phases
require a clean committed repository. Real inputs, local configurations, caches
and generated results are never committed.

`independent_eye.py` consumes only original eye CSV, original Q1.4 questionnaire,
participant/trial mapping, annotation JSON, images and authorized ValidScene
masks. Precomputed groups/predecessors are discarded. The candidate mapping is
an authorized measurement input, not independent evidence of experimental order;
its provisional human-review notes and unknown projection remain limitations.
The new code never imports production eye preprocessing or modelling helpers.
Raw sample validity uses both eyes ==1. Fixations are keyed by participant/trial/
index, retain one exported duration, use median coordinates and record conflicts.
Pixel masks round polygon vertices and fixation coordinates to their nearest
integer; declared overlap priority is recorded. C0 Equipment is structural NA.
No unvisited TTFF is replaced by a constant. Boundary masks use square dilation/
erosion by 5 pixels; 10 pixels is a separately recorded option. Spherical weighting
requires confirmed equirectangular projection.

Independent trial outputs precede R models. The R script directly calls lme4,
glmmTMB, clubSandwich and emmeans. It records fixed references (WWR15, C0, High,
Female, new order2), ML LMM, beta/binomial/negative-binomial GLMM, and exact formula.
Beta fits with boundaries use the interior conditional part plus a separate
Visited GLMM; no arbitrary epsilon is added. Logit-LMM is sensitivity. LMM
coefficient inference uses participant CR2/Satterthwaite; factor tests use equal
weight CR2/HTZ. GLMM tests use Wald inference. Family A/B correction is separate
for each effect, variant and LOPO omission with three prescribed outcomes. Missing
members leave final q unavailable. WWR contrasts use outcome-specific three-test
Holm; C0-C1 is explicitly labelled. Modelling failures and warnings are retained.
All independent outputs are sealed before access to production trial/results.
This is computational isolation; the analyst has previously seen project summaries.

The acquisition source map supplies paths only. Original EASY files are independently
read as eight EEG columns, three accelerometer columns, trigger and epoch-ms. Raw
sample count, timestamps, INFO metadata and 7/8 trigger latencies are compared with
SET, with missing/unresolved people explicitly retained in the source inventory.

EEG MATLAB audit reads current preprocessed SET/FDT directly, with no filtering,
ICA or source mutation. It independently pairs adjacent 7/8 events, retains
inclusive rounded endpoints, trims samples, averages ROI channels in time, and
uses numeric-window pwelch, 50% overlap, next-power-of-two NFFT and mask+trapz.
QC features and historical 1-45 denominator are distinct from 1-40 sensitivity.
QC thresholds and participant exclusion are rebuilt per variant; all identities
are compared. Current-source QC drift is distinguished from archived-feature QC
verification, which cannot substitute for unavailable original acquisition data.
The existing independent A/B/C audit is reused only with its recorded source,
output and relevant statistical-code hashes intact. A small new core refit adds
ordinary SE and diagnostics absent from the previous record. Prior models and
bootstrap retain their original calculation provenance.

ICA special audit was explicitly cancelled. Retain the researcher's statement
about GUI ICA without component removal and existing evidence limitations; do
not generate an ICA audit table, add artifact removal, or label absent records as
proof of removal/nonremoval.

Current-source QC drift is an additional D/E sensitivity, not a change to the
historical frozen A/B/C sample. `--phase eeg-current-models` freezes common-QC
identities across all four windows, keeps absolute powers/design identical for
D/E, fits independent CR2/HTZ models and preserves complete BH family skeletons.
`--phase eeg-compare` can independently verify EEG before eye comparisons.
Raw/SET scene events are compared by adjacent 7→8 endpoint identity; ordinal
marker comparisons alone are insufficient in the presence of extra triggers.
Generic raw User labels require an explicit `unresolved_export_labels` entry and
a participant-bearing filename; linkage remains unverified and must be disclosed.
Every phase records its own committed code SHA; older computed stages retain
original execution provenance.

The first sealed eye audit omitted nonnegative finite gaze coordinates from its
validity definition and deduplicated unfiltered sample rows. Keep that initial
run intact. The corrected run requires both eye codes ==1 and finite nonnegative
Gaze Point X/Y, then deduplicates only valid sample rows; TTFF retains the original
recording time origin. It is explicitly a post-comparison revision, not a second
blind audit. Bounded primary shares use independent ordered-beta mixed models;
conditional interior beta, logit-LMM and a raw-share CR2 companion are separate
supplements. The formal CSV contains all candidate trials, so comparisons must
apply its explicit IncludedPrimary flag. `reuse-eeg` copies already verified EEG
stages byte-for-byte, retaining original computation SHAs and documenting reuse.

`eye-canonical` independently reconstructs the recorded PIL polygon/floor sample
semantics from frozen valid fixations, preserving rounded/cv2 outputs as a pixel
rule sensitivity. It fits only six main outcomes and two share CR2 companions.
The comparison uses like inference layers; primary likelihood and companion CR2
estimates, p values, CIs and correction families must never be mixed.

The focused publisher accepts the same `--request-config`, with
`request_type=eye_eeg_incremental`, for read-only `--dry-run`, `prepare`,
`document` and `finalize`. It creates the fixed nine-sheet handoff payload and
requires byte-identical evidence copies, sealed calculations, independent source
reader agreement, and recorded visual QA before publication. Old deliveries are
archived outside the active root transactionally. Research data and artifacts
remain local; computed phase SHAs are retained separately from publication SHA.
