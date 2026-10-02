# EEG denominator sensitivity

Run the standalone teacher command with a local copy of
`configs/eeg_denominator_sensitivity.example.json`:

```powershell
python scripts/run_teacher_analysis.py eeg-denominator-sensitivity --config configs/eeg_denominator_sensitivity.local.json --dry-run
python scripts/run_teacher_analysis.py eeg-denominator-sensitivity --config configs/eeg_denominator_sensitivity.local.json --outdir D:/study-local/new-run
```

The command requires a committed, clean repository and a new output directory.
Read-only preflight checks the archived four-window sample, design, epoch fields,
waveform pairs, launchers and R dependencies. MATLAB checks the actual waveform,
ROI and spectrum validity during execution. EEGLAB does not run: existing SET
metadata and single-precision little-endian FDT data are read directly, with no
filtering, ICA, reference change or QC recomputation.
R receives exact UTF-8 input copies under a temporary ASCII working path to
avoid legacy Windows command-line filename conversion; Python copies the
unchanged results back into the requested Unicode output directory.

## Three versions

A is the archived 0805 model input. B is reproduced 1–45 Hz power from the current
preprocessed waveforms. C uses exactly B's spectra and numerators with a 1–40 Hz
denominator. B→C isolates the denominator effect. A→B quantifies reproduction
drift; A→C checks manuscript values. Archive keys, covariates, predecessors,
inclusion, QC and epoch bounds are frozen; no invalid row is silently discarded.
The archive's missing Complexity is restored to C0 only with archived Cond=C0
evidence. PreviousComplexity is never rebuilt from the retained subset.
Its missing C0 encoding is decoded only where archived PreviousWWR identifies an
existing predecessor. Historical auxiliary bands/ratios outside the requested
18 outcomes remain archived values and are not modeled; bare theta/alpha/beta
absolute aliases are updated together with their explicit absolute columns.

ROI averaging precedes pwelch. Numeric-window MATLAB pwelch, inclusive sample
endpoints, rounded onset removal, original frequency masks and trapz are retained.
Each participant MAT cache stores complete f/pxx and spectrum parameters. Cache
identity includes source SHA256, trial bounds, ROI, PSD parameters and MATLAB
implementation hashes. Every artifact hash must match on reuse; incomplete or
modified caches are refused. Relative and historical QC denominators are separate.
The 40–45 fraction differs from (P1–45−P1–40)/P1–45 because the discrete masks leave
a trapezoid between their boundary bins. Both quantities are reported per spectrum.

## Statistical contract

ML random-intercept LMM and participant-clustered CR2 coefficient inference remain
unchanged. Core Model0, PreviousScene and Block1 are retained. PreviousScene uses
PreviousWWR + PreviousComplexity, reflecting actual historical execution; the
manuscript interaction description is recorded as inconsistent.

Factor-level inference is explicitly reimplemented, not recovered historical code:
six equal-weight marginal contrasts with CR2 covariance and HTZ Wald tests. WWR
main effects average over Complexity and ExperienceGroup; the latter main effects
average over all three WWR levels and both levels of the other binary factor.
Contrast matrices and denominator degrees of freedom are saved. Multi-df F tests
have no fabricated beta or confidence interval. Single-df contrasts include
unadjusted 95% intervals. Near-zero beta percentage changes are left unavailable.

Within-window and joint BH families are separate: coefficient core relative144,
expanded relative324, expanded absolute324; factor96/216/216; temporal72/72;
previous core48. Failed inference leaves the complete family present and final q
unavailable. The A version must pass 240 archived core Model1 coefficients at
beta1e-8 and p1e-6, plus the independently checked factor table at p/q1e-6.
B/C absolute input and output equality is asserted before reuse.

CSV comparisons include all prescribed families, numerical changes, raw p,
window q, joint q, direction and significance flips. A Chinese report maps the
manuscript claims to these tests. Status complete means the analysis is complete,
not that conclusions must remain unchanged. Local research results and PSD caches
are never Git content. The dedicated sensitivity result pointer is distinct from
the existing multimodal result entry.

`all_model_coefficients_unadjusted` also pairs intercepts, covariates and core
Model0/Block1/PreviousScene coefficients without inventing additional BH families.
Their q fields are deliberately unavailable. The companion
`scripts/export_eeg_denominator_workbook.mjs` exports typed comparison tables via
the bundled Artifact Tool; statistics remain owned by the R/Python pipeline.
Generate its typed input with `python scripts/prepare_eeg_denominator_workbook.py
<completed-run-directory>`. Workbook tables contain aggregate model results and
diagnostics, without trial-level participant names. Preserve both raw p and both
q scopes; the main sheet judges direction and joint-BH significance per claim.
