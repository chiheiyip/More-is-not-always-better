# Fresh eye and EEG results

`all-results --scope eye-eeg --fresh-from-source --config <local.json>` starts
from current preprocessed SET/FDT and original eye CSVs. Use `--dry-run` for a
read-only input/runtime preflight. The example configuration supplies measurement
inputs, runtime locations and optional exact hashes for corrected source files.
No questionnaire outcome models run; original Q1.4 supplies grouping covariates.

Formal execution now enforces the exact numerical environment and a frozen AOI
pixel reference. See [analysis_environment_lock.md](analysis_environment_lock.md).
Old configurations without `eye.aoi_pixel_lock` and its SHA256 cannot start a
new fresh run. Original completed manifests retain their actual old versions.

The coordinator preserves the original exporter, HF/RMS/peak-to-peak QC,
median + 3.5 MAD thresholds, >30% subject exclusion, four parallel windows,
eye 60% main and 50/70% sensitivity, and existing statistical methods.
The cohort is recomputed and is never required to contain 42 people.
The 1–40 sensitivity shares the current 1–45 sample, PSD and band numerators.
Factor-level tests remain a documented equal-margin CR2/HTZ reimplementation;
the historical script provenance gap remains open.

Historical numerical reuse and `--skip-r` are rejected. `--resume` only accepts
sealed stages inside the same run with matching SHA, config, source and output
hashes. An interrupted unsealed stage is preserved and requires a new run.
Current source hashes are checked before and after execution. Original waveforms
are never filtered, rereferenced, or ICA-edited by this workflow.

After tests and repository publication, execute a new run ID. Build the reader
artifacts using `publish_latest_teacher_results.py document --request-config
<local.json>`. The nine-sheet index uses the bundled Artifact Tool (make its
node_modules available to the temporary builder). Review workbook previews and
all Word pages, validate numbers/relative links and register reviewed artifact
SHA256 in `artifact_verification.json`. Then use the same command with `finalize`.
`--promote` also performs finalization, and requires those completed reviews.

Publication archives the old teacher delivery outside its root with rollback,
validates ZIP contents and source-copy hashes, and promotes the scoped results.
The latest pointer explicitly states eye/EEG scope, while the old questionnaire
run and its provenance remain historical. Large waveform/PSD caches stay in the
new run directory. Optional historical comparison paths are read only after
fresh calculations complete and never supply model inputs.

The numerical calculation SHA and later verification/publication SHA are recorded
separately. Presentation fixes must never relabel existing models as newly fitted.
The standalone `verify_fresh_waveform_psd.py --run-root <completed-run>` reads
current FDT waveforms and source hashes, independently runs SciPy Welch and checks
band integrals against the committed MATLAB cache. Its default sample includes a
locked corrected participant when that participant survived QC; all cached
frequency grids are registered. It does not alter the analysis or source files.

Finalization archives outdated root summaries and indices, updates the formal
nine-sheet index and latest pointers, and keeps prior questionnaire provenance.
The document builder uses the bundled Python runtime (`fresh.artifact_python`
can specify it explicitly); the workbook uses the bundled Node/Artifact Tool.
Both teacher and formal indices, and all Word pages, require recorded review.
The teacher index uses relative links inside the portable flat-source folder;
the formal root index links to the current published delivery files.

All counts refer to their declared grain. Synchronized TimeBins counts table
rows with both AOI and onset-window dimensions, rather than unique physical bins.
A missing Stage 3 S3 outcome input leaves questionnaire-prediction models and
S3-specific intersections deferred; it does not claim these as freshly verified.
