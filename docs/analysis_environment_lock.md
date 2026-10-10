# Locked formal calculation environment

The formal teacher workflow requires Windows x64 and Python **3.12.10**.
Install `requirements-analysis.lock.txt` in an isolated `.venv-analysis`.
All eight Python libraries recorded in the 0805 eye manifest are pinned exactly:
Pillow 12.2.0, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, statsmodels 0.14.6,
matplotlib 3.11.0, openpyxl 3.1.5 and python-docx 1.2.0. Additional runtime
dependencies are pinned too. The historical manifest did not record every
transitive dependency, so do not describe the installation as a complete image
of the old operating system or every old dependency.

```powershell
py -3.12 -m venv .venv-analysis
.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.lock.txt
.venv-analysis\Scripts\python.exe scripts/check_analysis_environment.py --config configs/run.local.json
```

The Python guard checks the exact patch version, operating system, architecture
and all locked packages. The R guard probes the actual launcher and selected
libraries before execution **and before historical reuse**. Primary R 4.4.2
and independent R 4.5.3 have separate package snapshots in
`analysis/r/runtime-versions.lock.json`; they must not be substituted for one
another. A mismatch stops before model outputs are created. Artifact authoring
uses its bundled runtime separately; this lock governs numerical calculation.

Primary R also locks the historical Chinese UTF-8 collation and locale categories
and `Asia/Shanghai` time zone. The staged execution applies the same profile as
the preflight; factor ordering must not change merely because the parent shell
uses a C locale. Independent R retains its separately registered locale profile.

## Freeze AOI evidence once, then compare exactly

Use the committed checker with an explicitly declared formal area table and
its original run manifest. Freeze into a new local directory outside Git:

```powershell
.venv-analysis\Scripts\python.exe scripts/check_analysis_environment.py --config configs/run.local.json --freeze-aoi D:/study-local/frozen_aoi/aoi_pixel_lock.json --reference-area D:/study-local/reference/08_AOI_area_report.xlsx --reference-run D:/study-local/reference/run_manifest.json
```

The freeze refuses to overwrite an existing reference. It requires exact integer
pixel counts against the declared original table; floating area shares permit
only XLSX decimal serialization error (`rtol=atol=1e-15`). Later JSON-to-recomputed
mask/area comparisons are exact. It saves the complete masks and
ValidScene arrays, and registers original source hashes, canvas dimensions,
overlap priority, projection, mask bit order, per-mask hashes, area values,
reference calculation SHA and environment-lock SHA.

Put the returned path and SHA256 into `eye.aoi_pixel_lock` and
`eye.aoi_pixel_lock_sha256`. Fresh all-results and eye stage commands require
both and validate them before writing outputs. A changed source, mask,
area, archive or environment stops the run; the reference is never silently
refreshed. The full fresh run fingerprint includes the runtime/AOI checks, so
same-run resume cannot bypass a changed environment.

Choosing the historical Pillow version restores a rasterization convention;
it does not establish that every historical model interpretation is correct.
Record fresh recalculation and historical comparisons separately. In
particular, the 20261008 results were computed with Pillow 10.3.0 and must retain
that original provenance. Version locking does not change EEG QC or repair the
known upstream waveform-history, projection or inference limitations.

## All three calculation paths

Python additionally verifies numerical DLL SHA256 and the effective NumPy/SciPy
OpenBLAS thread counts. R verifies Rblas/Rlapack SHA256, RNG kinds and thread
environment variables. The registered settings are observed historical-compatible
settings, not a new single-thread or optimizer policy.

MATLAB R2026a Update 5, Signal Toolbox 26.1 and EEGLAB 2025.1.0 are registered
from the current installation. `configs/analysis_matlab.lock.json` records actual
function paths/hashes, MKL/LAPACK descriptions, thread count and RNG algorithm.
This is a current-source baseline, not proof of an unrecorded historical install.
Export/PSD functions check the executing process before numerical outputs.

`check_analysis_environment.py --config <config> --scope eye|eeg|joint|all`
reports environment readiness, never a result-verification conclusion. Direct
raw export, clock-cache and clock-synchronization CLI paths also enforce locks.
Synchronization rejects duplicate/missing trial keys and records both modalities,
QC and clock input hashes. Existing formal statistical formulas are retained.

## Offline recovery

Run `archive_analysis_environment.py --phase archive --archive <new-directory>
--independent-r <R-root> --independent-library <package-library>` with the locked
Python. It archives Python base, all exact wheels, both R runtimes and package
library, records all file hashes and generates hash-required offline requirements.
Preserve the returned manifest SHA256 independently from the archive.

Restore with `--phase restore --archive <directory> --target <new-directory>
--manifest-sha256 <trusted-SHA256>`. Restoration downloads nothing and refuses
existing destinations; it checks archive/lock hashes and all restored runtimes.
The portable R bootstrap now uses this archive rather than latest downloads.
MATLAB needs the registered licensed installation; it is checked separately and
is not redistributed in this archive. Numerical CSVs are compared; container
metadata such as XLSX/ZIP timestamps need not be byte-identical.

`verify_analysis_reproducibility.py --request-config <local-json> --outdir <new-directory>`
compares explicit left/right CSVs with declared keys, separately for `eye`, `eeg`
and `joint`. The request lists `config`, `comparisons` (scope, left, right, keys,
model boolean), and scope_notes. It seals inputs and comparisons, treats every
0.05 crossing as failure even within tolerance, and refuses missing test families.
Publish requires the generated `reproducibility_verification.json` and unchanged
evidence. The calculation environment and numerical evidence are separate checks.

EEG bootstrap writes `bootstrap_draw_provenance.json`: participant ordering,
sampling unit, per-outcome starting/ending RNG state, requested draws and draw
digests. `analysis/r/verify_bootstrap_draws.R` replays every sampled participant
sequence without repeating the model fits. Recording digests does not call RNG or
change the sampling algorithm. A real-R simulation repeats the production fits
and checks exact bootstrap estimates as well as replayed draw identities.

## Recalculate against a historical eye package

`scripts/verify_historical_eye_environment.py` has `--phase prepare`, `run`
and `compare`. All phases require `--historical-package` and a new `--outdir`;
prepare also takes `--config`. Preparation verifies the package's recorded
Python versions, R/loaded-package versions, original eye source hashes and
the reference AOI areas. It creates a separate local configuration and mask
lock. Run requires clean committed code, calculates eye Stages 1/2/3 freshly,
and seals stage outputs for same-run resume. Historical model outcomes are
read only after all fresh calculations finish. Comparison checks primary
model keys/fields, missing positions, area values and trial metrics, including
direction/significance crossings separately from numerical tolerances.

Current EEG-derived common-sample results are declared separately: changing
the eye environment does not restore a historical EEG inclusion list. Existing
formal publications and historical archives are preserved by this verifier.

`--phase refit-r --source-run <sealed-fresh-eye-run> --outdir <new-sibling-run>`
can reuse verified Python trial/AOI/boundary tables when only the R runtime
profile changes. It verifies source, environment and stage hashes, keeps the
original Python calculation SHA, and refits all R models with a separately
recorded code SHA. No source waveform or eye CSV needs to be processed again.
