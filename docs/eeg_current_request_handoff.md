# EEG current-request verification and handoff

Update, test, commit and synchronize code before executing a real request. Preserve
the original sensitivity computation SHA and environment separately from the new
verification SHA. The prior 1–45 primary results remain primary; 1–40 is sensitivity
and final Methods are not frozen. No unrelated modality or 5000-repeat bootstrap
rerun is required for packaging.

Use a local copy of `configs/eeg_request_handoff.example.json`. System R 4.5.3 is
supported; install readxl, R.matlab and clubSandwich in a dedicated library. The
launcher retains the existing R 4.5 user library and clears incompatible inherited
R_HOME/locale variables. Original R remains available to investigate environment
differences; a refit discrepancy is never silently accepted or averaged away.

```powershell
python scripts/publish_latest_teacher_results.py --request-config configs/eeg_request_handoff.local.json --dry-run
python scripts/publish_latest_teacher_results.py verify --request-config configs/eeg_request_handoff.local.json
python scripts/publish_latest_teacher_results.py prepare --request-config configs/eeg_request_handoff.local.json
```

Python and R independently read source CSVs, the selected original questionnaire
columns from its first sheet, and each original MAT PSD. MAT endpoint identities
are checked against the original cache power table. No parser output is handed
to the other source reader. All model input cells, Q1.4 groups, complete families,
PSD integrals and B/C identities must pass. Cited and borderline models are
deduplicated and independently refitted. Equal-weight contrast matrices are built
from a prediction grid independently of the production matrices. Coefficient
tolerance is 1e-8, p/q tolerance 1e-6; source reads/integrals use rtol1e-10/atol1e-12.
Input/code/output hashes guard reuse, and source hashes are checked again afterward.

Prepare writes a focused report, a typed nine-sheet workbook payload, and a source
manifest in the new analysis-root request directory. Author the workbook using the
bundled Artifact Tool exporter with its optional preview directory. Run the
`document` phase to author the guide. Render and review every Word page and the
workbook sheets, validate numerical cells/formulas/relative links, and record
`artifact_verification.json` with artifact SHA256 and `visually_reviewed: true`.
Artifacts cannot be finalized without this verification. Builders/previews stay
outside the delivery folder.

```powershell
python scripts/publish_latest_teacher_results.py document --request-config configs/eeg_request_handoff.local.json
python scripts/publish_latest_teacher_results.py finalize --request-config configs/eeg_request_handoff.local.json
```

Final publication creates a ZIP and checks every archived payload hash. The current
delivery root contains exactly report MD, source index XLSX, guide DOCX, the flat
source folder, and the ZIP. Old contents move intact to a separate archive root;
failed moves roll back. Old source paths receive an archival mapping. Research
data/caches/generated artifacts/local configurations never enter Git. The full
multimodal report remains in the analysis root; `eeg_request_handoff_latest.json`
and `老师本次EEG核查报告.md` identify the focused current request.
