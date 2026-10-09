# Discussion supplement for a published delivery

The Discussion supplement updates presentation and interpretation only. It keeps
the existing numerical calculation SHA, statistical tables, Source IDs, report,
and reading guide unchanged. Locally supplied writing and source mapping payloads
must distinguish newly verified results from inherited questionnaire claims.

Use `scripts/augment_discussion_handoff.py --config <local.json> --action check`
for read-only validation, then `prepare` to copy the existing delivery outside the
public root and append source records and paragraph mappings. Local configuration
specifies source/input hashes, additional evidence files, artifact names, external
archive location, current pointer files, and formal artifact targets.

Render `scripts/build_discussion_framework.py <payload.json> <output.docx>` with
the bundled document runtime. Build the augmented nine-sheet indices using
`scripts/build_fresh_handoff_index.mjs`. Review every Word page and the relevant
Excel sheets; independently compare preserved values/formulas/source hashes and
offline links. Record hash-bound results in `artifact_verification.json` before
`publish`. A changed configuration, code version, source, current delivery, or
unreviewed artifact blocks publication. The old root is archived outside the
delivery; the complete ZIP is recreated and checked member by member. Archived
historical ZIP references and numerical version identities remain intact.

Study payloads, manuscript text, statistics, artifacts, and machine configuration
stay local and are not committed to Git.

After exporting indices, run `scripts/fix_hyperlink_caches.py <teacher.xlsx>
<formal.xlsx>` with the bundled Python runtime. This repairs unsupported
HYPERLINK cache diagnostics to their literal friendly labels, preserves the
formulas and numerical cells, and enables native recalculation on open. Check
both formula-mode and cached-value reads before sealing artifact QA.
