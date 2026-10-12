# Complete manuscript delivery

`scripts/augment_discussion_handoff.py --config <local.json> --action check|prepare|publish`
accepts `mode: manuscript` for a complete multi-document package. Omitting mode
preserves the existing Discussion-only append contract. Check is read-only;
prepare needs a new staging directory outside delivery; publish requires reviewed
artifacts and archives the old delivery outside the active root.

The example config documents the required report, nine-sheet index, reading
guide, Results report, Discussion framework and full manuscript copy. Real
configs, input hashes, result data and documents remain outside Git. Bind the
original manuscript, exact verification proof, analysis configuration, source
manifest and index payload in `inputs`. Existing scientific source bytes and
Source IDs are preserved. Additional tables use new Source IDs and safe flat
filenames. The analysis environment and relevant statistical-code hashes must
match; the passed three-scope exact proof must include actual repeated 80000
EEG bootstrap fits. No fitting occurs in this packaging command.

Artifact builders run separately with the bundled authoring runtimes. Each
staged artifact is bound to content QA and its final hash; DOCX/XLSX artifacts
also require actual visual review. `artifact_verification.json` must record
`files[name].sha256`, `content_verified`, `visually_reviewed`,
`source_and_index_checks_passed`, `manuscript_consistency_passed`, and
`formal_index_sha256`. Failure or invalid inference is a diagnostic rather than
passed model evidence. The scientific calculation SHA is never replaced by the
document-generation SHA.

`scripts/build_manuscript_document.py --source <original.docx> --payload <edits.json>
--output <copy.docx> --changes <change_log.json>` applies targeted edits. Payloads
bind the source SHA and exact paragraph/table-cell text; insertion anchors are
checked. Unedited ZIP members, embedded pictures, relationships and namespaces
are preserved. Destructive edits of paragraphs containing drawings or Word
fields are rejected. Original documents cannot be overwritten. `--inspect`
reads paragraph/table/media structure without writing a document.

Manuscript publication validates ZIP names and every member hash. Formal
document and active-pointer targets are backed up; an entrypoint write failure
restores them and the previous delivery. Targets inside scientific sources,
the original manuscript, delivery or the archive are rejected. Historical
calculation metadata and historical ZIP references stay unchanged.
