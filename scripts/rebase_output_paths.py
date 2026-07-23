#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".tsv"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebase absolute provenance paths after a validated candidate directory is promoted."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    changed: list[dict[str, object]] = []
    replacements = [
        (args.old_prefix, args.new_prefix),
        (
            args.old_prefix.replace("\\", "\\\\"),
            args.new_prefix.replace("\\", "\\\\"),
        ),
    ]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        encoding = "utf-8-sig" if path.suffix.lower() in {".csv", ".tsv"} else "utf-8"
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        count = sum(text.count(old) for old, _ in replacements)
        if not count:
            continue
        for old, new in replacements:
            text = text.replace(old, new)
        path.write_text(text, encoding=encoding)
        changed.append({"file": str(path.relative_to(root)), "replacements": count})
    audit = root / "11_audit" / "output_path_rebase.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        json.dumps(
            {
                "old_prefix": args.old_prefix,
                "new_prefix": args.new_prefix,
                "changed_files": changed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Rebased {sum(int(row['replacements']) for row in changed)} paths in {len(changed)} files.")


if __name__ == "__main__":
    main()
