"""Build a manuscript copy with checked, targeted paragraph/table/media edits."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from paper_analysis.teacher.manuscript_document import build, inspect

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--payload', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--changes', type=Path)
p.add_argument('--inspect', action='store_true')
a = p.parse_args()
if a.inspect:
    print(json.dumps(inspect(a.source), ensure_ascii=False, indent=2))
else:
    if not all([a.payload, a.output, a.changes]): p.error('Build needs --payload, --output and --changes')
    changes = build(a.source, json.loads(a.payload.read_text(encoding='utf-8-sig')), a.output)
    a.changes.write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Targeted manuscript edits:', len(changes))
