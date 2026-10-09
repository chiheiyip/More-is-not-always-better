"""Prepare/check/publish a reviewed Discussion supplement from a local config."""
import argparse
import json
from pathlib import Path

from paper_analysis.teacher.discussion_handoff import check, prepare, publish

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--action', choices=['check', 'prepare', 'publish'], default='check')
args = p.parse_args()
config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
repo = Path(__file__).resolve().parents[1]
if args.action == 'check':
    print('Read-only checks passed:', len(check(config)), 'existing sources')
elif args.action == 'prepare':
    print(prepare(config, repo))
else:
    print(json.dumps(publish(config, repo), ensure_ascii=False))
