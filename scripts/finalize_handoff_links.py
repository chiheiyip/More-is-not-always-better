"""Finalize constant hyperlink display caches without changing formulas or data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paper_analysis.teacher.xlsx_hyperlinks import repair

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
print('Constant HYPERLINK display caches repaired:', repair(a.source, a.output))
