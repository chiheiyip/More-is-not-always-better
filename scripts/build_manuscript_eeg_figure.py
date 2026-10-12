"""Render the five manuscript EEG temporal panels from the full family CSV."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paper_analysis.teacher.manuscript_figure import render

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output-prefix',type=Path,required=True)
p.add_argument('--alignment-script',type=Path,required=True)
a=p.parse_args();render(a.input,a.output_prefix,a.alignment_script)
