#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_analysis.fusion.clock_sync import combine_parallel_synchronized_timebins


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and combine legacy 10-second and 0/5/15-second "
            "synchronized exports into one four-window parallel table."
        )
    )
    parser.add_argument("--reference-timebins", required=True)
    parser.add_argument("--variant-timebins", required=True)
    parser.add_argument("--eeg-onset-trials", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outputs = combine_parallel_synchronized_timebins(
        reference_timebin_csv=args.reference_timebins,
        variant_timebin_csv=args.variant_timebins,
        eeg_onset_trial_csv=args.eeg_onset_trials,
        outdir=args.outdir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
