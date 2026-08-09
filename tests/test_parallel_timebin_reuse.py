from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.fusion.clock_sync import combine_parallel_synchronized_timebins


def _rows(trim: float) -> list[dict]:
    rows = []
    for elapsed in (trim, trim + 2):
        rows.append({
            "participant_id": "P01",
            "scene_id": 1,
            "onset_trim_s": trim,
            "class_name": "table",
            "bin_start_epoch_ms": 100_000 + elapsed * 1000,
            "bin_end_epoch_ms": 102_000 + elapsed * 1000,
            "scene_elapsed_s": elapsed,
            "analysis_elapsed_s": elapsed - trim,
            "eeg_sample_count": 1000,
            "eye_sample_count": 480,
            "eeg_O_theta": elapsed,
            "TFD": elapsed / 10,
            "analysis_status": "primary",
            "hypothesis_family": "legacy",
        })
    return rows


def test_combine_parallel_timebins_restores_common_scene_axis(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    variants = tmp_path / "variants.csv"
    onset = tmp_path / "onset.csv"
    pd.DataFrame(_rows(10)).to_csv(reference, index=False)
    pd.DataFrame(_rows(0) + _rows(5) + _rows(15)).to_csv(variants, index=False)
    pd.DataFrame([
        {
            "participant_id": "P01", "scene_id": 1,
            "onset_trim_s": trim, "view_dur_s": 60,
        }
        for trim in (0, 5, 10, 15)
    ]).to_csv(onset, index=False)

    outputs = combine_parallel_synchronized_timebins(
        reference, variants, onset, tmp_path / "out"
    )
    table = pd.read_csv(outputs["aligned_synchronized_timebin"])
    assert sorted(table["onset_trim_s"].unique().tolist()) == [0, 5, 10, 15]
    assert set(table["analysis_status"]) == {"parallel"}
    assert np.allclose(
        table["scene_time_norm"], table["scene_elapsed_s"] / 60
    )
    assert (
        table["scene_elapsed_s"]
        == table["analysis_elapsed_s"] + table["onset_trim_s"]
    ).all()
    audit = pd.read_csv(outputs["parallel_timebin_alignment_audit"])
    assert len(audit) == 6
    assert audit["absolute_clock_alignment_pass"].all()
