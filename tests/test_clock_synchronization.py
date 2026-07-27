from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.fusion.clock_sync import (
    add_eye_epoch_ms,
    align_eye_to_eeg,
    synchronized_timebins,
)


def test_eye_clock_crosses_midnight_and_matches_500hz_eeg_with_2ms_tolerance() -> None:
    eye = pd.DataFrame({
        "Time of Day[HH:mm:ss.ms]": [
            "23:59:59.996", "23:59:59.998", "00:00:00.000", "00:00:00.002", "00:00:00.006",
        ],
        "Recording Time Stamp[ms]": [0, 2, 4, 6, 10],
        "Gaze Point X[px]": [1, 2, 3, np.nan, 5],
        "Gaze Point Y[px]": [1, 2, 3, np.nan, 5],
    })
    dated = add_eye_epoch_ms(eye, {"eye_record_id": "260526_sample"})
    assert dated["eye_epoch_ms"].is_monotonic_increasing
    assert dated["eye_epoch_ms"].diff().iloc[2] == 2

    start = int(dated["eye_epoch_ms"].iloc[1])
    eeg = pd.DataFrame({
        "sample_index_recording": np.arange(100, 105),
        "sample_index_scene": np.arange(5),
        "eeg_epoch_ms": start + np.arange(5) * 2,
        "eeg_scene_time_ms": np.arange(5) * 2,
        "preproc_F3_uV": np.arange(5, dtype=float),
    })
    aligned, qc = align_eye_to_eeg(dated, eeg, "P01", 1, match_tolerance_ms=2)
    assert len(aligned) == len(eye)
    assert aligned.loc[0, "alignment_status"] == "before_eeg_view"
    assert aligned.loc[1:4, "alignment_status"].eq("matched").all()
    assert aligned.loc[3, "Gaze Point X[px]"] != aligned.loc[3, "Gaze Point X[px]"]
    assert aligned.loc[1, "eeg_sample_index"] == 100
    assert aligned.loc[1:4, "match_delta_ms"].abs().max() <= 2
    assert qc["eye_time_monotonic"]
    assert qc["eeg_time_monotonic"]


def test_synchronized_eeg_windows_are_window_specific() -> None:
    srate = 500
    n = 2000
    epoch = 1_800_000_000_000 + np.arange(n) * 2
    t = np.arange(n) / srate
    signal = np.r_[
        np.sin(2 * np.pi * 6 * t[:1000]),
        4 * np.sin(2 * np.pi * 6 * t[1000:]),
    ]
    eeg = pd.DataFrame({
        "eeg_epoch_ms": epoch,
        "preproc_F3_uV": signal,
        "preproc_F4_uV": signal,
    })
    eye = pd.DataFrame({
        "eye_epoch_ms": epoch[::2],
        "Recording Time Stamp[ms]": np.arange(0, 4000, 4),
        "Gaze Point X[px]": 100.0,
        "Gaze Point Y[px]": 100.0,
        "Fixation Index": np.arange(1000),
    })
    trial = pd.Series({
        "participant_id": "P01", "scene_id": 1, "WWR": 0.15,
        "Complexity": 0, "ExperienceGroup": "Low", "block": 1, "position": 1,
        "aoi_json_path": "",
    })
    bins = pd.DataFrame(synchronized_timebins(eye, eeg, trial, bin_size_ms=2000))
    assert bins["bin_index"].tolist() == [0, 1]
    assert bins.loc[1, "eeg_F_theta"] > bins.loc[0, "eeg_F_theta"] * 5
    assert bins["eeg_temporal_resolution"].eq("window_specific").all()
    assert bins["analysis_status"].eq("primary").all()


def test_synchronized_timebins_exclude_trailing_partial_window() -> None:
    srate = 500
    n = 2250
    epoch = 1_800_000_000_000 + np.arange(n) * 2
    signal = np.sin(2 * np.pi * 6 * np.arange(n) / srate)
    eeg = pd.DataFrame({
        "eeg_epoch_ms": epoch,
        "preproc_F3_uV": signal,
        "preproc_F4_uV": signal,
    })
    eye = pd.DataFrame({
        "eye_epoch_ms": epoch[::2],
        "Recording Time Stamp[ms]": np.arange(0, 4500, 4),
        "Gaze Point X[px]": 100.0,
        "Gaze Point Y[px]": 100.0,
    })
    trial = pd.Series({
        "participant_id": "P01", "scene_id": 1, "WWR": 0.15,
        "Complexity": 0, "ExperienceGroup": "Low", "block": 1, "position": 1,
        "aoi_json_path": "",
    })
    bins = pd.DataFrame(synchronized_timebins(eye, eeg, trial, bin_size_ms=2000))
    assert bins["bin_index"].tolist() == [0, 1]
    assert bins["bin_end_ms"].tolist() == [2000, 4000]
