from __future__ import annotations

import numpy as np
import pandas as pd

from paper_analysis.eye_tracking.aoi import PolygonAOI
from paper_analysis.eye_tracking.dynamics import (
    build_fixation_sequence,
    build_transition_events,
    build_transition_matrix,
    compute_dynamic_eye_metrics,
)


def test_overlap_uses_smaller_aoi_and_outside_transition_is_retained() -> None:
    aois = [
        PolygonAOI("large", 1, [(0, 0), (100, 0), (100, 100), (0, 100)]),
        PolygonAOI("small", 1, [(25, 25), (75, 25), (75, 75), (25, 75)]),
    ]
    frame = _fixation_frame(
        times=[0, 100, 200, 300],
        ids=[1, 2, 3, 4],
        xs=[50, 10, 150, 50],
        ys=[50, 10, 150, 50],
    )
    fixations = build_fixation_sequence(frame, aois, {"aoi_image_width_px": 200, "aoi_image_height_px": 200})
    assert fixations.loc[0, "aoi_state"] == "small"
    assert bool(fixations.loc[0, "aoi_ambiguous"])
    assert fixations.loc[0, "aoi_candidates"] == "large;small"

    transitions = build_transition_events(fixations)
    named = transitions.loc[transitions["transition_scope"].eq("named_aoi")]
    large_to_small = named.loc[(named["from_aoi"] == "large") & (named["to_aoi"] == "small")]
    assert len(large_to_small) == 1
    assert bool(large_to_small.iloc[0]["passed_outside"])
    state = transitions.loc[transitions["transition_scope"].eq("state")]
    assert set(zip(state["from_aoi"], state["to_aoi"])) >= {("large", "outside"), ("outside", "small")}


def test_transition_matrix_is_directional_and_probabilities_are_conditional() -> None:
    transitions = pd.DataFrame({
        "transition_scope": ["named_aoi"] * 4,
        "from_aoi": ["A", "A", "A", "B"],
        "to_aoi": ["B", "B", "C", "A"],
    })
    matrix = build_transition_matrix(transitions).set_index(["from_aoi", "to_aoi"])
    assert matrix.loc[("A", "B"), "transition_count"] == 2
    assert np.isclose(matrix.loc[("A", "B"), "transition_probability"], 2 / 3)
    assert np.isclose(matrix.loc[("B", "A"), "transition_probability"], 1.0)


def test_scanpath_and_angular_length_do_not_cross_large_gap() -> None:
    aois = [PolygonAOI("A", 1, [(-1, -1), (20, -1), (20, 20), (-1, 20)])]
    frame = _fixation_frame(
        times=[0, 1000, 7001, 8001],
        ids=[1, 2, 3, 4],
        xs=[0, 3, 100, 100],
        ys=[0, 4, 100, 105],
    )
    directions = [(1, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0)]
    for axis, i in zip("XYZ", range(3)):
        frame[f"Gaze Direction Left {axis}"] = [v[i] for v in directions]
        frame[f"Gaze Direction Right {axis}"] = [v[i] for v in directions]
    result = compute_dynamic_eye_metrics(frame, aois, {"aoi_image_width_px": 100, "aoi_image_height_px": 100})
    assert np.isclose(result.trial_metrics["scanpath_length_px"], 10.0)
    assert np.isclose(result.trial_metrics["angular_scanpath_deg"], 90.0)
    assert result.fixations["segment_id"].nunique() == 2


def test_revisit_entropy_and_event_indices_are_deduplicated() -> None:
    aois = [
        PolygonAOI("A", 1, [(0, 0), (40, 0), (40, 40), (0, 40)]),
        PolygonAOI("B", 1, [(50, 0), (90, 0), (90, 40), (50, 40)]),
    ]
    frame = _fixation_frame(times=[0, 100, 300], ids=[1, 2, 3], xs=[10, 60, 10], ys=[10, 10, 10])
    frame["Fixation Duration[ms]"] = [50, 50, 50]
    frame["Saccade Index"] = [7, 7, 8]
    frame["Saccade Amplitude[px]"] = [10, 10, 20]
    frame["Saccade Velocity Average[px/ms]"] = [1, 1, 2]
    frame["Saccade Velocity Peak[px/ms]"] = [2, 2, 4]
    frame["Blink Index"] = [np.nan, 4, 4]
    frame["Blink Duration[ms]"] = [np.nan, 80, 80]
    result = compute_dynamic_eye_metrics(frame, aois)
    assert result.trial_metrics["aoi_transition_count"] == 2
    assert np.isclose(result.trial_metrics["transition_entropy_bits"], 1.0)
    assert np.isclose(result.trial_metrics["transition_entropy_normalized"], 1.0)
    revisit = result.revisit_by_aoi.set_index("class_name")
    assert revisit.loc["A", "median_revisit_latency_ms"] == 250
    assert result.trial_metrics["saccade_count"] == 2
    assert result.trial_metrics["saccade_amplitude_median_px"] == 15
    assert result.trial_metrics["blink_count"] == 1


def test_pupil_blink_mask_short_gap_and_single_eye_fallback() -> None:
    frame = pd.DataFrame({
        "Recording Time Stamp[ms]": [0, 100, 200, 300, 2100, 2200],
        "Pupil Diameter Left[mm]": [3.0, 3.0, np.nan, 3.0, 3.2, 3.2],
        "Pupil Diameter Right[mm]": [np.nan] * 6,
        "Blink Index": [np.nan, np.nan, 1, np.nan, np.nan, np.nan],
        "Blink Duration[ms]": [np.nan, np.nan, 50, np.nan, np.nan, np.nan],
    })
    result = compute_dynamic_eye_metrics(frame, [])
    assert np.isclose(result.trial_metrics["pupil_early_reference_mm"], 3.0)
    assert np.isclose(result.trial_metrics["pupil_post_early_delta_mm"], 0.2)
    assert result.trial_metrics["pupil_binocular_coverage_ratio"] == 0
    assert "exploratory" in result.trial_metrics["pupil_interpretation_tier"]


def _fixation_frame(times: list[float], ids: list[int], xs: list[float], ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "Recording Time Stamp[ms]": times,
        "Fixation Index": ids,
        "Fixation Point X[px]": xs,
        "Fixation Point Y[px]": ys,
        "Gaze Point X[px]": xs,
        "Gaze Point Y[px]": ys,
        "Fixation Duration[ms]": [50] * len(times),
    })
