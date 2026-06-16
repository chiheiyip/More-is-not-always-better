from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from more_is_not_always_better.aoi import PolygonAOI as LegacyAOI
from more_is_not_always_better.aoi import compute_metrics as compute_legacy_metrics
from paper_analysis.eye_tracking.aoi import PolygonAOI, compute_aoi_metrics, point_in_poly
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline


def test_paper_aoi_metrics_prefer_fixation_points_and_report_qc() -> None:
    df = _eye_df()
    aois = [
        PolygonAOI("left", 1, [(0, 0), (100, 0), (100, 100), (0, 100)]),
        PolygonAOI("middle", 1, [(50, 0), (150, 0), (150, 100), (50, 100)]),
    ]

    _, metrics = compute_aoi_metrics(
        df,
        aois,
        point_source="auto",
        screen_w=200,
        screen_h=200,
        validity_accepted=("1",),
        timestamp_gap_ms=5000,
    )
    by_class = metrics.set_index("class_name")
    qc = metrics.attrs["eye_qc"]
    overlap = metrics.attrs["aoi_overlap"]

    assert qc["point_source_used"] == "fixation"
    assert qc["analysis_valid_count"] == 4
    assert qc["timestamp_gap_count"] == 1
    assert qc["time_segment_count"] == 2
    assert by_class.loc["left", "FC"] == 2
    assert by_class.loc["left", "TFD_ms"] == 300
    assert np.isclose(by_class.loc["left", "attention_share"], 300 / 450)
    assert np.isclose(by_class.loc["left", "FC_rate"], 2 / 6.1)
    assert by_class.loc["left", "RFF"] == 1
    assert by_class.loc["middle", "FC"] == 2
    assert by_class.loc["middle", "MFD_ms"] == 125
    assert overlap == [{
        "class_a": "left",
        "class_b": "middle",
        "overlap_samples": 2,
        "samples_a": 3,
        "samples_b": 3,
        "overlap_ratio_a": 2 / 3,
        "overlap_ratio_b": 2 / 3,
    }]


def test_paper_aoi_pipeline_writes_qc_and_overlap_outputs(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scene = tmp_path / "scene_manifest.csv"
    eye_csv = tmp_path / "eye.csv"
    aoi_json = tmp_path / "aoi.json"
    participants.write_text("participant_id,exclude\nP01,false\n", encoding="utf-8")
    _eye_df().to_csv(eye_csv, index=False)
    aoi_json.write_text(json.dumps({
        "aoi_classes": {
            "left": [{"points": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
            "middle": [{"points": [[50, 0], [150, 0], [150, 100], [50, 100]]}],
        }
    }), encoding="utf-8")
    scene.write_text(
        "participant_id,scene_id,eye_csv_path,aoi_json_path,WWR,Complexity\n"
        f"P01,1,{eye_csv.as_posix()},{aoi_json.as_posix()},45,1\n",
        encoding="utf-8",
    )

    out = run_eye_pipeline(
        participants,
        scene,
        outdir=tmp_path / "eye_out",
        point_source="fixation",
        screen_w=200,
        screen_h=200,
        validity_accepted=("1",),
    )

    qc = pd.read_csv(out["eye_qc"])
    overlap = pd.read_csv(out["aoi_overlap_summary"])
    metrics = pd.read_csv(out["eye_aoi_trial_long"])
    assert {"analysis_valid_ratio", "point_source_used", "timestamp_gap_count"}.issubset(qc.columns)
    assert qc.loc[0, "point_source_used"] == "fixation"
    assert qc.loc[0, "analysis_valid_count"] == 4
    assert not overlap.empty
    assert {"FFD_ms", "MFD_ms", "RFF", "MPD", "share_pct", "FC_share"}.issubset(metrics.columns)


def test_aoi_boundary_points_are_counted_inside_for_reproducibility() -> None:
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    result = point_in_poly(np.array([100.0, 101.0]), np.array([50.0, 50.0]), poly)
    assert result.tolist() == [True, False]


def test_legacy_aoi_metrics_support_explicit_gaze_or_fixation_point_source() -> None:
    df = pd.DataFrame({
        "Recording Time Stamp[ms]": [0, 100],
        "Gaze Point X[px]": [140, 140],
        "Gaze Point Y[px]": [50, 50],
        "Fixation Point X[px]": [100, 100],
        "Fixation Point Y[px]": [50, 50],
        "Fixation Index": [1, 1],
        "Fixation Duration[ms]": [200, 200],
    })
    aois = [LegacyAOI("edge", 1, [(0, 0), (100, 0), (100, 100), (0, 100)])]

    _, auto_metrics = compute_legacy_metrics(df, aois, point_source="auto")
    _, gaze_metrics = compute_legacy_metrics(df, aois, point_source="gaze")

    assert auto_metrics.loc[0, "point_source_used"] == "fixation"
    assert auto_metrics.loc[0, "FC"] == 1
    assert auto_metrics.loc[0, "samples"] == 2
    assert gaze_metrics.loc[0, "point_source_used"] == "gaze"
    assert gaze_metrics.loc[0, "FC"] == 0
    assert gaze_metrics.loc[0, "samples"] == 0


def _eye_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Recording Time Stamp[ms]": [0, 100, 200, 6100, 6200],
        "Gaze Point X[px]": [10, 10, 10, 10, 10],
        "Gaze Point Y[px]": [10, 10, 10, 10, 10],
        "Fixation Point X[px]": [60, 60, 120, 10, 300],
        "Fixation Point Y[px]": [60, 60, 60, 10, 60],
        "Fixation Index": [1, 1, 2, 3, 4],
        "Fixation Duration[ms]": [100, 100, 150, 200, 100],
        "Validity Left": [1, 1, 1, 1, 0],
        "Validity Right": [1, 1, 1, 1, 0],
        "Pupil Diameter Left[mm]": [3.0, 3.1, 3.2, 3.3, 3.4],
        "Pupil Diameter Right[mm]": [3.2, 3.3, 3.4, 3.5, 3.6],
    })
