from __future__ import annotations

import numpy as np
import pandas as pd

from paper_analysis.stats.timebin import _eligible_timebins, run_timebin_models


def test_timebin_gee_runs_four_parallel_windows_on_original_scene_axis(tmp_path) -> None:
    rng = np.random.default_rng(20260724)
    rows = []
    for subject in range(8):
        for scene_id in range(1, 13):
            wwr = (0.15, 0.30, 0.45)[(scene_id + subject) % 3]
            complexity = ((scene_id // 2) + subject) % 2
            for trim in (0, 5, 10, 15):
                for bin_index in range(6):
                    scene_time_norm = (trim + bin_index * 2) / 60
                    analysis_time_norm = bin_index / 5
                    value = (
                        1 + subject * 0.01 + scene_time_norm * (1 + wwr)
                        + complexity * scene_time_norm
                        + 0.002 * subject * bin_index ** 2
                        + rng.normal(0, 0.05)
                    )
                    rows.append({
                        "participant_id": f"P{subject:02d}", "scene_id": scene_id,
                        "bin_index": bin_index, "time_norm": analysis_time_norm,
                        "scene_time_norm": scene_time_norm,
                        "onset_trim_s": float(trim),
                        "eeg_window_coverage": 1.0, "WWR": wwr,
                        "Complexity": complexity,
                        "ExperienceGroup": "High" if subject % 2 else "Low",
                        "block": 1 if scene_id <= 6 else 2,
                        "position": ((scene_id - 1) % 6) + 1,
                        "class_name": "whole_scene", "eeg_F_theta": value,
                        "visited": bin_index % 2 == 0, "TFD_ms": 100 + value,
                        "attention_share": 0.5, "FCR": 1 + value,
                    })
    timebin_path = tmp_path / "timebins.csv"
    qc_path = tmp_path / "qc.csv"
    pd.DataFrame(rows).to_csv(timebin_path, index=False)
    pd.DataFrame([
        {"participant_id": f"P{subject:02d}", "scene_id": scene_id, "clock_alignment_pass": True}
        for subject in range(8) for scene_id in range(1, 13)
    ]).to_csv(qc_path, index=False)

    outputs = run_timebin_models(
        timebin_path, qc_path, tmp_path / "models",
        expected_onset_trims_s=[0, 5, 10, 15],
        time_column="scene_time_norm",
    )
    models = pd.read_csv(outputs["timebin_model_results"])
    diagnostics = pd.read_csv(outputs["timebin_model_diagnostics"])
    summaries = pd.read_csv(outputs["temporal_scene_summaries"])
    assert not models.empty
    assert models["analysis_resolution"].eq("synchronized_timebin").all()
    assert models["analysis_status"].eq("parallel").all()
    assert models["model_type"].str.contains("independent_robust").all()
    assert models["formula"].str.contains(r"C\(WWR\):scene_time_norm", regex=True).any()
    assert models["formula"].str.contains(r"C\(Complexity\):scene_time_norm", regex=True).any()
    assert models.loc[models["hypothesis_family"].str.startswith("H_time_"), "p_fdr_bh"].notna().all()
    assert diagnostics["model_type"].str.contains("participant_clustered_independent_robust").all()
    assert {"early_mean", "middle_mean", "late_mean", "late_minus_early", "scene_slope_per_time_norm"}.issubset(summaries.columns)
    assert sorted(summaries["onset_trim_s"].unique().tolist()) == [0, 5, 10, 15]
    assert models["p_fdr_bh_parallel"].notna().any()
    sample_flow = pd.read_csv(outputs["parallel_window_sample_flow"])
    stability = pd.read_csv(outputs["parallel_window_stability"])
    assert sample_flow["trials_common"].nunique() == 1
    assert sorted(sample_flow["onset_trim_s"].tolist()) == [0, 5, 10, 15]
    assert not stability.empty


def test_timebin_eligibility_is_scene_level_not_complete_participant() -> None:
    timebins = pd.DataFrame([
        {
            "participant_id": participant,
            "scene_id": scene,
            "bin_index": 0,
            "time_norm": 0.5,
            "eeg_window_coverage": 1.0,
        }
        for participant in ("P01", "P02")
        for scene in (1, 2)
    ])
    qc = pd.DataFrame([
        {"participant_id": "P01", "scene_id": 1, "clock_alignment_pass": True},
        {"participant_id": "P01", "scene_id": 2, "clock_alignment_pass": False},
        {"participant_id": "P02", "scene_id": 1, "clock_alignment_pass": True},
        {"participant_id": "P02", "scene_id": 2, "clock_alignment_pass": True},
    ])

    eligible = _eligible_timebins(timebins, qc)

    assert set(map(tuple, eligible[["participant_id", "scene_id"]].to_numpy())) == {
        ("P01", 1),
        ("P02", 1),
        ("P02", 2),
    }
