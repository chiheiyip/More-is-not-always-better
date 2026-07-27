from __future__ import annotations

import numpy as np
import pandas as pd

from paper_analysis.stats.timebin import run_timebin_models


def test_timebin_gee_is_independent_robust_and_coprimary(tmp_path) -> None:
    rng = np.random.default_rng(20260724)
    rows = []
    for subject in range(8):
        for scene_id in range(1, 13):
            wwr = (0.15, 0.30, 0.45)[(scene_id + subject) % 3]
            complexity = ((scene_id // 2) + subject) % 2
            for bin_index in range(6):
                time_norm = bin_index / 5
                value = (
                    1 + subject * 0.01 + time_norm * (1 + wwr)
                    + complexity * time_norm + 0.002 * subject * bin_index ** 2
                    + rng.normal(0, 0.05)
                )
                rows.append({
                    "participant_id": f"P{subject:02d}", "scene_id": scene_id,
                    "bin_index": bin_index, "time_norm": time_norm,
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

    outputs = run_timebin_models(timebin_path, qc_path, tmp_path / "models")
    models = pd.read_csv(outputs["timebin_model_results"])
    diagnostics = pd.read_csv(outputs["timebin_model_diagnostics"])
    summaries = pd.read_csv(outputs["temporal_scene_summaries"])
    assert not models.empty
    assert models["analysis_resolution"].eq("synchronized_timebin").all()
    assert models["analysis_status"].eq("primary").all()
    assert models["model_type"].str.contains("independent_robust").all()
    assert models["formula"].str.contains(r"C\(WWR\):time_norm", regex=True).any()
    assert models["formula"].str.contains(r"C\(Complexity\):time_norm", regex=True).any()
    assert models.loc[models["hypothesis_family"].str.startswith("H_time_"), "p_fdr_bh"].notna().all()
    assert diagnostics["model_type"].str.contains("participant_clustered_independent_robust").all()
    assert {"early_mean", "middle_mean", "late_mean", "late_minus_early", "scene_slope_per_time_norm"}.issubset(summaries.columns)
