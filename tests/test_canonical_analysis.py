from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.stats.canonical import run_canonical_analysis


def test_canonical_models_use_modality_specific_samples_clustered_gee_and_fdr(tmp_path: Path) -> None:
    participants = pd.DataFrame({
        "participant_id": [f"P{i:02d}" for i in range(1, 9)],
        "ExperienceGroup": ["Low", "High"] * 4,
        "Gender": ["Female", "Male"] * 4,
        "Age": np.arange(21, 29),
        "DateBatch": ["Initial"] * 4 + ["Supplement"] * 4,
    })
    q_rows, dynamic_rows, aoi_rows, eeg_rows, eye_qc_rows, eeg_qc_rows = [], [], [], [], [], []
    rng = np.random.default_rng(7)
    for subject_index, participant in enumerate(participants["participant_id"]):
        for scene_id in range(1, 9):
            wwr = [15, 45, 75, 45][(scene_id - 1) % 4]
            complexity = scene_id % 2
            base = {
                "participant_id": participant, "scene_id": scene_id, "WWR": wwr,
                "Complexity": complexity, "block": 1 + (scene_id > 4),
                "position": (scene_id - 1) % 4 + 1,
            }
            q_rows.append({**base, **{f"q_S{i}": 4 + .2 * complexity + .01 * wwr + rng.normal(0, .2) for i in range(1, 6)}})
            duration = 20 + rng.uniform(0, 2)
            dynamic_rows.append({
                **base, "valid_eye_duration_s": duration,
                "aoi_transition_count": 8 + complexity * 2 + subject_index % 3,
                "aoi_transition_rate_per_min": 24 + complexity * 5 + rng.uniform(.1, 1),
                "transition_entropy_normalized": .4 + .1 * complexity + rng.normal(0, .03),
                "angular_scanpath_deg_per_s": 5 + complexity + rng.uniform(.1, .5),
                "saccade_count": 10 + complexity * 2, "saccade_rate_per_min": 30 + complexity * 3,
                "saccade_amplitude_median_px": 50 + wwr / 5,
                "saccade_velocity_average_median_px_per_ms": 1 + complexity * .1,
                "saccade_velocity_peak_median_px_per_ms": 2 + complexity * .2,
                "window_entry_count": 2 + int(wwr >= 45),
                "window_directed_transition_rate_per_min": 5 + wwr / 20,
                "first_window_fixation_latency_ms": 1000 - wwr * 2 + rng.uniform(1, 30),
                "blink_count": 2 + int(scene_id > 4), "blink_rate_per_min": 6 + int(scene_id > 4),
                "pupil_post_early_delta_mm": .05 * complexity + rng.normal(0, .01),
            })
            for class_name in ["table", "window"]:
                visited = not (class_name == "window" and scene_id == 1 and subject_index % 3 == 0)
                aoi_rows.append({
                    **base, "class_name": class_name, "visited": visited,
                    "FC": 4 + complexity + int(class_name == "window"),
                    "FCR": .2 + .02 * complexity + .01 * int(class_name == "window"),
                    "TFD_ms": 500 + 20 * complexity + 30 * int(class_name == "window"),
                    "TTFF_ms": 300 + 10 * complexity, "attention_share": .4 + .1 * int(class_name == "window"),
                    "median_revisit_latency_ms": 800 + 20 * complexity,
                })
            eeg_rows.append({**base, "eeg_O_theta": 1 + .1 * complexity + rng.uniform(.01, .1)})
            eye_qc_rows.append({**base, "analysis_valid_ratio": .55 + .05 * ((scene_id + subject_index) % 8)})
            eeg_qc_rows.append({**base, "bad_eeg_quality": subject_index >= 6})

    paths = {}
    for name, frame in {
        "participants": participants, "questionnaire": pd.DataFrame(q_rows),
        "eye_dynamic": pd.DataFrame(dynamic_rows), "eye_aoi": pd.DataFrame(aoi_rows),
        "eeg": pd.DataFrame(eeg_rows), "eye_qc": pd.DataFrame(eye_qc_rows),
        "eeg_qc": pd.DataFrame(eeg_qc_rows),
    }.items():
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)

    out = run_canonical_analysis(
        questionnaire_csv=paths["questionnaire"], eye_aoi_csv=paths["eye_aoi"],
        eye_dynamic_csv=paths["eye_dynamic"], eye_qc_csv=paths["eye_qc"],
        eeg_csv=paths["eeg"], eeg_scene_qc_csv=paths["eeg_qc"],
        participants_csv=paths["participants"], outdir=tmp_path / "models",
        mde_simulations=20,
    )
    models = pd.read_csv(out["model_results"])
    diagnostics = pd.read_csv(out["model_diagnostics"])
    flow = pd.read_csv(out["modality_sample_flow"])
    eye_flow = flow.loc[(flow["modality"] == "eye") & (flow["eligibility_policy"] == "eye_metric_specific_available_no_eeg_filter")].iloc[0]
    eeg_flow = flow.loc[flow["modality"] == "eeg"].iloc[0]

    assert eye_flow["n_subjects"] == 8
    assert eeg_flow["n_subjects"] == 6
    assert models["model_type"].astype(str).str.startswith("gee_").all()
    assert not models["model_type"].astype(str).str.contains("ols", case=False).any()
    assert diagnostics["status"].astype(str).str.contains("ols", case=False).sum() == 0
    assert models["scope"].astype(str).str.startswith("eye_valid_coordinates_ge_").any()
    assert models["hypothesis_block"].isin(["H1_complexity_exploration", "H2_wwr_window_direction", "H3_experience_moderation", "H4_fatigue_order"]).any()
    assert {"grain", "family", "scope", "n_subjects", "n_trials", "formula", "p_fdr_bh"}.issubset(models.columns)
    pupil = models.loc[models["outcome"].eq("pupil_post_early_delta_mm")]
    assert pupil["interpretation_tier"].eq("exploratory").all()
    assert not (tmp_path / "models" / "legacy").exists()
