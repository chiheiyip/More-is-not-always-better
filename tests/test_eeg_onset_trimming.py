from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from paper_analysis.eeg.contract import validate_eeg_scene_summary_frame
from paper_analysis.eeg.onset import build_common_qc_table
from paper_analysis.eeg.pipeline import apply_eeg_quality_qc, run_eeg_pipeline
from paper_analysis.fusion.clock_sync import synchronized_timebins
from paper_analysis.stats import onset_sensitivity as onset_stats


def _onset_rows() -> pd.DataFrame:
    rows = []
    for participant_index in range(4):
        for scene_id in (1, 2):
            for trim in (0.0, 5.0, 10.0, 15.0):
                view_start = 100.0
                view_end = 160.0
                rows.append({
                    "participant_id": f"P{participant_index + 1:02d}",
                    "scene_id": scene_id,
                    "view_start_s": view_start,
                    "view_end_s": view_end,
                    "view_dur_s": 60.0,
                    "onset_trim_s": trim,
                    "analysis_start_s": view_start + trim,
                    "analysis_end_s": view_end,
                    "analysis_dur_s": 60.0 - trim,
                    "onset_samples_removed": int(trim * 500),
                    "trim_status": "ok",
                    "segment_valid_duration": True,
                    "O_theta": 1.0 + participant_index + scene_id / 10 + trim / 100,
                    "F_theta": 2.0 + participant_index + trim / 50,
                    "O_alpha": 3.0 + participant_index + trim / 40,
                    "hf_ratio_20_40Hz": 0.05 + participant_index / 100 + trim / 1000,
                    "rms_mean_uV": 5.0 + participant_index + trim / 10,
                    "peak_to_peak_uV": 20.0 + participant_index + trim,
                    "nan_fraction": 0.0,
                    "flat_fraction": 0.0,
                })
    return pd.DataFrame(rows)


def test_primary_and_long_onset_tables_have_locked_contract_and_independent_qc(tmp_path: Path) -> None:
    sensitivity = _onset_rows()
    primary = sensitivity.loc[sensitivity["onset_trim_s"].eq(10)].copy()
    export = tmp_path / "summary"
    export.mkdir()
    primary_path = export / "all_subjects_scene_level.csv"
    sensitivity_path = export / "all_subjects_scene_level_onset_sensitivity.csv"
    primary.to_csv(primary_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    participants = tmp_path / "participants.csv"
    scene = tmp_path / "scene.csv"
    pd.DataFrame({
        "participant_id": [f"P{i:02d}" for i in range(1, 5)],
    }).to_csv(participants, index=False)
    pd.DataFrame([
        {
            "participant_id": f"P{i:02d}", "scene_id": scene_id,
            "WWR": 15 if scene_id == 1 else 45,
            "Complexity": scene_id - 1, "block": 1, "position": scene_id,
        }
        for i in range(1, 5) for scene_id in (1, 2)
    ]).to_csv(scene, index=False)
    outputs = run_eeg_pipeline(
        participants, scene, primary_path, tmp_path / "eeg",
        eeg_qc_config={
            "policy": "robust", "robust_min_n": 4, "robust_k": 3.5,
            "bad_scene_fraction_threshold": 1.0,
        },
        eeg_analysis_config={
            "primary_onset_trim_s": 10,
            "onset_trim_variants_s": [0, 5, 10, 15],
            "equivalence_bound_sd": 0.2,
            "bootstrap_iterations": 10,
            "random_seed": 1,
        },
        require_onset_metadata=True,
    )
    trial = pd.read_csv(outputs["eeg_trial_long"])
    long = pd.read_csv(outputs["eeg_onset_sensitivity_trial_long"])
    thresholds = pd.read_csv(outputs["eeg_onset_sensitivity_qc_thresholds"])
    assert trial[["participant_id", "scene_id"]].duplicated().sum() == 0
    assert long[["participant_id", "scene_id", "onset_trim_s"]].duplicated().sum() == 0
    assert sorted(thresholds["onset_trim_s"].unique().tolist()) == [0, 5, 10, 15]
    expected = long.loc[long["onset_trim_s"].eq(10), trial.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(trial.reset_index(drop=True), expected, check_dtype=False)
    common = pd.read_csv(outputs["eeg_onset_common_qc"])
    assert len(common) == 8
    assert {"qc_pass_trim_5", "qc_pass_trim_10", "qc_pass_trim_15", "onset_common_qc_pass"}.issubset(common)


def test_analysis_duration_takes_priority_over_full_view_duration_for_hard_qc() -> None:
    frame = pd.DataFrame({
        "participant_id": ["P01"], "scene_id": [1],
        "view_dur_s": [60.0], "analysis_dur_s": [0.5],
        "segment_valid_duration": [True], "O_theta": [1.0],
        "hf_ratio_20_40Hz": [0.1],
    })
    trial, *_ = apply_eeg_quality_qc(frame, {
        "policy": "robust", "min_segment_duration_s": 1.0,
        "robust_min_n": 4, "robust_metrics": ["hf_ratio_20_40Hz"],
        "bad_scene_fraction_threshold": 1.0,
    })
    assert bool(trial.loc[0, "bad_eeg_quality"])
    assert "segment_duration" in trial.loc[0, "eeg_qc_reasons"]


def test_legacy_csv_is_compatible_but_formal_contract_requires_trim_metadata() -> None:
    legacy = pd.DataFrame({
        "participant_id": ["P01"], "scene_id": [1],
        "view_dur_s": [60.0], "O_theta": [1.0],
    })
    compatible = validate_eeg_scene_summary_frame(legacy)
    assert compatible["status"] in {"pass", "warning"}
    assert compatible["compatibility_mode"]
    formal = validate_eeg_scene_summary_frame(
        legacy, expected_onset_trim_s=10, require_onset_metadata=True
    )
    assert formal["status"] == "error"
    assert any("onset-trim metadata" in error for error in formal["errors"])


def test_synchronized_bins_reanchor_both_modalities_at_exact_trim() -> None:
    srate = 500
    seconds = 20
    epoch = 1_800_000_000_000 + np.arange(seconds * srate) * 2
    eeg = pd.DataFrame({
        "eeg_epoch_ms": epoch,
        "preproc_F3_uV": np.sin(2 * np.pi * 6 * np.arange(len(epoch)) / srate),
        "preproc_F4_uV": np.sin(2 * np.pi * 6 * np.arange(len(epoch)) / srate),
    })
    eye = pd.DataFrame({
        "eye_epoch_ms": epoch[::2],
        "Recording Time Stamp[ms]": np.arange(0, seconds * 1000, 4),
        "Gaze Point X[px]": 100.0,
        "Gaze Point Y[px]": 100.0,
    })
    trial = pd.Series({"participant_id": "P01", "scene_id": 1, "aoi_json_path": ""})
    bins10 = pd.DataFrame(synchronized_timebins(eye, eeg, trial, onset_trim_s=10))
    bins15 = pd.DataFrame(synchronized_timebins(eye, eeg, trial, onset_trim_s=15))
    assert bins10["bin_index"].drop_duplicates().tolist() == [0, 1, 2, 3, 4]
    assert bins15["bin_index"].drop_duplicates().tolist() == [0, 1]
    assert bins10["bin_start_epoch_ms"].min() == epoch[0] + 10_000
    assert bins15["bin_start_epoch_ms"].min() == epoch[0] + 15_000
    assert bins10["scene_elapsed_s"].min() == 10
    assert bins10["analysis_elapsed_s"].min() == 0
    assert bins10["onset_trim_s"].eq(10).all()


def _equivalence_frame(slope_change: float) -> pd.DataFrame:
    rows = []
    for participant in range(10):
        participant_offset = participant / 10
        for scene_id, x in enumerate((-1.5, -0.5, 0.5, 1.5), start=1):
            for trim in (10.0, 15.0):
                rows.append({
                    "participant_id": f"P{participant:02d}", "scene_id": scene_id,
                    "onset_trim_s": trim, "x": x,
                    "eeg_F_theta": participant_offset + x * (1 + (slope_change if trim == 15 else 0)),
                })
    return pd.DataFrame(rows)


def test_paired_bootstrap_equivalence_equivalent_non_equivalent_and_failure() -> None:
    equivalent = onset_stats.paired_cluster_bootstrap_equivalence(
        _equivalence_frame(0.02), outcomes=["eeg_F_theta"],
        iterations=80, seed=4, formula_terms=["x"],
    )
    assert equivalent.loc[equivalent["term"].eq("x"), "equivalent"].item()

    non_equivalent = onset_stats.paired_cluster_bootstrap_equivalence(
        _equivalence_frame(1.0), outcomes=["eeg_F_theta"],
        iterations=40, seed=4, formula_terms=["x"],
    )
    assert not non_equivalent.loc[non_equivalent["term"].eq("x"), "equivalent"].item()

    original = onset_stats._fit_coefficients
    calls = {"count": 0}

    def fail_bootstrap(data: pd.DataFrame, formula: str) -> pd.Series:
        calls["count"] += 1
        if calls["count"] > 2:
            raise RuntimeError("synthetic bootstrap failure")
        return original(data, formula)

    with patch.object(onset_stats, "_fit_coefficients", side_effect=fail_bootstrap):
        failed = onset_stats.paired_cluster_bootstrap_equivalence(
            _equivalence_frame(0.02), outcomes=["eeg_F_theta"],
            iterations=10, seed=4, formula_terms=["x"],
        )
    assert failed.loc[failed["term"].eq("x"), "status"].item() == "insufficient_bootstrap_success"


def test_common_qc_excludes_any_window_or_subject_failure() -> None:
    frame = _onset_rows().loc[lambda value: value["onset_trim_s"].isin([5, 10, 15])].copy()
    frame["bad_eeg_quality"] = False
    frame["eeg_subject_quality_exclusion"] = False
    frame.loc[
        frame["participant_id"].eq("P01") & frame["scene_id"].eq(1)
        & frame["onset_trim_s"].eq(15), "bad_eeg_quality"
    ] = True
    frame.loc[frame["participant_id"].eq("P02") & frame["onset_trim_s"].eq(10), "eeg_subject_quality_exclusion"] = True
    common = build_common_qc_table(frame)
    assert not common.loc[common["participant_id"].eq("P01") & common["scene_id"].eq(1), "onset_common_qc_pass"].item()
    assert not common.loc[common["participant_id"].eq("P02"), "onset_common_qc_pass"].any()
