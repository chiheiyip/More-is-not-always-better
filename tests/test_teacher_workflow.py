from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_analysis.teacher.contracts import (
    build_modality_registry,
    canonicalize_trials,
)
from paper_analysis.teacher.eeg import FORBIDDEN_PRIMARY_TERMS, _metric_columns
from paper_analysis.teacher.eye import (
    SceneMasks,
    build_eye_trial_metrics,
    classify_fixations,
    deduplicate_fixations,
    recompute_tracking_quality,
    scene_area_rows,
    six_core_outcomes,
)
from paper_analysis.teacher.state import (
    StageBlockedError,
    require_approval,
    write_approval_template,
)


def test_modality_registry_is_union_and_eye_does_not_require_eeg() -> None:
    base = pd.DataFrame({
        "Participant": ["P01"],
        "Gender": ["F"],
        "OrderGroup": ["order1"],
        "IncludeEEGValid": [True],
    })
    registry = build_modality_registry(
        base,
        eye_participants=["P02"],
        questionnaire_participants=["P03"],
    ).set_index("Participant")
    assert set(registry.index) == {"P01", "P02", "P03"}
    assert bool(registry.loc["P02", "IncludeEyeCandidate"])
    assert not bool(registry.loc["P02", "IncludeEEGValid"])


def test_explicit_modality_exclusion_is_not_overridden_by_file_presence() -> None:
    base = pd.DataFrame({
        "Participant": ["P01"],
        "IncludeEyeCandidate": [False],
        "EyeExclusionReason": ["manual identity mismatch"],
    })
    registry = build_modality_registry(
        base, eye_participants=["P01", "P02"]
    ).set_index("Participant")
    assert not bool(registry.loc["P01", "IncludeEyeCandidate"])
    assert bool(registry.loc["P02", "IncludeEyeCandidate"])


def test_trial_contract_has_three_order_groups_and_block_local_lags() -> None:
    rows = []
    for participant, order in (
        ("P01", "order1"),
        ("P02", "order2"),
        ("P03", "neworder2"),
    ):
        for trial in range(1, 13):
            rows.append({
                "Participant": participant,
                "OrderGroup": order,
                "Block": 1 if trial <= 6 else 2,
                "PositionWithinBlock": (trial - 1) % 6 + 1,
                "GlobalTrialOrder": trial,
                "SceneID": trial,
                "WWR": [15, 45, 75][(trial - 1) % 3],
                "Complexity": trial % 2,
                "CSVFile": f"{participant}_{trial}.csv",
            })
    trials = canonicalize_trials(pd.DataFrame(rows), require_complete=True)
    assert set(trials["OrderGroup"]) == {"order1", "order2", "new order2"}
    first = trials["PositionWithinBlock"].eq(1)
    assert trials.loc[first, "PreviousWWR"].isna().all()
    assert trials.loc[first, "PreviousComplexity"].isna().all()


@pytest.mark.parametrize("valid_count,passes", [(599, False), (600, True)])
def test_tracking_threshold_boundary(valid_count: int, passes: bool) -> None:
    frame = pd.DataFrame({
        "Validity Left": [1] * valid_count + [0] * (1000 - valid_count),
        "Validity Right": [1] * valid_count + [0] * (1000 - valid_count),
        "Gaze Point X[px]": np.ones(1000),
        "Gaze Point Y[px]": np.ones(1000),
        "Tracking Ratio[%]": np.full(1000, 61.0),
    })
    _, quality = recompute_tracking_quality(frame)
    assert (quality["ValidTrackingRatio"] >= 0.6) is passes


def test_fixation_dedup_does_not_accumulate_repeated_duration() -> None:
    frame = pd.DataFrame({
        "Fixation Index": [1, 1, 1, 2, 2],
        "Fixation Point X[px]": [10.0, 10.4, 10.2, 2.0, 9.0],
        "Fixation Point Y[px]": [5.0, 5.2, 5.1, 3.0, 3.0],
        "Fixation Duration[ms]": [120, 120, 120, 80, 80],
    })
    fixations, conflicts = deduplicate_fixations(
        frame, participant="P01", global_trial_order=1,
        coordinate_tolerance_px=1.0,
    )
    assert fixations.loc[fixations["FixationIndex"].eq(1), "FixationDuration"].item() == 120
    assert conflicts["FixationIndex"].tolist() == [2]


def _scene(complexity: str = "C1") -> SceneMasks:
    valid = np.ones((4, 8), dtype=bool)
    table = np.zeros_like(valid)
    window = np.zeros_like(valid)
    equipment = np.zeros_like(valid)
    table[:, :2] = True
    window[:, 2:4] = True
    if complexity == "C1":
        equipment[:, 4:5] = True
    return SceneMasks(
        image_id="S1", width=8, height=4,
        masks={"Table": table, "Window": window, "Equipment": equipment},
        valid_scene=valid, valid_scene_reliable=True,
        projection_type="equirectangular", source_json=Path("S1.json"),
        base_image=None, overlap_pixels=0,
    )


def test_aoi_partition_conservation_c0_structural_na_and_core_outcomes() -> None:
    fixations = pd.DataFrame({
        "FixationX": [0.5, 2.5, 6.5, 20.0],
        "FixationY": [1.5, 1.5, 1.5, 1.5],
        "FixationDuration": [100.0, 100.0, 200.0, 50.0],
    })
    events = classify_fixations(fixations, _scene("C0"), complexity="C0")
    assert events["AOICategory"].tolist() == [
        "Table", "Window", "Background", "OffStimulus"
    ]
    area = {row["AOICategory"]: row["AOIAreaShare"] for row in scene_area_rows(_scene("C0"))}
    metrics, conservation = build_eye_trial_metrics(
        events, area_shares=area, complexity="C0"
    )
    assert conservation["ConservationPass"]
    equipment = metrics.loc[metrics["AOICategory"].eq("Equipment")].iloc[0]
    assert bool(equipment["StructuralNA"])
    outcomes = six_core_outcomes(metrics)
    assert set(outcomes) == {
        "TableShare", "WindowShare", "RawCompetition",
        "LogTableEnrichment", "LogWindowEnrichment", "AdjustedCompetition",
    }
    assert outcomes["TableShare"] == pytest.approx(0.25)
    assert outcomes["WindowShare"] == pytest.approx(0.25)


def test_equirectangular_area_has_pixel_and_cos_latitude_outputs() -> None:
    rows = pd.DataFrame(scene_area_rows(_scene("C1")))
    assert rows["PixelArea"].gt(0).any()
    assert rows["SphericalWeightedArea"].notna().all()
    assert rows["AreaBasis"].eq("cos_latitude_spherical").all()


def test_aoi_overlap_blocks_and_zero_share_has_no_pseudoconstant() -> None:
    scene = _scene("C1")
    overlapping = dict(scene.masks)
    overlapping["Window"] = overlapping["Table"].copy()
    bad = SceneMasks(
        **{
            **scene.__dict__,
            "masks": overlapping,
            "overlap_pixels": int(overlapping["Table"].sum()),
        }
    )
    fixations = pd.DataFrame({
        "FixationX": [0.5],
        "FixationY": [0.5],
        "FixationDuration": [100.0],
    })
    with pytest.raises(StageBlockedError, match="overlap"):
        classify_fixations(fixations, bad, complexity="C1")

    events = classify_fixations(fixations, scene, complexity="C1")
    area = {
        row["AOICategory"]: row["AOIAreaShare"]
        for row in scene_area_rows(scene)
    }
    metrics, _ = build_eye_trial_metrics(
        events, area_shares=area, complexity="C1"
    )
    outcomes = six_core_outcomes(metrics)
    assert outcomes["WindowShare"] == 0
    assert pd.isna(outcomes["LogWindowEnrichment"])


def test_approval_missing_unapproved_and_stale_are_blocked(tmp_path: Path) -> None:
    path = tmp_path / "approval.txt"
    with pytest.raises(StageBlockedError):
        require_approval(path, fingerprint="new", stage="eye-stage1")
    write_approval_template(path, fingerprint="old", stage="eye-stage1")
    with pytest.raises(StageBlockedError):
        require_approval(path, fingerprint="old", stage="eye-stage1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["approved"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageBlockedError, match="stale"):
        require_approval(path, fingerprint="new", stage="eye-stage1")


def test_teacher_eeg_contract_forbids_three_way_and_condition_order_terms() -> None:
    assert "WWR:Complexity:ExerciseFrequency" in FORBIDDEN_PRIMARY_TERMS
    assert "WWR:OrderGroup" in FORBIDDEN_PRIMARY_TERMS


def test_eeg_metric_contract_separates_relative_and_absolute_power() -> None:
    frame = pd.DataFrame({
        "O_theta": [10.0],
        "O_theta_absolute": [10.0],
        "O_theta_relative": [0.2],
    })
    relative, absolute = _metric_columns(frame, ["O_theta"])
    assert relative == ["O_theta_relative"]
    assert absolute == ["O_theta_absolute"]


def test_locked_r_scripts_encode_teacher_models_without_ols_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    eye = (root / "analysis/r/eye_stage2_analysis.R").read_text(encoding="utf-8")
    eeg = (root / "analysis/r/eeg_primary_analysis.R").read_text(encoding="utf-8")
    common = (root / "analysis/r/common.R").read_text(encoding="utf-8")
    assert "ordbeta" in eye
    assert "c(.50, .60, .70)" in eye
    assert "WWR * Complexity + WWR * ExerciseFrequency" in eeg
    assert "iterations <- as.integer" in eeg
    assert 'vcov = "CR2"' in common
    assert "lm(" not in common
    assert "WWR * Complexity * ExerciseFrequency" not in eeg
