from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_analysis.teacher.contracts import (
    REQUIRED_PARTICIPANT_COLUMNS,
    build_modality_registry,
    canonicalize_trials,
    normalize_participant_information,
)
from paper_analysis.teacher.eeg import (
    FORBIDDEN_PRIMARY_TERMS,
    _metric_columns,
    _onset_contract_errors,
    _scene_qc_mask,
)
from paper_analysis.teacher.eye import (
    SceneMasks,
    _eye_filename_participant_candidate,
    _write_authorized_self_review,
    build_eye_trial_metrics,
    classify_fixations,
    deduplicate_fixations,
    recompute_tracking_quality,
    scene_area_rows,
    six_core_outcomes,
)
from paper_analysis.teacher.state import (
    StageBlockedError,
    method_contract_hash,
    require_approval,
    write_approval_template,
    stage_method_contract_hash,
)
from paper_analysis.teacher.complete import (
    _all_files_index,
    _manifest_complete,
    _parallel_order_carryover_audit,
)
from paper_analysis.teacher.reporting import write_markdown_report


def test_all_files_index_excludes_its_own_output(tmp_path: Path) -> None:
    (tmp_path / "result.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "结果文件总索引.xlsx").write_bytes(b"stale index")

    index = _all_files_index(tmp_path)

    assert index["RelativePath"].tolist() == ["result.csv"]


@pytest.mark.parametrize(
    ("stage", "status", "required_file"),
    [
        ("eye-stage1", "review_required", "AOI_masks_approved.txt"),
        ("eye-stage3-plan", "approval_required", "stage3_plan.txt"),
    ],
)
def test_validated_gate_stage_can_be_reused(
    tmp_path: Path, stage: str, status: str, required_file: str
) -> None:
    stage_dir = tmp_path / stage
    stage_dir.mkdir()
    (stage_dir / required_file).write_text("approved", encoding="utf-8")
    (stage_dir / "input_hashes.json").write_text("[]", encoding="utf-8")
    (stage_dir / "run_manifest.json").write_text(json.dumps({
        "status": status,
        "stage_method_contract_hash": stage_method_contract_hash(
            Path(__file__).resolve().parents[1], stage
        ),
    }), encoding="utf-8")
    assert _manifest_complete(
        stage_dir / "run_manifest.json",
        Path(__file__).resolve().parents[1],
        stage,
    )


def test_parallel_order_carryover_audit_applies_joint_fdr(tmp_path: Path) -> None:
    for trim, p_value in zip((0, 5, 10, 15), (0.20, 0.04, 0.01, 0.30)):
        folder = tmp_path / "onset_window_models" / f"trim_{trim}s"
        folder.mkdir(parents=True)
        pd.DataFrame([
            {
                "outcome": "O_theta_relative",
                "model": "PreviousScene",
                "term": "PreviousWWRWWR75",
                "estimate": 0.01,
                "std.error": 0.01,
                "df": 30,
                "p.value": p_value,
            },
            {
                "outcome": "O_theta_relative",
                "model": "Model1",
                "term": "Block",
                "estimate": 0.01,
                "std.error": 0.01,
                "df": 30,
                "p.value": 0.001,
            },
        ]).to_csv(folder / "09_eeg_order_CR2.csv", index=False)
    audit = _parallel_order_carryover_audit(tmp_path)
    assert len(audit) == 4
    assert audit["detected_raw_0_05"].sum() == 2
    assert audit["detected_parallel_fdr_0_05"].sum() == 1
    assert audit["term"].eq("PreviousWWRWWR75").all()


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


def test_teacher_registry_uses_experience_group_not_exercise_frequency() -> None:
    normalized = normalize_participant_information(pd.DataFrame({
        "Participant": ["P01", "P02"],
        "ExperienceRaw": [
            "偶尔（每月1–2次）",
            "经常（每月≥5次）",
        ],
        "ExerciseFrequency": ["High", "Low"],
    }))
    assert normalized["ExperienceGroup"].tolist() == ["Low", "High"]
    assert "ExperienceGroup" in REQUIRED_PARTICIPANT_COLUMNS
    assert "ExerciseFrequency" not in REQUIRED_PARTICIPANT_COLUMNS


def test_eye_raw_trial_filename_yields_participant_candidate() -> None:
    path = Path("raw_P01_260130211058_0617142206.csv")
    assert _eye_filename_participant_candidate(path) == "P01"
    assert _eye_filename_participant_candidate(Path("unmatched.csv")) == "unmatched"


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


def test_fixation_dedup_empty_input_preserves_event_contract() -> None:
    fixations, conflicts = deduplicate_fixations(
        pd.DataFrame({"Fixation Index": [np.nan]}),
        participant="P01",
        global_trial_order=1,
    )
    assert fixations.empty
    assert {"FixationX", "FixationY", "FixationDuration"}.issubset(
        fixations.columns
    )
    assert conflicts.empty
    assert "Reason" in conflicts.columns


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


def test_authorized_self_review_refreshes_a_stale_resume_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "participant_retention_approved.txt"
    _write_authorized_self_review(
        path,
        fingerprint="old",
        stage="eye-stage2-retention",
        notes=["old run"],
    )
    _write_authorized_self_review(
        path,
        fingerprint="new",
        stage="eye-stage2-retention",
        notes=["current run"],
    )
    approval = require_approval(
        path,
        fingerprint="new",
        stage="eye-stage2-retention",
    )
    assert approval["approved_by"] == "user_authorized_codex_self_audit"
    assert approval["notes"] == ["current run"]


def test_method_contract_hash_covers_non_teacher_analysis_dependencies(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "src" / "paper_analysis" / "stats" / "timebin.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("POLICY = 'participant_complete'\n", encoding="utf-8")
    old_hash = method_contract_hash(tmp_path)
    dependency.write_text("POLICY = 'scene_level'\n", encoding="utf-8")
    assert method_contract_hash(tmp_path) != old_hash


def test_teacher_eeg_contract_forbids_three_way_and_condition_order_terms() -> None:
    assert "WWR:Complexity:ExperienceGroup" in FORBIDDEN_PRIMARY_TERMS
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


def test_eeg_structural_cohort_is_separated_from_scene_qc_model_cohort() -> None:
    frame = pd.DataFrame({
        "bad_eeg_quality": [False, True, False, False],
        "eeg_subject_quality_exclusion": [False, False, True, False],
    })
    assert _scene_qc_mask(frame).tolist() == [True, False, False, True]
    with pytest.raises(StageBlockedError, match="QC columns"):
        _scene_qc_mask(pd.DataFrame({"bad_eeg_quality": [False]}))


def test_teacher_eeg_requires_locked_formal_onset_contract_when_configured() -> None:
    config = {
        "onset_trim_strategy": "parallel",
        "reference_onset_trim_s": 10,
        "onset_trim_variants_s": [0, 5, 10, 15],
        "equivalence_bound_sd": 0.20,
        "onset_random_seed": 20260802,
    }
    valid = pd.DataFrame({
        "onset_trim_s": [10], "analysis_start_s": [10],
        "analysis_end_s": [60], "analysis_dur_s": [50],
        "onset_samples_removed": [5000], "trim_status": ["ok"],
    })
    assert _onset_contract_errors(valid, config) == []
    invalid = valid.drop(columns=["analysis_dur_s"])
    assert "onset-trim metadata" in _onset_contract_errors(invalid, config)[0]


def test_canonicalize_trials_coalesces_equivalent_legacy_aliases() -> None:
    frame = pd.DataFrame({
        "subject_id": ["P01", "P01"],
        "participant_id": ["P01", "P01"],
        "scene_id": [1, 2],
        "block_id": [1, 1],
        "block": [1.0, 1.0],
        "round": [1, 1],
        "cycle_in_block": [1, 2],
        "position": [1.0, 2.0],
        "WWR": ["WWR20", "WWR40"],
        "Complexity": ["C0", "C1"],
    })
    result = canonicalize_trials(frame)
    assert result["Participant"].tolist() == ["P01", "P01"]
    assert result["Block"].tolist() == [1, 1]
    assert result["PositionWithinBlock"].tolist() == [1, 2]
    assert not result.columns.duplicated().any()


def test_teacher_reports_are_markdown_and_preserve_machine_table_preview(
    tmp_path: Path,
) -> None:
    target = write_markdown_report(
        tmp_path / "stage_report.md",
        title="Stage report",
        paragraphs=["Evidence-first narrative."],
        tables=[("Results", pd.DataFrame({"estimate": [1.25], "p": [0.01]}))],
    )
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Stage report")
    assert "```csv" in text
    assert "1.25" in text
    assert not (tmp_path / "stage_report.docx").exists()


def test_locked_r_scripts_encode_teacher_models_without_ols_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    eye = (root / "analysis/r/eye_stage2_analysis.R").read_text(encoding="utf-8")
    eeg = (root / "analysis/r/eeg_primary_analysis.R").read_text(encoding="utf-8")
    common = (root / "analysis/r/common.R").read_text(encoding="utf-8")
    assert "ordbeta" in eye
    assert "c(.50, .60, .70)" in eye
    assert "WWR * Complexity + WWR * ExperienceGroup" in eeg
    assert "iterations <- as.integer" in eeg
    assert "IncludeEEGTrialValid" in eye
    assert "input[as.logical(input$IncludeEEGValid)" not in eye
    assert 'vcov = "CR2"' in common
    assert "lm(" not in common
    assert "WWR * Complexity * ExperienceGroup" not in eeg
    assert "09c_eeg_secondary_models.csv" in eeg
    assert "09d_eeg_supplemental_models.csv" in eeg
