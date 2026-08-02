#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from more_is_not_always_better.discovery import load_questionnaire_metadata
from paper_analysis.teacher.eye import _eye_filename_participant_candidate
from paper_analysis.utils.coding import experience_group


def _order_group(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    if text.startswith("neworder2") or text.startswith("new order2"):
        return "new order2"
    if text.startswith("order2"):
        return "order2"
    if text.startswith("order1"):
        return "order1"
    return ""


def _raw_record_date(paths: list[Path], participant: str) -> datetime | None:
    for path in paths:
        if _eye_filename_participant_candidate(path) != participant:
            continue
        match = re.match(
            r"^raw_.+?_(\d{12})_\d+$", path.stem
        )
        if match:
            try:
                return datetime.strptime(match.group(1)[:6], "%y%m%d")
            except ValueError:
                pass
    return None


def _derived_order_group(
    participant: str,
    questionnaire_base: pd.DataFrame,
    raw_files: list[Path],
) -> str:
    order = (
        questionnaire_base.loc[participant].get("Order")
        if participant in questionnaire_base.index
        else pd.NA
    )
    if pd.to_numeric(order, errors="coerce") == 1:
        return "order1"
    if pd.to_numeric(order, errors="coerce") == 2:
        date = _raw_record_date(raw_files, participant)
        return (
            "new order2"
            if date is not None and date >= datetime(2026, 5, 1)
            else "order2"
        )
    return ""


def _image_dimensions(aoi_path: Path) -> tuple[int | None, int | None]:
    try:
        document = json.loads(aoi_path.read_text(encoding="utf-8"))
        image = document.get("image", {})
        return int(image["width"]), int(image["height"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None, None


def _write_valid_scene_mask(
    image_path: Path,
    output_path: Path,
) -> tuple[Path, float]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    near_black = image.max(axis=2) <= 12
    labels, _ = ndimage.label(
        near_black,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    boundary_labels = np.unique(np.concatenate([
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1],
    ]))
    boundary_labels = boundary_labels[boundary_labels != 0]
    off_stimulus = np.isin(labels, boundary_labels)
    valid_scene = ~off_stimulus
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((valid_scene * 255).astype(np.uint8), mode="L").save(
        output_path
    )
    return output_path, float(off_stimulus.mean())


def build_inputs(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(args.participants_csv)
    manifest = pd.read_csv(args.scene_manifest_csv)
    questionnaire = load_questionnaire_metadata(args.questionnaire_xlsx)
    eeg_qc = pd.read_csv(args.eeg_qc_csv)

    raw_files = sorted(args.eye_root.rglob("*.csv"))
    raw_eye_ids = {
        _eye_filename_participant_candidate(path)
        for path in raw_files
    }
    all_ids = sorted(
        set(participants["participant_id"].astype(str))
        | set(questionnaire["participant_id"].astype(str))
        | raw_eye_ids
        | set(eeg_qc["participant_id"].astype(str))
    )

    participant_base = (
        participants.drop_duplicates("participant_id")
        .set_index("participant_id")
    )
    questionnaire_base = (
        questionnaire.drop_duplicates("participant_id")
        .set_index("participant_id")
    )
    order_by_participant = (
        manifest.assign(
            OrderGroup=manifest["order_scheme"].map(_order_group)
        )
        .groupby("participant_id", sort=False)["OrderGroup"]
        .agg(lambda values: values.dropna().iloc[0] if not values.dropna().empty else "")
    )
    for participant_id in raw_eye_ids - set(order_by_participant.index.astype(str)):
        order_by_participant.loc[participant_id] = _derived_order_group(
            participant_id, questionnaire_base, raw_files
        )
    eeg_valid = (
        ~eeg_qc.groupby("participant_id")["eeg_subject_quality_exclusion"]
        .max()
        .astype(bool)
    )

    registry_rows: list[dict[str, object]] = []
    for participant_id in all_ids:
        qrow = (
            questionnaire_base.loc[participant_id]
            if participant_id in questionnaire_base.index
            else pd.Series(dtype=object)
        )
        prow = (
            participant_base.loc[participant_id]
            if participant_id in participant_base.index
            else pd.Series(dtype=object)
        )
        experience_raw = qrow.get(
            "Experience", prow.get("ExperienceRaw", "")
        )
        include_eye = participant_id in raw_eye_ids
        registry_rows.append({
            "Participant": participant_id,
            "Gender": qrow.get("Gender", prow.get("Gender", "")),
            "ExperienceRaw": experience_raw,
            "ExperienceGroup": experience_group(experience_raw),
            "OrderGroup": order_by_participant.get(participant_id, ""),
            "IncludeEyeCandidate": include_eye,
            "IncludeEEGValid": bool(eeg_valid.get(participant_id, False)),
            "IncludeQuestionnaireValid": participant_id in questionnaire_base.index,
            "EyeExclusionReason": "",
            "CandidateOnly": True,
        })
    registry = pd.DataFrame(registry_rows)

    trials = pd.DataFrame({
        "Participant": manifest["participant_id"].astype(str),
        "OrderGroup": manifest["order_scheme"].map(_order_group),
        "Block": pd.to_numeric(manifest["block"], errors="raise").astype(int),
        "PositionWithinBlock": pd.to_numeric(
            manifest["position"], errors="raise"
        ).astype(int),
        "GlobalTrialOrder": pd.to_numeric(
            manifest["scene_id"], errors="raise"
        ).astype(int),
        "SceneID": pd.to_numeric(
            manifest["scene_id"], errors="raise"
        ).astype(int),
        "SceneLabel": manifest["scene_name"].astype(str),
        "WWR": pd.to_numeric(manifest["WWR"], errors="raise").astype(int),
        "Complexity": pd.to_numeric(
            manifest["Complexity"], errors="raise"
        ).astype(int),
        "CSVFile": manifest["eye_csv_path"].astype(str),
        "AOIFileCandidate": manifest["aoi_json_path"].astype(str),
    })
    mapped_participants = set(trials["Participant"].astype(str))
    extra_trial_rows: list[dict[str, object]] = []
    for participant_id in sorted(raw_eye_ids - mapped_participants):
        order_group = str(order_by_participant.get(participant_id, ""))
        template = (
            trials.loc[trials["OrderGroup"].eq(order_group)]
            .sort_values("GlobalTrialOrder")
            .drop_duplicates("GlobalTrialOrder")
        )
        if len(template) != 12:
            continue
        for template_row in template.itertuples(index=False):
            folder = args.eye_root / (
                f"{int(template_row.Block)}-{template_row.SceneLabel}"
            )
            matches = sorted(folder.glob(f"raw_{participant_id}_*.csv"))
            if len(matches) != 1:
                continue
            extra_trial_rows.append({
                "Participant": participant_id,
                "OrderGroup": order_group,
                "Block": int(template_row.Block),
                "PositionWithinBlock": int(
                    template_row.PositionWithinBlock
                ),
                "GlobalTrialOrder": int(template_row.GlobalTrialOrder),
                "SceneID": int(template_row.SceneID),
                "SceneLabel": template_row.SceneLabel,
                "WWR": int(template_row.WWR),
                "Complexity": int(template_row.Complexity),
                "CSVFile": str(matches[0]),
                "AOIFileCandidate": str(
                    folder / f"{int(template_row.Block)}-"
                    f"{template_row.SceneLabel}.json"
                ),
            })
    if extra_trial_rows:
        trials = pd.concat(
            [trials, pd.DataFrame(extra_trial_rows)],
            ignore_index=True,
        )
    trials = trials.sort_values(["Participant", "GlobalTrialOrder"])
    group = trials.groupby(["Participant", "Block"], sort=False)
    trials["PreviousWWR"] = group["WWR"].shift(1)
    trials["PreviousComplexity"] = group["Complexity"].shift(1)

    mapping_rows: list[dict[str, object]] = []
    mapping_role = "formal" if args.provisional_formal else "candidate"
    coordinate_confirmed = bool(args.provisional_formal)
    valid_scene_cache: dict[Path, tuple[Path, float]] = {}
    mapping_source = trials.drop_duplicates(
        ["OrderGroup", "Block", "SceneID", "AOIFileCandidate"]
    )
    for row in mapping_source.itertuples(index=False):
        aoi_path = Path(row.AOIFileCandidate)
        width, height = _image_dimensions(aoi_path)
        image_path = aoi_path.with_suffix(".png")
        valid_scene_path = ""
        off_stimulus_fraction: float | str = ""
        if args.provisional_formal and image_path.exists():
            if image_path not in valid_scene_cache:
                valid_scene_cache[image_path] = _write_valid_scene_mask(
                    image_path,
                    output_dir / "valid_scene_masks"
                    / f"{aoi_path.stem}_ValidScene.png",
                )
            valid_scene_path, off_stimulus_fraction = valid_scene_cache[
                image_path
            ]
        mapping_rows.append({
            "AOIImageID": aoi_path.stem,
            "SceneID": row.SceneID,
            "WWR": row.WWR,
            "Complexity": row.Complexity,
            "OrderGroup": row.OrderGroup,
            "Block": row.Block,
            "BaseImageFile": str(image_path),
            "AOIFile": str(aoi_path),
            "SameImageAcrossBlocks": "",
            "ProjectionType": "unknown",
            "ImageWidth": width,
            "ImageHeight": height,
            "CoordinateSystemConfirmed": coordinate_confirmed,
            "ValidSceneConfirmed": coordinate_confirmed,
            "ValidSceneFile": str(valid_scene_path),
            "OffStimulusCanvasFraction": off_stimulus_fraction,
            "MappingRole": mapping_role,
            "OverlapResolution": (
                "table_window_equipment_priority"
                if args.provisional_formal else ""
            ),
            "AOIResolutionReason": (
                "User-authorized provisional filename mapping; manual "
                "scene/AOI/coordinate/ValidScene verification remains pending."
                if args.provisional_formal
                else
                "Filename-derived candidate only; teacher-required manual "
                "scene/AOI/coordinate confirmation is pending."
            ),
            "Notes": (
                "Provisional formal mapping authorized for an interim run; "
                "results must not be labelled final before human review."
                if args.provisional_formal
                else "Do not promote to formal without human review."
            ),
        })
    mapping = pd.DataFrame(mapping_rows)

    registry_path = output_dir / "participant_information_candidate.xlsx"
    trials_path = output_dir / "trial_order_mapping_candidate.xlsx"
    mapping_path = output_dir / "scene_AOI_mapping_candidate.xlsx"
    registry.to_excel(registry_path, index=False)
    trials.to_excel(trials_path, index=False)
    mapping.to_excel(mapping_path, index=False)
    preprocessing_parameters = {
        "input_state": "preprocessed EEGLAB .set",
        "upstream_filter_and_reference": (
            "not recoverable from exporter; reported as unavailable"
        ),
        "view_start_marker": "7",
        "view_end_marker": "8",
        "theta_hz": "4-7",
        "alpha_hz": "8-12",
        "beta_hz": "13-30",
        "relative_power_denominator_hz": "1-45",
        "frontal_roi": "F3,F4",
        "parietal_roi": "P3,PZ,P4",
        "occipital_roi": "O1,OZ,O2",
        "power_estimator": "Welch PSD integrated over band",
        "estimand": "sustained-state activity after scene entry",
        "primary_onset_trim_s": 10,
        "onset_trim_variants_s": "0,5,10,15",
        "onset_equivalence_bound_sd": 0.20,
        "onset_random_seed": 20260802,
    }
    preprocessing_audit_path = (
        output_dir / "confirmed_eeg_preprocessing_audit.xlsx"
    )
    pd.DataFrame([
        {
            "Parameter": key,
            "Value": value,
            "Evidence": (
                "matlab/eeg_bandpower_pipeline/run_eeg_bandpower_pipeline.m "
                "and generated methods snapshot"
            ),
        }
        for key, value in preprocessing_parameters.items()
    ]).to_excel(preprocessing_audit_path, index=False)

    config = {
        "outputs_root": str(args.outputs_root.resolve()),
        "run_id": None,
        "rscript": str(REPO_ROOT / "scripts" / "portable_rscript.cmd"),
        "questionnaire_file": str(args.questionnaire_xlsx.resolve()),
        "participant_information": str(registry_path),
        "trial_order_mapping": str(trials_path),
        "scene_aoi_mapping": str(mapping_path),
        "eye": {
            "raw_root": str(args.eye_root.resolve()),
            "aoi_root": str(args.eye_root.resolve()),
            "base_images_root": str(args.eye_root.resolve()),
            "point_source": "fixation",
            "validity_accepted": [1],
            "validity_rule": "both",
            "primary_tracking_threshold": 0.6,
            "sensitivity_tracking_thresholds": [0.5, 0.7],
            "fixation_coordinate_tolerance_px": 1.0,
            "projection_default": "unknown",
            "expected_eye_candidates": len(raw_eye_ids),
            "teacher_reported_eye_candidates": 56,
            "eye_candidate_count_resolution": (
                "Raw data contain 57 complete 12-trial eye records. The additional "
                "participant has questionnaire and eye data but no EEG .set/.fdt; "
                "the teacher rule prohibits excluding eye data solely for missing EEG."
                if args.provisional_formal else ""
            ),
            "expected_aoi_images": 12,
            "teacher_reported_aoi_images": 12,
            "aoi_count_resolution": (
                "The experiment has 12 real scenes and 12 corresponding AOI files. "
                "No scene or AOI is merged merely to reduce the file count."
            ),
            "provisional_user_authorization": bool(args.provisional_formal),
            "stage1_dir": "",
            "stage2_dir": "",
            "stage3_plan_dir": "",
        },
        "eeg": {
            "trial_file": str(args.eeg_trial_csv.resolve()),
            "onset_sensitivity_trial_file": str(
                args.eeg_trial_csv.resolve().with_name(
                    "eeg_onset_sensitivity_trial_long.csv"
                )
            ),
            "primary_onset_trim_s": 10,
            "onset_trim_variants_s": [0, 5, 10, 15],
            "equivalence_bound_sd": 0.20,
            "onset_random_seed": 20260802,
            "onset_analysis_dir": "",
            "preprocessing_audit_file": str(
                preprocessing_audit_path
            ),
            "preprocessing_confirmed": bool(args.provisional_formal),
            "preprocessing_parameters": preprocessing_parameters,
            "order_stage_dir": "",
            "core_metrics": ["O_theta", "F_theta", "O_alpha", "O_beta"],
            "secondary_metrics": ["P_theta", "P_alpha", "F_alpha"],
            "supplemental_metrics": ["F_beta", "P_beta"],
            "primary_measure": "relative_power",
            "crossmodal_metrics": ["O_theta", "O_alpha"],
            "bootstrap_iterations": 5000,
        },
        "stage3": {
            "zero_rate_trigger": 0.1,
            "s3_trial_file": str(
                (
                    args.outputs_root
                    / "02_questionnaire"
                    / "questionnaire_long.csv"
                ).resolve()
            ),
            "bootstrap_iterations": 5000,
            "boundary_material_share_change": 0.05,
            "run_boundary_sensitivity": True,
            "run_area_composition_sensitivity": True,
            "run_time_effects": True,
            "run_experience_moderation": True,
        },
    }
    config_path = output_dir / "teacher_analysis.candidate.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    note_path = output_dir / "CANDIDATE_INPUTS_REQUIRE_MANUAL_REVIEW.txt"
    note_path.write_text(
        "\n".join([
            "These files are candidate mappings, not approvals.",
            f"Union registry participants: {len(registry)}",
            f"Raw eye filename candidates: {len(raw_eye_ids)}",
            f"Formally mapped trial rows: {len(trials)}",
            f"Candidate scene/AOI rows: {len(mapping)}",
            (
                "Provisional formal mappings were requested by the user. Any Stage 2 "
                "result remains provisional until identity, scene/AOI, coordinate "
                "system, projection, ValidScene, and all 12 scene/AOI mappings are reviewed."
                if args.provisional_formal
                else
                "Do not create AOI_masks_approved.txt until identity, scene/AOI, "
                "coordinate system, projection, and ValidScene are manually confirmed."
            ),
            "",
        ]),
        encoding="utf-8",
    )
    return {
        "participant_information": registry_path,
        "trial_order_mapping": trials_path,
        "scene_aoi_mapping": mapping_path,
        "config": config_path,
        "review_note": note_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clearly labelled candidate inputs for teacher Stage 1."
    )
    parser.add_argument("--questionnaire-xlsx", required=True, type=Path)
    parser.add_argument("--participants-csv", required=True, type=Path)
    parser.add_argument("--scene-manifest-csv", required=True, type=Path)
    parser.add_argument("--eeg-qc-csv", required=True, type=Path)
    parser.add_argument("--eeg-trial-csv", required=True, type=Path)
    parser.add_argument("--eye-root", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--provisional-formal",
        action="store_true",
        help=(
            "Mark filename-derived mappings as provisionally formal after explicit "
            "user authorization; generated notes still require later human review."
        ),
    )
    args = parser.parse_args()
    for name, path in build_inputs(args).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
