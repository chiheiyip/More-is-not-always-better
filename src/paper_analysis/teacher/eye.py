from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from paper_analysis.eye_tracking.aoi import PolygonAOI, load_aoi_document, point_in_poly
from paper_analysis.teacher.contracts import (
    REQUIRED_TRIAL_COLUMNS,
    build_modality_registry,
    canonicalize_trials,
    normalize_complexity,
    trial_contract_audit,
)
from paper_analysis.teacher.state import (
    StageBlockedError,
    require_approval,
    stage_fingerprint,
    write_approval_template,
    write_input_hashes,
    write_run_manifest,
)
from paper_analysis.utils.io import is_truthy, read_table, write_table, write_text


THEORETICAL_AOIS = ("Table", "Window", "Equipment")
ALL_CATEGORIES = ("Table", "Window", "Equipment", "Background", "OffStimulus")
AOI_COLORS = {
    "Table": (220, 45, 45, 180),
    "Window": (238, 196, 40, 180),
    "Equipment": (46, 106, 220, 180),
    "ValidScene": (40, 170, 90, 120),
}


@dataclass(frozen=True)
class SceneMasks:
    image_id: str
    width: int
    height: int
    masks: dict[str, np.ndarray]
    valid_scene: np.ndarray
    valid_scene_reliable: bool
    projection_type: str
    source_json: Path
    base_image: Path | None
    overlap_pixels: int
    original_overlap_pixels: int = 0
    overlap_resolution: str = ""


def _eye_stage_input_paths(
    participant_path: Path,
    trial_path: Path,
    mapping_path: Path,
    raw_root: Path | None = None,
    aoi_root: Path | None = None,
) -> list[Path]:
    paths = [participant_path, trial_path, mapping_path]
    if trial_path.exists():
        trials = read_table(trial_path)
        for value in trials.get("CSVFile", pd.Series(dtype=object)).dropna():
            resolved = _resolve(value, trial_path.parent)
            if resolved is not None:
                paths.append(resolved)
    if mapping_path.exists():
        mapping = read_table(mapping_path)
        for column in ("AOIFile", "BaseImageFile", "ValidSceneFile"):
            for value in mapping.get(column, pd.Series(dtype=object)).dropna():
                resolved = _resolve(value, mapping_path.parent)
                if resolved is not None:
                    paths.append(resolved)
    if raw_root is not None and raw_root.exists():
        paths.extend(raw_root.rglob("*.csv"))
    if aoi_root is not None and aoi_root.exists():
        paths.extend(aoi_root.rglob("*.json"))
    return paths


def _resolve(value: object, base: Path) -> Path | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value).strip())
    return path if path.is_absolute() else (base / path).resolve()


def _eye_filename_participant_candidate(path: Path) -> str:
    """Return the participant-like token without treating every trial as a person."""
    match = re.match(r"^raw_(.+?)_(\d{12})_(\d+)$", path.stem)
    candidate = match.group(1).strip() if match else path.stem
    suffix = re.match(r"^(?P<name>.+?)-\d+-\d+$", candidate)
    return suffix.group("name").strip() if suffix else candidate


def _canonical_aoi_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "").replace(" ", "")
    return {
        "table": "Table",
        "window": "Window",
        "equipment": "Equipment",
        "validscene": "ValidScene",
        "validcanvas": "ValidScene",
    }.get(text, str(value or "").strip())


def _mask_from_polygons(
    polygons: list[PolygonAOI], width: int, height: int
) -> dict[str, np.ndarray]:
    images: dict[str, Image.Image] = {}
    for polygon in polygons:
        name = _canonical_aoi_name(polygon.class_name)
        image = images.setdefault(name, Image.new("1", (width, height), 0))
        ImageDraw.Draw(image).polygon(polygon.points, fill=1)
    return {
        name: np.asarray(image, dtype=bool)
        for name, image in images.items()
    }


def _read_binary_mask(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        raise ValueError(
            f"Mask size {image.size} does not match AOI canvas {(width, height)}: {path}"
        )
    return np.asarray(image, dtype=np.uint8) > 0


def load_scene_masks(mapping_row: pd.Series, mapping_base: Path) -> SceneMasks:
    aoi_path = _resolve(mapping_row.get("AOIFile"), mapping_base)
    if aoi_path is None or not aoi_path.exists():
        raise FileNotFoundError(f"AOIFile is missing: {aoi_path}")
    polygons, metadata = load_aoi_document(aoi_path)
    width = int(
        pd.to_numeric(mapping_row.get("ImageWidth"), errors="coerce")
        if pd.notna(pd.to_numeric(mapping_row.get("ImageWidth"), errors="coerce"))
        else metadata.get("aoi_image_width_px")
    )
    height = int(
        pd.to_numeric(mapping_row.get("ImageHeight"), errors="coerce")
        if pd.notna(pd.to_numeric(mapping_row.get("ImageHeight"), errors="coerce"))
        else metadata.get("aoi_image_height_px")
    )
    masks = _mask_from_polygons(polygons, width, height)
    valid_scene_file = _resolve(mapping_row.get("ValidSceneFile"), mapping_base)
    if valid_scene_file and valid_scene_file.exists():
        valid_scene = _read_binary_mask(valid_scene_file, width, height)
        valid_scene_reliable = bool(mapping_row.get("ValidSceneConfirmed", True))
    elif "ValidScene" in masks:
        valid_scene = masks.pop("ValidScene")
        valid_scene_reliable = bool(mapping_row.get("ValidSceneConfirmed", False))
    else:
        valid_scene = np.ones((height, width), dtype=bool)
        valid_scene_reliable = False
    theoretical = [
        masks.get(name, np.zeros_like(valid_scene))
        for name in THEORETICAL_AOIS
    ]
    original_overlap = np.sum(np.stack(theoretical, axis=0), axis=0) > 1
    overlap_resolution = str(
        mapping_row.get("OverlapResolution", "")
    ).strip().lower()
    if original_overlap.any() and overlap_resolution == "table_window_equipment_priority":
        occupied = np.zeros_like(valid_scene)
        for name in THEORETICAL_AOIS:
            masks[name] = masks.get(
                name, np.zeros_like(valid_scene)
            ) & ~occupied
            occupied |= masks[name]
    theoretical = [
        masks.get(name, np.zeros_like(valid_scene))
        for name in THEORETICAL_AOIS
    ]
    overlap = np.sum(np.stack(theoretical, axis=0), axis=0) > 1
    base_image = _resolve(mapping_row.get("BaseImageFile"), mapping_base)
    return SceneMasks(
        image_id=str(mapping_row.get("AOIImageID", aoi_path.stem)),
        width=width,
        height=height,
        masks={name: masks.get(name, np.zeros_like(valid_scene)) for name in THEORETICAL_AOIS},
        valid_scene=valid_scene,
        valid_scene_reliable=valid_scene_reliable,
        projection_type=str(mapping_row.get("ProjectionType", "unknown")).strip().lower(),
        source_json=aoi_path,
        base_image=base_image if base_image and base_image.exists() else None,
        overlap_pixels=int(overlap.sum()),
        original_overlap_pixels=int(original_overlap.sum()),
        overlap_resolution=overlap_resolution,
    )


def scene_area_rows(scene: SceneMasks) -> list[dict[str, Any]]:
    valid = scene.valid_scene
    if not valid.any():
        raise ValueError(f"ValidScene is empty for {scene.image_id}")
    theoretical_union = np.zeros_like(valid)
    for mask in scene.masks.values():
        theoretical_union |= mask
    background = valid & ~theoretical_union
    masks = {**scene.masks, "Background": background}
    y = np.arange(scene.height, dtype=float)
    latitude = (0.5 - (y + 0.5) / scene.height) * math.pi
    spherical_weight = np.cos(latitude)[:, None]
    weighted_valid = float(np.sum(spherical_weight * valid))
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        clipped = mask & valid
        pixel_area = int(clipped.sum())
        pixel_share = pixel_area / int(valid.sum())
        if scene.projection_type == "equirectangular":
            weighted_area = float(np.sum(spherical_weight * clipped))
            weighted_share = weighted_area / weighted_valid if weighted_valid else np.nan
            primary_share = weighted_share
            area_basis = "cos_latitude_spherical"
        else:
            weighted_area = np.nan
            weighted_share = np.nan
            primary_share = pixel_share
            area_basis = (
                "rectilinear_image_plane"
                if scene.projection_type == "rectilinear"
                else "projected_image_operational"
            )
        rows.append({
            "AOIImageID": scene.image_id,
            "AOICategory": name,
            "ProjectionType": scene.projection_type or "unknown",
            "PixelArea": pixel_area,
            "PixelAreaShare": pixel_share,
            "SphericalWeightedArea": weighted_area,
            "SphericalWeightedAreaShare": weighted_share,
            "AOIAreaShare": primary_share,
            "AreaBasis": area_basis,
            "ValidScenePixels": int(valid.sum()),
        })
    return rows


def _draw_scene_preview(scene: SceneMasks, output: Path, mode: str) -> None:
    if scene.base_image:
        canvas = Image.open(scene.base_image).convert("RGBA")
        if canvas.size != (scene.width, scene.height):
            canvas = canvas.resize((scene.width, scene.height))
    else:
        canvas = Image.new("RGBA", (scene.width, scene.height), (245, 245, 245, 255))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    if mode == "valid_scene":
        mask = Image.fromarray((scene.valid_scene * 120).astype(np.uint8), mode="L")
        color = Image.new("RGBA", canvas.size, AOI_COLORS["ValidScene"])
        overlay.alpha_composite(Image.composite(color, Image.new("RGBA", canvas.size), mask))
    elif mode == "overlap":
        stack = np.stack(list(scene.masks.values()), axis=0).sum(axis=0)
        mask = Image.fromarray(((stack > 1) * 220).astype(np.uint8), mode="L")
        color = Image.new("RGBA", canvas.size, (255, 0, 180, 210))
        overlay.alpha_composite(Image.composite(color, Image.new("RGBA", canvas.size), mask))
    else:
        for name, array in scene.masks.items():
            mask = Image.fromarray((array * 120).astype(np.uint8), mode="L")
            color = Image.new("RGBA", canvas.size, AOI_COLORS[name])
            overlay.alpha_composite(Image.composite(color, Image.new("RGBA", canvas.size), mask))
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output)


def _match_scene_mapping(
    trial: pd.Series,
    mapping: pd.DataFrame,
    *,
    include_candidate: bool = False,
) -> pd.Series:
    candidates = mapping.copy()
    if "MappingRole" in candidates:
        role = candidates["MappingRole"].fillna("formal").astype(str).str.strip().str.lower()
        allowed = {"formal", "candidate"} if include_candidate else {"formal"}
        candidates = candidates.loc[role.isin(allowed)].copy()
    for column in ("SceneID", "OrderGroup", "Block"):
        if column not in candidates or column not in trial.index or pd.isna(trial.get(column)):
            continue
        left = candidates[column].astype(str).str.strip()
        right = str(trial[column]).strip()
        if column == "OrderGroup":
            left = left.str.lower().str.replace("_", " ", regex=False)
            right = right.lower().replace("_", " ")
        narrowed = candidates.loc[left.eq(right)]
        if not narrowed.empty:
            candidates = narrowed
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one AOI mapping for Participant={trial.get('Participant')} "
            f"SceneID={trial.get('SceneID')}; found {len(candidates)}"
        )
    return candidates.iloc[0]


def _read_csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.astype(str).tolist()


def _eye_file_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "FileExists": False,
            "Readable": False,
            "RowCount": 0,
            "RecordingDurationSeconds": np.nan,
            "EffectiveSamplingRateHz": np.nan,
            "FixationColumnsAvailable": False,
        }
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        return {
            "FileExists": True,
            "Readable": False,
            "ReadError": f"{type(exc).__name__}: {exc}",
            "RowCount": 0,
        }
    timestamp = pd.to_numeric(frame.get("Recording Time Stamp[ms]"), errors="coerce")
    duration_ms = float(timestamp.max() - timestamp.min()) if timestamp.notna().sum() > 1 else np.nan
    sampling = (
        1000.0 / float(timestamp.sort_values().diff().dropna().median())
        if timestamp.notna().sum() > 2 and timestamp.sort_values().diff().dropna().median() > 0
        else np.nan
    )
    return {
        "FileExists": True,
        "Readable": True,
        "RowCount": int(len(frame)),
        "RecordingDurationSeconds": duration_ms / 1000.0 if pd.notna(duration_ms) else np.nan,
        "EffectiveSamplingRateHz": sampling,
        "FixationColumnsAvailable": {
            "Fixation Index",
            "Fixation Duration[ms]",
            "Fixation Point X[px]",
            "Fixation Point Y[px]",
        }.issubset(frame.columns),
        "SourceMinX": pd.to_numeric(frame.get("Fixation Point X[px]"), errors="coerce").min(),
        "SourceMaxX": pd.to_numeric(frame.get("Fixation Point X[px]"), errors="coerce").max(),
        "SourceMinY": pd.to_numeric(frame.get("Fixation Point Y[px]"), errors="coerce").min(),
        "SourceMaxY": pd.to_numeric(frame.get("Fixation Point Y[px]"), errors="coerce").max(),
    }


def run_eye_stage1(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    participant_path = Path(config["participant_information"])
    trial_path = Path(config["trial_order_mapping"])
    mapping_path = Path(config["scene_aoi_mapping"])
    raw_root_value = str(config.get("eye", {}).get("raw_root", "")).strip()
    raw_root = Path(raw_root_value) if raw_root_value else None
    aoi_root_value = str(config.get("eye", {}).get("aoi_root", "")).strip()
    aoi_root = Path(aoi_root_value) if aoi_root_value else None
    stage1_inputs = _eye_stage_input_paths(
        participant_path, trial_path, mapping_path, raw_root, aoi_root
    )
    fingerprint = stage_fingerprint(
        "eye-stage1", config.get("eye", {}),
        stage1_inputs,
    )
    trials = canonicalize_trials(read_table(trial_path), require_complete=True)
    raw_files = (
        sorted(raw_root.rglob("*.csv"))
        if raw_root is not None and raw_root.exists()
        else []
    )
    raw_candidates = {
        _eye_filename_participant_candidate(path)
        for path in raw_files
    }
    raw_aoi_files = (
        sorted(aoi_root.rglob("*.json"))
        if aoi_root is not None and aoi_root.exists()
        else []
    )
    participants = build_modality_registry(
        participant_path,
        eye_participants=set(trials["Participant"].astype(str)) | raw_candidates,
    )
    registry_by_participant = participants.set_index("Participant")
    mapping = read_table(mapping_path)
    mapping_base = mapping_path.parent
    mapped_files: set[str] = set()
    for value in trials["CSVFile"].dropna():
        resolved = _resolve(value, trial_path.parent)
        if resolved is not None:
            mapped_files.add(str(resolved.resolve()))
    raw_inventory = pd.DataFrame([
        {
            "RawFile": str(path),
            "FilenameCandidateOnly": _eye_filename_participant_candidate(path),
            "InFormalTrialMapping": str(path.resolve()) in mapped_files,
            "ExplicitlyExcluded": (
                _eye_filename_participant_candidate(path)
                in registry_by_participant.index
                and not is_truthy(
                    registry_by_participant.loc[
                        _eye_filename_participant_candidate(path),
                        "IncludeEyeCandidate",
                    ]
                )
                and bool(str(
                    registry_by_participant.loc[
                        _eye_filename_participant_candidate(path)
                    ].get(
                        "EyeExclusionReason", ""
                    )
                ).strip())
            ),
        }
        for path in raw_files
    ])
    if not raw_inventory.empty:
        raw_inventory["CoveredByMappingOrExplicitExclusion"] = (
            raw_inventory["InFormalTrialMapping"]
            | raw_inventory["ExplicitlyExcluded"]
        )
    aoi_resolution: dict[str, dict[str, Any]] = {}
    if "AOIFile" in mapping:
        for _, row in mapping.iterrows():
            resolved = _resolve(row.get("AOIFile"), mapping_base)
            if resolved is None:
                continue
            aoi_resolution[str(resolved.resolve())] = {
                "MappingRole": str(row.get("MappingRole", "formal")).strip().lower(),
                "ResolutionReason": str(row.get("AOIResolutionReason", "")).strip(),
            }
    raw_aoi_inventory = pd.DataFrame([
        {
            "AOIJSONFile": str(path),
            "MappingRole": aoi_resolution.get(
                str(path.resolve()), {}
            ).get("MappingRole", ""),
            "ResolutionReason": aoi_resolution.get(
                str(path.resolve()), {}
            ).get("ResolutionReason", ""),
        }
        for path in raw_aoi_files
    ])
    if not raw_aoi_inventory.empty:
        raw_aoi_inventory["Resolved"] = (
            raw_aoi_inventory["MappingRole"].eq("formal")
            | (
                raw_aoi_inventory["MappingRole"].isin(
                    ["alias", "excluded", "historical"]
                )
                & raw_aoi_inventory["ResolutionReason"].str.len().gt(0)
            )
        )
    required_mapping = {
        "AOIImageID", "SceneID", "WWR", "Complexity", "BaseImageFile",
        "AOIFile", "ProjectionType", "ImageWidth", "ImageHeight",
    }
    missing_mapping = sorted(required_mapping - set(mapping.columns))
    audit = trial_contract_audit(trials)
    inventory_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    aoi_quality_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    scenes: dict[str, SceneMasks] = {}
    blockers: list[str] = []
    expected_aoi_images = int(
        config.get("eye", {}).get("expected_aoi_images", 9)
    )
    teacher_reported_aoi_images = int(
        config.get("eye", {}).get("teacher_reported_aoi_images", 9)
    )
    expected_eye_candidates = int(
        config.get("eye", {}).get("expected_eye_candidates", 56)
    )
    teacher_reported_eye_candidates = int(
        config.get("eye", {}).get("teacher_reported_eye_candidates", 56)
    )
    if missing_mapping:
        blockers.append(f"scene_AOI_mapping missing columns: {missing_mapping}")
    if raw_root is None or not raw_root.exists():
        blockers.append("configured eye.raw_root is missing")
    elif (
        len(raw_candidates) != expected_eye_candidates
        and not (
            int(participants["IncludeEyeCandidate"].map(is_truthy).sum())
            == expected_eye_candidates
            and all(
                candidate in registry_by_participant.index
                and (
                    is_truthy(
                        registry_by_participant.loc[
                            candidate, "IncludeEyeCandidate"
                        ]
                    )
                    or bool(str(
                        registry_by_participant.loc[candidate].get(
                            "EyeExclusionReason", ""
                        )
                    ).strip())
                )
                for candidate in raw_candidates
            )
        )
    ):
        blockers.append(
            f"raw eye directory contains {len(raw_candidates)} filename candidates "
            f"({len(raw_files)} CSV files), while the configured self-audited "
            f"count is {expected_eye_candidates}; "
            "filename inference remains candidate-only and requires manual resolution"
        )
    if aoi_root is None or not aoi_root.exists():
        blockers.append("configured eye.aoi_root is missing")
    elif (
        len(raw_aoi_files) != 9
        and (
            raw_aoi_inventory.empty
            or not raw_aoi_inventory["Resolved"].all()
        )
    ):
        blockers.append(
            f"raw AOI directory contains {len(raw_aoi_files)} JSON files, while "
            "teacher material mentions 9 AOI images; unresolved files require "
            "MappingRole plus AOIResolutionReason in the formal mapping"
        )
    if (
        not raw_inventory.empty
        and not raw_inventory["CoveredByMappingOrExplicitExclusion"].all()
    ):
        blockers.append(
            f"{int((~raw_inventory['CoveredByMappingOrExplicitExclusion']).sum())} "
            "raw eye CSV files are neither formally mapped nor linked to an "
            "explicit participant exclusion reason"
        )
    for _, trial in trials.iterrows():
        csv_path = _resolve(trial.get("CSVFile"), trial_path.parent)
        file_audit = _eye_file_audit(csv_path) if csv_path else {"FileExists": False, "Readable": False}
        inventory_rows.append({
            **{column: trial.get(column) for column in REQUIRED_TRIAL_COLUMNS},
            "ResolvedCSVFile": str(csv_path or ""),
            **file_audit,
        })
        if csv_path and csv_path.exists() and file_audit.get("Readable"):
            for column in _read_csv_header(csv_path):
                dictionary_rows.append({
                    "Source": "eye_csv",
                    "Column": column,
                    "ObservedInFile": str(csv_path),
                })
        try:
            match = _match_scene_mapping(
                trial, mapping, include_candidate=True
            )
            image_id = str(match["AOIImageID"])
            if image_id not in scenes:
                scenes[image_id] = load_scene_masks(match, mapping_base)
            scene = scenes[image_id]
            canvas_ok = (
                file_audit.get("SourceMaxX", np.inf) <= scene.width
                and file_audit.get("SourceMaxY", np.inf) <= scene.height
                and file_audit.get("SourceMinX", -np.inf) >= -1
                and file_audit.get("SourceMinY", -np.inf) >= -1
            )
            coordinate_rows.append({
                "Participant": trial["Participant"],
                "GlobalTrialOrder": trial["GlobalTrialOrder"],
                "AOIImageID": image_id,
                "ImageWidth": scene.width,
                "ImageHeight": scene.height,
                "CoordinateRangeWithinCanvas": bool(canvas_ok),
                "CoordinateSystemConfirmed": bool(match.get("CoordinateSystemConfirmed", False)),
                "ProjectionType": scene.projection_type,
                "ValidSceneReliable": scene.valid_scene_reliable,
            })
            mapping_rows.append({
                "Participant": trial["Participant"],
                "GlobalTrialOrder": trial["GlobalTrialOrder"],
                "AOIImageID": image_id,
                "MappingUnique": True,
                "MappingFormallyConfirmed": str(
                    match.get("MappingRole", "formal")
                ).strip().lower() == "formal",
                "SameImageAcrossBlocks": match.get("SameImageAcrossBlocks"),
            })
        except Exception as exc:
            mapping_rows.append({
                "Participant": trial["Participant"],
                "GlobalTrialOrder": trial["GlobalTrialOrder"],
                "MappingUnique": False,
                "MappingError": f"{type(exc).__name__}: {exc}",
            })
    for scene in scenes.values():
        aoi_quality_rows.append({
            "AOIImageID": scene.image_id,
            "AOIFile": str(scene.source_json),
            "ImageWidth": scene.width,
            "ImageHeight": scene.height,
            "TablePresent": bool(scene.masks["Table"].any()),
            "WindowPresent": bool(scene.masks["Window"].any()),
            "EquipmentPresent": bool(scene.masks["Equipment"].any()),
            "AOIOverlapPixels": scene.overlap_pixels,
            "AOIOverlapPixelsBeforeResolution": scene.original_overlap_pixels,
            "AOIOverlapResolution": scene.overlap_resolution,
            "ValidSceneReliable": scene.valid_scene_reliable,
            "ProjectionType": scene.projection_type,
        })
        area_rows.extend(scene_area_rows(scene))
    inventory = pd.DataFrame(inventory_rows)
    coordinates = pd.DataFrame(coordinate_rows)
    mapping_check = pd.DataFrame(mapping_rows)
    aoi_quality = pd.DataFrame(aoi_quality_rows)
    areas = pd.DataFrame(area_rows)
    if not audit.empty and not audit["ContractPass"].all():
        blockers.append("one or more participants fail the 12-trial/two-block contract")
    observed_order_groups = sorted(
        trials["OrderGroup"].dropna().astype(str).unique()
    )
    if observed_order_groups != ["new order2", "order1", "order2"]:
        blockers.append(
            "expected order1, order2, and new order2; observed "
            f"{observed_order_groups}"
        )
    if inventory.empty or not inventory.get("Readable", pd.Series(dtype=bool)).fillna(False).all():
        blockers.append("one or more eye CSV files are missing or unreadable")
    if not mapping_check.empty and not mapping_check["MappingUnique"].fillna(False).all():
        blockers.append("one or more trials lack a unique manually confirmed AOI mapping")
    if (
        not mapping_check.empty
        and "MappingFormallyConfirmed" in mapping_check
        and not mapping_check["MappingFormallyConfirmed"].fillna(False).all()
    ):
        blockers.append(
            "candidate filename-based scene/AOI mappings are available for review "
            "but have not been manually confirmed"
        )
    if not coordinates.empty and not coordinates["CoordinateSystemConfirmed"].map(is_truthy).all():
        blockers.append("coordinate system has not been manually confirmed for every mapped scene")
    if not coordinates.empty and not coordinates["ValidSceneReliable"].map(is_truthy).all():
        blockers.append("ValidScene is not manually confirmed for every mapped scene")
    if not aoi_quality.empty and aoi_quality["AOIOverlapPixels"].gt(0).any():
        blockers.append("AOI overlap detected; formal fixation classification is blocked")
    if len(scenes) != expected_aoi_images:
        blockers.append(
            f"configured formal mapping expects {expected_aoi_images} AOI images, "
            f"but mapping resolves {len(scenes)}; "
            "resolve with SameImageAcrossBlocks before approval"
        )
    if (
        int(participants["IncludeEyeCandidate"].map(is_truthy).sum())
        != expected_eye_candidates
    ):
        blockers.append(
            f"configured self-audited eye candidates are {expected_eye_candidates}, "
            f"but registry currently marks "
            f"{int(participants['IncludeEyeCandidate'].map(is_truthy).sum())}; "
            "confirm without deleting records"
        )
    order_sequence = (
        trials.groupby([
            "OrderGroup", "Block", "PositionWithinBlock", "GlobalTrialOrder",
            "SceneID", "WWR", "Complexity",
        ], dropna=False)["Participant"]
        .nunique()
        .rename("ParticipantCount")
        .reset_index()
    )
    candidate_summary = (
        audit.merge(
            inventory.groupby("Participant", dropna=False)["Readable"].sum().rename("ReadableTrialCount"),
            on="Participant", how="left",
        )
    )
    outputs = {
        "input_hashes": write_input_hashes(out, stage1_inputs),
        "data_dictionary": write_table(
            pd.DataFrame(dictionary_rows).drop_duplicates(["Source", "Column"]),
            out / "01_data_dictionary.xlsx",
        ),
        "file_inventory": write_table(inventory, out / "02_file_inventory.xlsx"),
        "raw_candidate_inventory": write_table(
            raw_inventory, out / "02b_raw_candidate_inventory.xlsx"
        ),
        "raw_aoi_inventory": write_table(
            raw_aoi_inventory,
            out / "02c_raw_AOI_JSON_inventory.xlsx",
        ),
        "participant_trial_audit": write_table(audit, out / "03_participant_trial_audit.xlsx"),
        "candidate_summary": write_table(candidate_summary, out / "03b_eye_QC_candidate_summary.xlsx"),
        "order_sequence": write_table(order_sequence, out / "04_order_sequence_check.xlsx"),
        "coordinate_report": write_table(coordinates, out / "05_coordinate_system_report.xlsx"),
        "mapping_check": write_table(mapping_check, out / "06_scene_AOI_mapping_check.xlsx"),
        "aoi_quality": write_table(aoi_quality, out / "07_AOI_quality_report.xlsx"),
        "aoi_area": write_table(areas, out / "08_AOI_area_report.xlsx"),
    }
    if scenes:
        preview = next(iter(scenes.values()))
        for mode, name in (
            ("aoi", "FigureS_AOI_overlay_preview.png"),
            ("overlap", "FigureS_AOI_overlap_preview.png"),
            ("valid_scene", "FigureS_ValidScene_preview.png"),
        ):
            _draw_scene_preview(preview, out / name, mode)
            outputs[name] = out / name
    summary = "\n".join([
        "EYE STAGE 1 REVIEW REQUIRED",
        f"Stage fingerprint: {fingerprint}",
        f"Registry participants: {len(participants)}",
        f"Eye candidates: {int(participants['IncludeEyeCandidate'].map(is_truthy).sum())}",
        f"Teacher-reported eye candidates: {teacher_reported_eye_candidates}",
        (
            "Eye candidate count resolution: "
            + str(
                config.get("eye", {}).get(
                    "eye_candidate_count_resolution", ""
                )
            ).strip()
        ),
        f"Trial rows: {len(trials)}",
        f"Raw AOI JSON files: {len(raw_aoi_files)}",
        f"Resolved AOI images: {len(scenes)}",
        f"Teacher-reported AOI images: {teacher_reported_aoi_images}",
        f"Self-audited formal AOI images: {expected_aoi_images}",
        (
            "AOI count resolution: "
            + str(config.get("eye", {}).get("aoi_count_resolution", "")).strip()
        ),
        f"Blocking issues: {len(blockers)}",
        *[f"- {item}" for item in blockers],
        "",
        "Formal statistics are not permitted until AOI_masks_approved.txt is approved.",
    ])
    outputs["review_required"] = write_text(summary, out / "00_STAGE1_REVIEW_REQUIRED.txt")
    outputs["analysis_log"] = write_text(summary, out / "stage1_analysis_log.txt")
    outputs["approval_template"] = write_approval_template(
        out / "AOI_masks_approved.template.txt",
        fingerprint=fingerprint,
        stage="eye-stage1",
        notes=blockers,
    )
    if (
        not blockers
        and bool(eye_config := config.get("eye", {}))
        and is_truthy(eye_config.get("provisional_user_authorization", False))
    ):
        approval = {
            "approved": True,
            "stage": "eye-stage1",
            "stage_fingerprint": fingerprint,
            "approved_by": "user_authorized_codex_self_audit",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approved_triggers": [],
            "notes": [
                "User explicitly authorized Codex to self-review or skip the "
                "manual checkpoint for this run.",
                "Results remain subject to later human AOI/ValidScene verification.",
            ],
        }
        approval_path = out / "AOI_masks_approved.txt"
        approval_path.write_text(
            json.dumps(approval, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["approval"] = approval_path
    outputs["manifest"] = write_run_manifest(
        out,
        stage="eye-stage1",
        fingerprint=fingerprint,
        config_path=config_path,
        arguments={"outdir": str(out)},
        repo_root=repo_root,
        extra={"status": "review_required", "blockers": blockers},
    )
    return outputs


def recompute_tracking_quality(
    frame: pd.DataFrame,
    *,
    validity_accepted: tuple[str, ...] = ("1",),
    validity_rule: str = "both",
) -> tuple[pd.Series, dict[str, Any]]:
    accepted = {str(value).strip().lower() for value in validity_accepted}
    left = frame.get("Validity Left", pd.Series(pd.NA, index=frame.index)).astype(str).str.strip().str.lower()
    right = frame.get("Validity Right", pd.Series(pd.NA, index=frame.index)).astype(str).str.strip().str.lower()
    left_valid = left.isin(accepted)
    right_valid = right.isin(accepted)
    if validity_rule == "either":
        binocular = left_valid | right_valid
    elif validity_rule == "both":
        binocular = left_valid & right_valid
    else:
        raise ValueError("validity_rule must be 'both' or 'either'")
    x_col = (
        "Gaze Point X[px]"
        if "Gaze Point X[px]" in frame
        else "Fixation Point X[px]"
    )
    y_col = (
        "Gaze Point Y[px]"
        if "Gaze Point Y[px]" in frame
        else "Fixation Point Y[px]"
    )
    x = pd.to_numeric(
        frame.get(x_col, pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    y = pd.to_numeric(
        frame.get(y_col, pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    coordinates = x.notna() & y.notna() & x.ge(0) & y.ge(0)
    valid = binocular & coordinates
    software = pd.to_numeric(
        frame.get("Tracking Ratio[%]", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).dropna()
    return valid, {
        "TotalSamples": int(len(frame)),
        "ValidSamples": int(valid.sum()),
        "ValidTrackingRatio": float(valid.mean()) if len(valid) else np.nan,
        "SoftwareTrackingRatio": float(software.iloc[0] / 100.0) if not software.empty else np.nan,
        "TrackingRatioDifference": (
            float(valid.mean() - software.iloc[0] / 100.0)
            if len(valid) and not software.empty else np.nan
        ),
        "LeftEyeValidRatio": float(left_valid.mean()) if len(frame) else np.nan,
        "RightEyeValidRatio": float(right_valid.mean()) if len(frame) else np.nan,
    }


def deduplicate_fixations(
    frame: pd.DataFrame,
    *,
    participant: object,
    global_trial_order: object,
    coordinate_tolerance_px: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixation_columns = [
        "Participant", "GlobalTrialOrder", "FixationIndex",
        "FixationX", "FixationY", "FixationDuration",
        "FixationCoordinateConflict",
    ]
    conflict_columns = [
        "Participant", "GlobalTrialOrder", "FixationIndex",
        "XRangePx", "YRangePx", "Reason",
    ]
    index = pd.to_numeric(frame.get("Fixation Index"), errors="coerce")
    work = frame.loc[index.notna()].copy()
    work["_FixationIndex"] = index.loc[index.notna()].astype(int)
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for fixation_index, sub in work.groupby("_FixationIndex", sort=True):
        x = pd.to_numeric(sub.get("Fixation Point X[px]"), errors="coerce").dropna()
        y = pd.to_numeric(sub.get("Fixation Point Y[px]"), errors="coerce").dropna()
        duration = pd.to_numeric(sub.get("Fixation Duration[ms]"), errors="coerce").dropna()
        x_range = float(x.max() - x.min()) if not x.empty else np.nan
        y_range = float(y.max() - y.min()) if not y.empty else np.nan
        conflict = (
            (pd.notna(x_range) and x_range > coordinate_tolerance_px)
            or (pd.notna(y_range) and y_range > coordinate_tolerance_px)
        )
        if conflict:
            conflicts.append({
                "Participant": participant,
                "GlobalTrialOrder": global_trial_order,
                "FixationIndex": fixation_index,
                "XRangePx": x_range,
                "YRangePx": y_range,
                "Reason": "fixation_coordinate_conflict",
            })
        rows.append({
            "Participant": participant,
            "GlobalTrialOrder": global_trial_order,
            "FixationIndex": int(fixation_index),
            "FixationX": float(x.median()) if not x.empty else np.nan,
            "FixationY": float(y.median()) if not y.empty else np.nan,
            "FixationDuration": float(duration.max()) if not duration.empty else np.nan,
            "FixationCoordinateConflict": conflict,
        })
    return (
        pd.DataFrame(rows, columns=fixation_columns),
        pd.DataFrame(conflicts, columns=conflict_columns),
    )


def classify_fixations(
    fixations: pd.DataFrame,
    scene: SceneMasks,
    *,
    complexity: object,
) -> pd.DataFrame:
    out = fixations.copy()
    x = pd.to_numeric(out["FixationX"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out["FixationY"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(len(x), dtype=int)
    yi = np.zeros(len(y), dtype=int)
    xi[finite] = np.floor(x[finite]).astype(int)
    yi[finite] = np.floor(y[finite]).astype(int)
    in_canvas = (
        finite
        & (xi >= 0) & (xi < scene.width)
        & (yi >= 0) & (yi < scene.height)
    )
    valid_scene = np.zeros(len(out), dtype=bool)
    valid_scene[in_canvas] = scene.valid_scene[yi[in_canvas], xi[in_canvas]]
    hits: dict[str, np.ndarray] = {}
    for name, mask in scene.masks.items():
        hit = np.zeros(len(out), dtype=bool)
        hit[in_canvas] = mask[yi[in_canvas], xi[in_canvas]]
        hits[name] = hit & valid_scene
    hit_count = np.stack(list(hits.values()), axis=0).sum(axis=0)
    if np.any(hit_count > 1):
        raise StageBlockedError(
            f"AOI overlap assigns a fixation to multiple theoretical AOIs for {scene.image_id}"
        )
    category = np.full(len(out), "OffStimulus", dtype=object)
    category[valid_scene] = "Background"
    for name, hit in hits.items():
        category[hit] = name
    if normalize_complexity(complexity) == "C0" and np.any(category == "Equipment"):
        raise StageBlockedError(
            f"C0 trial contains Equipment hits for {scene.image_id}; Equipment must be structural NA"
        )
    out["ValidSceneHit"] = valid_scene
    out["AOICategory"] = category
    return out


def build_eye_trial_metrics(
    events: pd.DataFrame,
    *,
    area_shares: dict[str, float],
    complexity: object,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = events.loc[events["ValidSceneHit"].map(is_truthy)].copy()
    total_tfd = pd.to_numeric(valid["FixationDuration"], errors="coerce").sum(min_count=1)
    rows: list[dict[str, Any]] = []
    for category in ALL_CATEGORIES:
        if category == "Equipment" and normalize_complexity(complexity) == "C0":
            rows.append({
                "AOICategory": category,
                "Visited": pd.NA,
                "FixationCount": pd.NA,
                "TFD": pd.NA,
                "AttentionShare": pd.NA,
                "AOIAreaShare": pd.NA,
                "StructuralNA": True,
            })
            continue
        sub = events.loc[events["AOICategory"].eq(category)]
        tfd = pd.to_numeric(sub["FixationDuration"], errors="coerce").sum(min_count=1)
        if sub.empty:
            tfd = 0.0
        share = (
            tfd / total_tfd
            if category != "OffStimulus"
            and pd.notna(total_tfd)
            and total_tfd > 0
            else np.nan
        )
        rows.append({
            "AOICategory": category,
            "Visited": bool(len(sub)),
            "FixationCount": int(len(sub)),
            "TFD": float(tfd) if pd.notna(tfd) else 0.0,
            "AttentionShare": float(share) if pd.notna(share) else np.nan,
            "AOIAreaShare": area_shares.get(category, np.nan),
            "StructuralNA": False,
        })
    metrics = pd.DataFrame(rows)
    expected = ["Table", "Window", "Background"]
    if normalize_complexity(complexity) == "C1":
        expected.append("Equipment")
    share_sum = pd.to_numeric(
        metrics.loc[metrics["AOICategory"].isin(expected), "AttentionShare"],
        errors="coerce",
    ).sum(min_count=1)
    conservation_pass = bool(pd.notna(share_sum) and np.isclose(share_sum, 1.0, atol=1e-6))
    return metrics, {
        "ValidSceneTFD": float(total_tfd) if pd.notna(total_tfd) else np.nan,
        "AttentionShareSum": float(share_sum) if pd.notna(share_sum) else np.nan,
        "ConservationPass": conservation_pass,
        "ValidSceneFixationCount": int(len(valid)),
        "OffStimulusFixationCount": int(events["AOICategory"].eq("OffStimulus").sum()),
    }


def six_core_outcomes(metrics: pd.DataFrame) -> dict[str, Any]:
    by = metrics.set_index("AOICategory")
    table_share = pd.to_numeric(pd.Series([by.at["Table", "AttentionShare"]]), errors="coerce").iloc[0]
    window_share = pd.to_numeric(pd.Series([by.at["Window", "AttentionShare"]]), errors="coerce").iloc[0]
    table_area = pd.to_numeric(pd.Series([by.at["Table", "AOIAreaShare"]]), errors="coerce").iloc[0]
    window_area = pd.to_numeric(pd.Series([by.at["Window", "AOIAreaShare"]]), errors="coerce").iloc[0]
    table_enrichment = (
        math.log(table_share / table_area)
        if pd.notna(table_share) and table_share > 0 and pd.notna(table_area) and table_area > 0
        else np.nan
    )
    window_enrichment = (
        math.log(window_share / window_area)
        if pd.notna(window_share) and window_share > 0 and pd.notna(window_area) and window_area > 0
        else np.nan
    )
    return {
        "TableShare": table_share,
        "WindowShare": window_share,
        "RawCompetition": table_share - window_share
        if pd.notna(table_share) and pd.notna(window_share) else np.nan,
        "LogTableEnrichment": table_enrichment,
        "LogWindowEnrichment": window_enrichment,
        "AdjustedCompetition": table_enrichment - window_enrichment
        if pd.notna(table_enrichment) and pd.notna(window_enrichment) else np.nan,
    }


def _invoke_r(
    rscript: str,
    script: Path,
    arguments: list[str],
    *,
    required: bool = True,
) -> bool:
    executable = shutil.which(rscript) if not Path(rscript).exists() else rscript
    if not executable:
        if required:
            raise StageBlockedError(
                f"Rscript is unavailable ({rscript}). Install R and restore renv before formal inference."
            )
        return False
    subprocess.run([str(executable), str(script), *arguments], check=True)
    return True


def _package_r_csv_outputs(outdir: Path) -> None:
    for csv_path in outdir.glob("*.csv"):
        if csv_path.name.endswith("_model_input.csv"):
            continue
        write_table(read_table(csv_path), csv_path.with_suffix(".xlsx"))


def run_eye_stage2(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
    r_required: bool = True,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    eye_config = config.get("eye", {})
    participant_path = Path(config["participant_information"])
    trial_path = Path(config["trial_order_mapping"])
    mapping_path = Path(config["scene_aoi_mapping"])
    raw_root_value = str(eye_config.get("raw_root", "")).strip()
    raw_root = Path(raw_root_value) if raw_root_value else None
    aoi_root_value = str(eye_config.get("aoi_root", "")).strip()
    aoi_root = Path(aoi_root_value) if aoi_root_value else None
    stage1_dir = Path(eye_config["stage1_dir"])
    stage1_manifest = json.loads((stage1_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stage1_fp = str(stage1_manifest["stage_fingerprint"])
    current_stage1_fp = stage_fingerprint(
        "eye-stage1", eye_config,
        _eye_stage_input_paths(
            participant_path, trial_path, mapping_path, raw_root, aoi_root
        ),
    )
    if current_stage1_fp != stage1_fp:
        raise StageBlockedError(
            "Eye Stage 1 inputs or configuration changed; rerun Stage 1 and approve again."
        )
    unresolved = stage1_manifest.get("blockers", [])
    if unresolved:
        raise StageBlockedError(
            "Eye Stage 1 still has unresolved blocking findings; update the "
            "manual mappings/confirmations and rerun Stage 1. "
            + " | ".join(str(value) for value in unresolved)
        )
    require_approval(
        stage1_dir / "AOI_masks_approved.txt",
        fingerprint=stage1_fp,
        stage="eye-stage1",
    )
    stage2_inputs = [
            *_eye_stage_input_paths(
                participant_path, trial_path, mapping_path, raw_root, aoi_root
            ),
        stage1_dir / "AOI_masks_approved.txt",
    ]
    fingerprint = stage_fingerprint(
        "eye-stage2",
        eye_config,
        stage2_inputs,
    )
    participants = build_modality_registry(participant_path)
    trials = canonicalize_trials(read_table(trial_path), require_complete=True)
    trials = trials.merge(
        participants[[
            "Participant", "Gender", "ExperienceGroup",
            "IncludeEyeCandidate", "IncludeEEGValid",
        ]],
        on="Participant", how="left", validate="many_to_one",
    )
    trials = trials.loc[trials["IncludeEyeCandidate"].map(is_truthy)].copy()
    mapping = read_table(mapping_path)
    areas = read_table(stage1_dir / "08_AOI_area_report.xlsx")
    primary_threshold = float(eye_config.get("primary_tracking_threshold", 0.6))
    sensitivity = [float(v) for v in eye_config.get("sensitivity_tracking_thresholds", [0.5, 0.7])]
    accepted = tuple(str(v) for v in eye_config.get("validity_accepted", [1]))
    validity_rule = str(eye_config.get("validity_rule", "both"))
    tolerance = float(eye_config.get("fixation_coordinate_tolerance_px", 1.0))
    event_frames: list[pd.DataFrame] = []
    trial_aoi_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    conflict_rows: list[pd.DataFrame] = []
    scenes: dict[str, SceneMasks] = {}
    for _, trial in trials.iterrows():
        csv_path = _resolve(trial["CSVFile"], trial_path.parent)
        base = {
            column: trial.get(column)
            for column in (
                "Participant", "OrderGroup", "Block", "PositionWithinBlock",
                "PositionWithinBlockCentered", "GlobalTrialOrder", "SceneID",
                "WWR", "Complexity", "Gender", "ExperienceGroup",
                "PreviousWWR", "PreviousComplexity", "IncludeEEGValid",
            )
        }
        if csv_path is None or not csv_path.exists():
            exclusion_rows.append({**base, "Reason": "missing_eye_csv"})
            continue
        frame = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        valid_mask, quality = recompute_tracking_quality(
            frame, validity_accepted=accepted, validity_rule=validity_rule
        )
        timestamp = pd.to_numeric(frame.get("Recording Time Stamp[ms]"), errors="coerce")
        quality["RecordingDuration"] = (
            float(timestamp.max() - timestamp.min()) / 1000.0
            if timestamp.notna().sum() > 1 else np.nan
        )
        quality["TrackingPassPrimary"] = bool(
            pd.notna(quality["ValidTrackingRatio"])
            and quality["ValidTrackingRatio"] >= primary_threshold
        )
        for threshold in sensitivity:
            quality[f"TrackingPass_{threshold:.2f}"] = bool(
                pd.notna(quality["ValidTrackingRatio"])
                and quality["ValidTrackingRatio"] >= threshold
            )
        fixations, conflicts = deduplicate_fixations(
            frame.loc[valid_mask].copy(),
            participant=trial["Participant"],
            global_trial_order=trial["GlobalTrialOrder"],
            coordinate_tolerance_px=tolerance,
        )
        if not conflicts.empty:
            conflict_rows.append(conflicts)
        match = _match_scene_mapping(trial, mapping)
        image_id = str(match["AOIImageID"])
        if image_id not in scenes:
            scenes[image_id] = load_scene_masks(match, mapping_path.parent)
        events = classify_fixations(
            fixations, scenes[image_id], complexity=trial["Complexity"]
        )
        for key, value in base.items():
            events[key] = value
        event_frames.append(events)
        area_share = (
            areas.loc[areas["AOIImageID"].astype(str).eq(image_id)]
            .set_index("AOICategory")["AOIAreaShare"]
            .pipe(pd.to_numeric, errors="coerce")
            .to_dict()
        )
        aoi_metrics, conservation = build_eye_trial_metrics(
            events,
            area_shares=area_share,
            complexity=trial["Complexity"],
        )
        for key, value in base.items():
            aoi_metrics[key] = value
        trial_aoi_frames.append(aoi_metrics)
        core = six_core_outcomes(aoi_metrics)
        included = (
            quality["TrackingPassPrimary"]
            and conservation["ConservationPass"]
            and conflicts.empty
        )
        trial_rows.append({
            **base,
            "AOIImageID": image_id,
            **core,
            **conservation,
            "ValidTrackingRatio": quality["ValidTrackingRatio"],
            "EligibleNonTracking": bool(
                conservation["ConservationPass"] and conflicts.empty
            ),
            "IncludedPrimary": included,
        })
        qc_rows.append({**base, **quality, **conservation})
        if not included:
            reasons = []
            if not quality["TrackingPassPrimary"]:
                reasons.append("ValidTrackingRatio_below_60_percent")
            if not conservation["ConservationPass"]:
                reasons.append("attention_share_conservation_failed")
            if not conflicts.empty:
                reasons.append("fixation_coordinate_conflict")
            exclusion_rows.append({**base, "Reason": ";".join(reasons)})
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    trial_aoi = pd.concat(trial_aoi_frames, ignore_index=True) if trial_aoi_frames else pd.DataFrame()
    trial_level = pd.DataFrame(trial_rows)
    qc = pd.DataFrame(qc_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    conflicts = pd.concat(conflict_rows, ignore_index=True) if conflict_rows else pd.DataFrame()
    participant_summary = (
        qc.groupby("Participant", dropna=False)
        .agg(
            CandidateTrials=("GlobalTrialOrder", "nunique"),
            PrimaryValidTrials=("TrackingPassPrimary", "sum"),
            MeanValidTrackingRatio=("ValidTrackingRatio", "mean"),
        )
        .reset_index()
    )
    threshold_rows = []
    for threshold in [*sensitivity[:1], primary_threshold, *sensitivity[1:]]:
        keep = pd.to_numeric(qc["ValidTrackingRatio"], errors="coerce").ge(threshold)
        threshold_rows.append({
            "Threshold": threshold,
            "ValidTrials": int(keep.sum()),
            "ParticipantsWithAtLeastOneValidTrial": int(qc.loc[keep, "Participant"].nunique()),
        })
    threshold_summary = pd.DataFrame(threshold_rows).sort_values("Threshold")
    primary = trial_level.loc[trial_level["IncludedPrimary"].map(is_truthy)].copy()
    core_outcomes = [
        "TableShare", "WindowShare", "RawCompetition",
        "LogTableEnrichment", "LogWindowEnrichment", "AdjustedCompetition",
    ]
    condition_summary = (
        primary.groupby(
            ["Participant", "WWR", "Complexity"], dropna=False
        )[core_outcomes]
        .mean()
        .reset_index()
        if not primary.empty else pd.DataFrame()
    )
    descriptive_rows: list[dict[str, Any]] = []
    if not condition_summary.empty:
        for (wwr, complexity), sub in condition_summary.groupby(["WWR", "Complexity"], dropna=False):
            for metric in (
                "TableShare", "WindowShare", "RawCompetition",
                "LogTableEnrichment", "LogWindowEnrichment", "AdjustedCompetition",
            ):
                values = pd.to_numeric(sub[metric], errors="coerce").dropna()
                descriptive_rows.append({
                    "WWR": wwr,
                    "Complexity": complexity,
                    "Outcome": metric,
                    "NParticipants": int(values.count()),
                    "Mean": float(values.mean()) if len(values) else np.nan,
                    "SD": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "Median": float(values.median()) if len(values) else np.nan,
                    "CI95Low": float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else np.nan,
                    "CI95High": float(values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else np.nan,
                })
    outputs = {
        "input_hashes": write_input_hashes(out, stage2_inputs),
        "fixation_events": write_table(events, out / "01_fixation_event_level_data.xlsx"),
        "trial_qc": write_table(qc, out / "02_trial_quality_control.xlsx"),
        "exclusions": write_table(exclusions, out / "03_exclusion_log.xlsx"),
        "participant_summary": write_table(participant_summary, out / "04_participant_quality_summary.xlsx"),
        "trial_aoi": write_table(trial_aoi, out / "05_trial_AOI_level_data.xlsx"),
        "trial_level": write_table(trial_level, out / "06_trial_level_eye_tracking_data.xlsx"),
        "participant_condition": write_table(condition_summary, out / "07_participant_condition_summary.xlsx"),
        "descriptives": write_table(pd.DataFrame(descriptive_rows), out / "08_core_descriptive_statistics.xlsx"),
        "thresholds": write_table(threshold_summary, out / "13_tracking_threshold_sensitivity.xlsx"),
        "conflicts": write_table(conflicts, out / "fixation_conflicts.xlsx"),
    }
    incomplete = participant_summary["PrimaryValidTrials"].lt(12).any() if not participant_summary.empty else True
    retention_path = out / "participant_retention_approved.txt"
    if incomplete and not retention_path.exists():
        if is_truthy(
            eye_config.get("provisional_user_authorization", False)
        ):
            approval = {
                "approved": True,
                "stage": "eye-stage2-retention",
                "stage_fingerprint": fingerprint,
                "approved_by": "user_authorized_codex_self_audit",
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "approved_triggers": [],
                "notes": [
                    "User authorized continuation after reporting the retained "
                    "trial distribution; no participant-level minimum was invented."
                ],
            }
            retention_path.write_text(
                json.dumps(approval, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            write_approval_template(
                out / "participant_retention_approved.template.txt",
                fingerprint=fingerprint,
                stage="eye-stage2-retention",
                notes=[
                    "At least one participant retains fewer than 12 valid trials."
                ],
            )
            write_run_manifest(
                out,
                stage="eye-stage2",
                fingerprint=fingerprint,
                config_path=config_path,
                arguments={"outdir": str(out)},
                repo_root=repo_root,
                extra={"status": "retention_review_required"},
            )
            raise StageBlockedError(
                "Eye QC outputs were written, but participant retention requires "
                "manual approval."
            )
    if incomplete:
        require_approval(
            retention_path,
            fingerprint=fingerprint,
            stage="eye-stage2-retention",
        )
    r_input = out / "eye_stage2_model_input.csv"
    trial_level.to_csv(r_input, index=False, encoding="utf-8-sig")
    r_script = Path(repo_root) / "analysis" / "r" / "eye_stage2_analysis.R"
    r_ran = _invoke_r(
        str(config.get("rscript", "Rscript")),
        r_script,
        [str(r_input), str(out), str(primary_threshold)],
        required=r_required,
    )
    if r_ran:
        _package_r_csv_outputs(out)
    if r_ran:
        try:
            from paper_analysis.teacher.reporting import (
                save_condition_plot,
                save_histogram,
                write_docx_report,
            )
        except ImportError as exc:
            raise StageBlockedError(
                "Teacher report packaging requires the declared python-docx dependency."
            ) from exc
    if r_ran:
        outputs["figure_tracking"] = save_histogram(
            qc["ValidTrackingRatio"],
            path=out / "Figure1_tracking_quality_distribution.png",
            title="Recomputed valid tracking ratio",
            xlabel="Valid tracking ratio",
            thresholds=[0.5, 0.6, 0.7],
        )
        outputs["figure_raw_competition"] = save_condition_plot(
            primary,
            outcome="RawCompetition",
            path=out / "Figure2_raw_competition_by_condition.png",
            title="Raw competition by categorical WWR and complexity",
        )
        outputs["figure_adjusted_competition"] = save_condition_plot(
            primary,
            outcome="AdjustedCompetition",
            path=out / "Figure3_adjusted_competition_by_condition.png",
            title="Area-adjusted competition by categorical WWR and complexity",
        )
        outputs["figure_retention"] = save_histogram(
            participant_summary["PrimaryValidTrials"],
            path=out / "Figure4_valid_trials_per_participant.png",
            title="Primary-valid trials retained per participant",
            xlabel="Valid trials",
        )
        diagnostics_path = out / "17_core_model_diagnostics.csv"
        diagnostics = read_table(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
        outputs["diagnostics_docx"] = write_docx_report(
            out / "17_core_model_diagnostics.docx",
            title="Eye Stage 2 Model Diagnostics",
            paragraphs=[
                "No model failure was replaced by OLS.",
                "Bounded shares use an ordered-beta mixed model; CR2 is reported "
                "from the registered Gaussian random-intercept companion.",
            ],
            tables=[
                ("Model diagnostics", diagnostics),
                ("QC thresholds", threshold_summary),
            ],
        )
        outputs["report_docx"] = write_docx_report(
            out / "20_eye_stage2_results_report.docx",
            title="Teacher-priority Eye Tracking Stage 2",
            paragraphs=[
                f"Primary threshold: {primary_threshold:.0%}.",
                f"Retained {len(primary)} trials from "
                f"{primary['Participant'].nunique() if not primary.empty else 0} "
                "participants.",
                "WWR was modeled only as a categorical factor.",
            ],
            tables=[
                ("Participant retention", participant_summary),
                ("Descriptive statistics", pd.DataFrame(descriptive_rows)),
            ],
        )
    summary = "\n".join([
        "EYE STAGE 2 SUMMARY",
        f"Stage fingerprint: {fingerprint}",
        f"Candidate participants: {participants['IncludeEyeCandidate'].map(is_truthy).sum()}",
        f"Primary threshold: {primary_threshold:.0%}",
        f"Primary valid trials: {len(primary)}",
        f"Primary participants: {primary['Participant'].nunique() if not primary.empty else 0}",
        f"Fixation conflicts: {len(conflicts)}",
        f"R inference completed: {r_ran}",
        "Stage 3 was not run automatically.",
    ])
    outputs["summary"] = write_text(summary, out / "21_stage2_summary.txt")
    outputs["analysis_log"] = write_text(summary, out / "22_stage2_analysis_log.txt")
    source_copy = out / "23_eye_processing.py"
    shutil.copy2(Path(__file__), source_copy)
    outputs["processing_source"] = source_copy
    r_copy = out / "24_eye_stage2_analysis.R"
    shutil.copy2(r_script, r_copy)
    outputs["r_source"] = r_copy
    outputs["manifest"] = write_run_manifest(
        out,
        stage="eye-stage2",
        fingerprint=fingerprint,
        config_path=config_path,
        arguments={"outdir": str(out)},
        repo_root=repo_root,
        extra={"status": "complete" if r_ran else "python_complete_r_skipped"},
    )
    return outputs


def run_eye_stage3_plan(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stage2_dir = Path(config.get("eye", {})["stage2_dir"])
    trial = read_table(stage2_dir / "06_trial_level_eye_tracking_data.xlsx")
    stage3_config = config.get("stage3", {})
    s3_value = str(stage3_config.get("s3_trial_file", "")).strip()
    fingerprint_inputs: list[str | Path] = [
        stage2_dir / "06_trial_level_eye_tracking_data.xlsx"
    ]
    if s3_value:
        fingerprint_inputs.append(s3_value)
    fingerprint = stage_fingerprint(
        "eye-stage3-plan",
        stage3_config,
        fingerprint_inputs,
    )
    zero_trigger = float(stage3_config.get("zero_rate_trigger", 0.1))
    primary = trial.loc[trial["IncludedPrimary"].map(is_truthy)].copy()
    diagnostic_path = stage2_dir / "17_core_model_diagnostics.csv"
    diagnostics = (
        read_table(diagnostic_path) if diagnostic_path.exists() else pd.DataFrame()
    )
    model_frames = []
    for filename in ("09_familyA_primary_models.csv", "10_familyB_primary_models.csv"):
        path = stage2_dir / filename
        if path.exists():
            model_frames.append(read_table(path))
    model_results = (
        pd.concat(model_frames, ignore_index=True)
        if model_frames else pd.DataFrame()
    )
    trial_aoi_path = stage2_dir / "05_trial_AOI_level_data.xlsx"
    trial_aoi = read_table(trial_aoi_path) if trial_aoi_path.exists() else pd.DataFrame()
    failed_models = (
        int(diagnostics.get("status", pd.Series(dtype=str)).astype(str).ne("fit").sum())
        if not diagnostics.empty else np.nan
    )
    singular_models = (
        int(diagnostics.get("singular", pd.Series(dtype=bool)).map(is_truthy).sum())
        if not diagnostics.empty else np.nan
    )
    position_terms = (
        model_results.loc[
            model_results.get("term", pd.Series(dtype=str))
            .astype(str).str.contains("PositionWithinBlock", case=False, na=False)
        ]
        if not model_results.empty else pd.DataFrame()
    )
    experience_terms = (
        model_results.loc[
            model_results.get("term", pd.Series(dtype=str))
            .astype(str).str.contains("ExperienceGroup", case=False, na=False)
        ]
        if not model_results.empty else pd.DataFrame()
    )
    equipment_c1 = (
        trial_aoi.loc[
            trial_aoi.get("Complexity", pd.Series(dtype=str)).astype(str).eq("C1")
            & trial_aoi.get("AOICategory", pd.Series(dtype=str)).astype(str).eq("Equipment")
        ]
        if not trial_aoi.empty else pd.DataFrame()
    )
    def _minimum_p(frame: pd.DataFrame) -> float:
        for column in ("p.value.BH", "p.value", "p_value"):
            if column in frame:
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not values.empty:
                    return float(values.min())
        return np.nan

    evidence = {
        "A": max(
            float(pd.to_numeric(primary.get("TableShare"), errors="coerce").eq(0).mean()),
            float(pd.to_numeric(primary.get("WindowShare"), errors="coerce").eq(0).mean()),
        ) if not primary.empty else np.nan,
        "B": f"failed_models={failed_models}; singular_models={singular_models}",
        "C": "not automatic; run only with boundary uncertainty approved by researcher",
        "D": "compare signs of raw and area-adjusted competition",
        "E": (
            f"position_term_rows={len(position_terms)}; "
            f"minimum_p={_minimum_p(position_terms)}"
            if not position_terms.empty else "position terms unavailable"
        ),
        "F": "run only for reviewer/time-structure need",
        "G": (
            f"experience_group_term_rows={len(experience_terms)}; "
            f"minimum_p={_minimum_p(experience_terms)}"
            if not experience_terms.empty else "experience-group terms unavailable"
        ),
        "H": (
            f"C1_equipment_rows={len(equipment_c1)}; "
            f"visited_rate={equipment_c1.get('Visited', pd.Series(dtype=bool)).map(is_truthy).mean()}"
            if not equipment_c1.empty else "C1 Equipment evidence unavailable"
        ),
        "I": "run only when needed to explain core outcomes",
        "J": "requires exact S3 trial-level intersection",
        "K": "candidate only when raw and area-adjusted interpretations diverge",
    }
    triggers = ["A"] if pd.notna(evidence["A"]) and evidence["A"] >= zero_trigger else []
    if pd.notna(failed_models) and (failed_models > 0 or singular_models > 0):
        triggers.append("B")
    raw = pd.to_numeric(primary.get("RawCompetition"), errors="coerce")
    adjusted = pd.to_numeric(primary.get("AdjustedCompetition"), errors="coerce")
    if raw.notna().any() and adjusted.notna().any() and np.sign(raw.mean()) != np.sign(adjusted.mean()):
        triggers.extend(["D", "K"])
    lines = [
        "EYE STAGE 3 PLAN",
        f"Stage fingerprint: {fingerprint}",
        f"Candidate triggers: {','.join(triggers) if triggers else 'none'}",
        "",
        "Trigger evidence:",
        *[f"{key}: {value}" for key, value in evidence.items()],
        "",
        "Only triggers explicitly listed in stage3_plan_approved.txt will run.",
    ]
    outputs = {
        "input_hashes": write_input_hashes(out, fingerprint_inputs),
        "plan": write_text("\n".join(lines), out / "stage3_plan.txt"),
        "approval_template": write_approval_template(
            out / "stage3_plan_approved.template.txt",
            fingerprint=fingerprint,
            stage="eye-stage3-plan",
            notes=[f"Candidate triggers: {triggers}"],
        ),
    }
    outputs["manifest"] = write_run_manifest(
        out,
        stage="eye-stage3-plan",
        fingerprint=fingerprint,
        config_path=config_path,
        arguments={"outdir": str(out)},
        repo_root=repo_root,
        extra={"status": "approval_required", "candidate_triggers": triggers},
    )
    return outputs


def run_eye_stage3(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
    r_required: bool = True,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plan_dir = Path(config.get("eye", {})["stage3_plan_dir"])
    manifest = json.loads((plan_dir / "run_manifest.json").read_text(encoding="utf-8"))
    fingerprint = str(manifest["stage_fingerprint"])
    stage2_dir = Path(config.get("eye", {})["stage2_dir"])
    stage3_config = config.get("stage3", {})
    current_inputs: list[str | Path] = [
        stage2_dir / "06_trial_level_eye_tracking_data.xlsx"
    ]
    s3_value = str(stage3_config.get("s3_trial_file", "")).strip()
    if s3_value:
        current_inputs.append(s3_value)
    current_fingerprint = stage_fingerprint(
        "eye-stage3-plan", stage3_config, current_inputs
    )
    if current_fingerprint != fingerprint:
        raise StageBlockedError(
            "Stage 3 plan inputs changed; regenerate and approve the plan."
        )
    approval = require_approval(
        plan_dir / "stage3_plan_approved.txt",
        fingerprint=fingerprint,
        stage="eye-stage3-plan",
    )
    triggers = [str(value).strip().upper() for value in approval.get("approved_triggers", [])]
    stage3_input = read_table(
        stage2_dir / "06_trial_level_eye_tracking_data.xlsx"
    )
    outputs: dict[str, Path] = {
        "input_hashes": write_input_hashes(out, current_inputs)
    }
    if "J" in triggers:
        stage3_config = config.get("stage3", {})
        s3_value = str(stage3_config.get("s3_trial_file", "")).strip()
        s3_path = Path(s3_value) if s3_value else None
        metrics = [str(v) for v in stage3_config.get("stable_eeg_metrics", [])]
        if s3_path is None or not s3_path.is_file():
            raise StageBlockedError(
                "Stage 3 trigger J requires stage3.s3_trial_file."
            )
        if not 1 <= len(metrics) <= 2:
            raise StageBlockedError(
                "Stage 3 trigger J requires exactly 1-2 pre-specified stable EEG metrics."
            )
        s3 = canonicalize_trials(read_table(s3_path))
        missing_metrics = sorted(set(metrics) - set(s3.columns))
        if missing_metrics:
            raise StageBlockedError(
                f"Stage 3 S3 input is missing stable EEG metrics: {missing_metrics}"
            )
        stage3_input = stage3_input.merge(
            s3[["Participant", "GlobalTrialOrder", *metrics]],
            on=["Participant", "GlobalTrialOrder"],
            how="inner",
            validate="one_to_one",
        )
        outputs["s3_intersection"] = write_table(
            stage3_input,
            out / "15_S3_eye_EEG_exact_trial_intersection.xlsx",
        )
        outputs["s3_intersection_counts"] = write_table(
            pd.DataFrame([{
                "Participants": stage3_input["Participant"].nunique(),
                "Trials": len(stage3_input),
                "EEGMetrics": ",".join(metrics),
            }]),
            out / "16_S3_eye_EEG_intersection_counts.xlsx",
        )
    input_path = out / "eye_stage3_model_input.csv"
    stage3_input.to_csv(input_path, index=False, encoding="utf-8-sig")
    r_script = Path(repo_root) / "analysis" / "r" / "eye_stage3_analysis.R"
    r_ran = _invoke_r(
        str(config.get("rscript", "Rscript")),
        r_script,
        [str(input_path), str(out), ",".join(triggers)],
        required=r_required,
    )
    if r_ran:
        _package_r_csv_outputs(out)
    trigger_log = pd.DataFrame({
        "Trigger": list("ABCDEFGHIJK"),
        "Approved": [letter in triggers for letter in "ABCDEFGHIJK"],
        "Reason": [
            "approved by stage3_plan_approved.txt" if letter in triggers
            else "not approved; no output generated"
            for letter in "ABCDEFGHIJK"
        ],
    })
    outputs.update({
        "trigger_log": write_table(trigger_log, out / "01_stage3_trigger_log.xlsx"),
        "summary": write_text(
            "EYE STAGE 3 SUMMARY\n"
            f"Approved triggers: {','.join(triggers) if triggers else 'none'}\n"
            "Only approved analyses were generated.\n",
            out / "17_stage3_summary.txt",
        ),
    })
    outputs["manifest"] = write_run_manifest(
        out,
        stage="eye-stage3-run",
        fingerprint=fingerprint,
        config_path=config_path,
        arguments={"outdir": str(out), "triggers": triggers},
        repo_root=repo_root,
        extra={"status": "complete" if r_ran else "python_complete_r_skipped"},
    )
    return outputs
