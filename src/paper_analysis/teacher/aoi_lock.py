"""Exact AOI pixel evidence, separate from historical statistical outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime_lock import validate_python
from .state import StageBlockedError, file_sha256


def mask_digest(mask: np.ndarray) -> str:
    # Shape, row order and packed-bit order are explicit; avoid NPZ metadata hashes.
    array = np.asarray(mask, dtype=bool)
    header = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + np.packbits(array.ravel(order="C"), bitorder="little").tobytes()).hexdigest()


def collect_aoi(mapping_path: str | Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    from .eye import load_scene_masks, scene_area_rows, _resolve
    path = Path(mapping_path)
    mapping = pd.read_excel(path)
    records, arrays, seen = [], {}, {}
    for _, row in mapping.iterrows():
        scene = load_scene_masks(row, path.parent)
        sources = {}
        for field in ("AOIFile", "BaseImageFile", "ValidSceneFile"):
            source = _resolve(row.get(field), path.parent)
            sources[field] = file_sha256(source) if source and source.is_file() else None
        masks = {**scene.masks, "ValidScene": scene.valid_scene}
        record = {"image_id": scene.image_id, "width": scene.width, "height": scene.height,
                  "source_sha256": sources, "overlap_resolution": scene.overlap_resolution,
                  "valid_scene_reliable": scene.valid_scene_reliable,
                  "projection_type": scene.projection_type,
                  "mask_sha256": {name: mask_digest(mask) for name, mask in sorted(masks.items())},
                  "mask_archive_keys": {name: f"s{hashlib.sha256(scene.image_id.encode()).hexdigest()[:16]}_{name}" for name in masks},
                  "areas": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                             for k, v in area.items()} for area in scene_area_rows(scene)]}
        if scene.image_id in seen:
            if seen[scene.image_id] != record:
                raise StageBlockedError(f"Conflicting AOI mapping for {scene.image_id}")
            continue
        seen[scene.image_id] = record
        for name, mask in masks.items():
            arrays[record["mask_archive_keys"][name]] = mask
        records.append(record)
    if not records:
        raise StageBlockedError("Cannot lock empty AOI mapping")
    return sorted(records, key=lambda r: r["image_id"]), arrays


def verify_aoi(config: dict, *, required: bool = False) -> dict | None:
    eye = config.get("eye", {})
    value = eye.get("aoi_pixel_lock")
    if not value:
        if required:
            raise StageBlockedError("Fresh analysis requires eye.aoi_pixel_lock and aoi_pixel_lock_sha256; "
                                   "freeze verified AOI evidence before running")
        return None
    path = Path(value)
    if not path.is_file() or file_sha256(path) != eye.get("aoi_pixel_lock_sha256"):
        raise StageBlockedError("AOI pixel lock missing or SHA256 mismatch")
    environment = validate_python()
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != 1 or lock["python_lock_sha256"] != environment["lock_sha256"]:
        raise StageBlockedError("AOI evidence uses a different environment lock")
    archive = path.parent / lock["mask_archive"]
    if not archive.is_file() or file_sha256(archive) != lock["mask_archive_sha256"]:
        raise StageBlockedError("Frozen AOI mask archive changed or missing")
    current, _ = collect_aoi(config["scene_aoi_mapping"])
    if current != lock["scenes"]:
        expected = {r["image_id"]: r for r in lock["scenes"]}
        changed = [r["image_id"] for r in current if expected.get(r["image_id"]) != r]
        raise StageBlockedError(f"AOI source, pixel mask, area or mapping changed: {changed}; "
                               "do not automatically refresh the frozen reference")
    return {"status": "passed", "scenes": len(current), "lock_sha256": file_sha256(path),
            "mask_archive_sha256": lock["mask_archive_sha256"], "comparison": "exact"}


def freeze_aoi(config: dict, target: Path, reference_area: Path, reference_run: Path) -> dict:
    environment = validate_python()
    if target.exists():
        raise StageBlockedError("AOI lock already exists; never overwrite the frozen reference")
    if not reference_run.is_file():
        raise StageBlockedError("Reference run manifest missing")
    records, arrays = collect_aoi(config["scene_aoi_mapping"])
    calculated = pd.DataFrame([area for record in records for area in record["areas"]])
    reference = pd.read_excel(reference_area)
    for field in ["SphericalWeightedArea", "SphericalWeightedAreaShare"]:
        calculated[field] = pd.to_numeric(calculated[field])
    keys = ["AOIImageID", "AOICategory"]
    for table in (reference, calculated):
        if table.duplicated(keys).any() or table[keys].isna().any().any():
            raise StageBlockedError("Duplicate or missing AOI area keys")
    fields = list(calculated.columns)
    try:
        calculated = calculated[fields].sort_values(keys).reset_index(drop=True)
        reference = reference[fields].sort_values(keys).reset_index(drop=True)
        # Integer areas are exact; XLSX serializes floating shares to decimal text.
        pd.testing.assert_frame_equal(calculated[[*keys, "PixelArea", "ValidScenePixels"]],
                                      reference[[*keys, "PixelArea", "ValidScenePixels"]],
                                      check_dtype=False, check_exact=True)
        pd.testing.assert_frame_equal(
            calculated, reference, check_dtype=False, check_exact=False, rtol=1e-15, atol=1e-15)
    except (AssertionError, KeyError) as exc:
        raise StageBlockedError("AOI lock must exactly reproduce the declared formal area table") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(".npz")
    if archive.exists():
        raise StageBlockedError("AOI mask archive already exists")
    np.savez_compressed(archive, **arrays)
    manifest = json.loads(reference_run.read_text(encoding="utf-8"))
    value = {"schema": 1, "python_lock_sha256": environment["lock_sha256"],
             "reference_area": str(reference_area.resolve()), "reference_area_sha256": file_sha256(reference_area),
             "reference_run_manifest": str(reference_run.resolve()), "reference_run_manifest_sha256": file_sha256(reference_run),
             "reference_calculation_sha": manifest["git_commit"],
             "mask_archive": archive.name, "mask_archive_sha256": file_sha256(archive), "scenes": records}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    return {"path": str(target.resolve()), "sha256": file_sha256(target), "scenes": len(records)}
