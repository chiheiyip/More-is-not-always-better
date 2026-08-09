from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from scipy import ndimage

from paper_analysis.teacher.eye import AOI_COLORS, SceneMasks, load_scene_masks
from paper_analysis.teacher.state import file_sha256
from paper_analysis.utils.io import is_truthy, read_table, require_columns, write_table


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7


CONDITION_ORDER = tuple((complexity, wwr) for complexity in (0, 1) for wwr in (15, 45, 75))
HEX_AOI_COLORS = {
    name: "#{:02X}{:02X}{:02X}".format(*rgba[:3])
    for name, rgba in AOI_COLORS.items()
}
AOI_OUTLINE_ORDER = ("Table", "Window", "Equipment")
AOI_FILL_ALPHA = 0.10
TOBII_HEATMAP_COLORS = (
    "#3155B7",
    "#2A8CCF",
    "#2FC56D",
    "#F1E51D",
    "#FF9418",
    "#D7191C",
)
TOBII_HEATMAP_ALPHA_MAX = 0.82
TOBII_HEATMAP_ALPHA_POWER = 0.55
IMAGE_ID_PATTERN = re.compile(r"^(?P<block>[12])-C(?P<complexity>[01])W(?P<wwr>15|45|75)$")


@dataclass(frozen=True)
class RegistrationResult:
    condition: str
    reference_image_id: str
    moving_image_id: str
    method: str
    matrix: np.ndarray
    feature_matches: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error_px: float
    p95_reprojection_error_px: float
    valid_scene_iou: float
    aoi_union_iou: float
    automatic_pass: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "Condition": self.condition,
            "ReferenceImageID": self.reference_image_id,
            "MovingImageID": self.moving_image_id,
            "Method": self.method,
            "FeatureMatches": self.feature_matches,
            "Inliers": self.inliers,
            "InlierRatio": self.inlier_ratio,
            "MedianReprojectionErrorPx": self.median_reprojection_error_px,
            "P95ReprojectionErrorPx": self.p95_reprojection_error_px,
            "ValidSceneIoU": self.valid_scene_iou,
            "AOIUnionIoU": self.aoi_union_iou,
            "RegistrationPass": self.automatic_pass,
        }


def _canonical_mapping(mapping_file: Path) -> tuple[pd.DataFrame, dict[str, SceneMasks]]:
    mapping = read_table(mapping_file)
    require_columns(
        mapping,
        ["AOIImageID", "AOIFile", "BaseImageFile", "ValidSceneFile"],
        "scene AOI mapping",
    )
    if "MappingRole" in mapping:
        role = mapping["MappingRole"].fillna("formal").astype(str).str.strip().str.lower()
        mapping = mapping.loc[role.eq("formal")].copy()
    comparison_columns = [
        column
        for column in ("AOIFile", "BaseImageFile", "ValidSceneFile", "ImageWidth", "ImageHeight")
        if column in mapping
    ]
    inconsistent = []
    for image_id, group in mapping.groupby("AOIImageID", dropna=False):
        if any(group[column].astype(str).nunique(dropna=False) > 1 for column in comparison_columns):
            inconsistent.append(str(image_id))
    if inconsistent:
        raise ValueError(f"Inconsistent duplicate scene mappings: {inconsistent}")
    canonical = mapping.drop_duplicates("AOIImageID").copy()
    expected = {f"{block}-C{complexity}W{wwr}" for block in (1, 2) for complexity, wwr in CONDITION_ORDER}
    observed = set(canonical["AOIImageID"].astype(str))
    if expected - observed:
        raise ValueError(f"Missing AOI image mappings: {sorted(expected - observed)}")
    scenes = {
        str(row["AOIImageID"]): load_scene_masks(row, mapping_file.parent)
        for _, row in canonical.iterrows()
        if str(row["AOIImageID"]) in expected
    }
    return canonical, scenes


def _masked_base(scene: SceneMasks) -> np.ndarray:
    if scene.base_image is None:
        raise FileNotFoundError(f"Base image is missing for {scene.image_id}")
    image = np.asarray(Image.open(scene.base_image).convert("RGB"))
    if image.shape[:2] != (scene.height, scene.width):
        image = np.asarray(
            Image.fromarray(image).resize((scene.width, scene.height), Image.Resampling.LANCZOS)
        )
    result = image.copy()
    result[~scene.valid_scene] = 255
    return result


def _draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int, int],
) -> None:
    if not mask.any():
        return
    rgb = np.asarray(color[:3], dtype=float)
    image[mask] = np.clip(
        (1.0 - AOI_FILL_ALPHA) * image[mask] + AOI_FILL_ALPHA * rgb,
        0,
        255,
    ).astype(np.uint8)
    scale = min(image.shape[:2])
    color_width = max(2, int(round(scale / 400)))
    halo_width = color_width + max(1, int(round(scale / 900)))
    halo = ndimage.binary_dilation(mask, iterations=halo_width) ^ ndimage.binary_erosion(
        mask, iterations=halo_width
    )
    boundary = ndimage.binary_dilation(mask, iterations=color_width) ^ ndimage.binary_erosion(
        mask, iterations=color_width
    )
    image[halo] = 255
    image[boundary] = rgb.astype(np.uint8)


def _aoi_panel_image(scene: SceneMasks) -> np.ndarray:
    image = _masked_base(scene)
    for name in AOI_OUTLINE_ORDER:
        _draw_mask_overlay(image, scene.masks[name] & scene.valid_scene, AOI_COLORS[name])
    return image


def _tobii_heatmap_cmap() -> mpl.colors.LinearSegmentedColormap:
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "tobii_fixation_density", TOBII_HEATMAP_COLORS, N=256
    )
    return cmap.with_extremes(under=(0, 0, 0, 0))


def _tobii_heatmap_alpha(density: np.ndarray, vmax: float) -> np.ndarray:
    relative = np.clip(density / vmax, 0, 1) if vmax > 0 else np.zeros_like(density)
    return np.where(
        relative > 0,
        TOBII_HEATMAP_ALPHA_MAX * np.power(relative, TOBII_HEATMAP_ALPHA_POWER),
        0.0,
    )


def _read_gray(scene: SceneMasks) -> np.ndarray:
    if scene.base_image is None:
        raise FileNotFoundError(f"Base image is missing for {scene.image_id}")
    return np.asarray(Image.open(scene.base_image).convert("L"), dtype=np.uint8)


def _union_aoi(scene: SceneMasks) -> np.ndarray:
    return np.logical_or.reduce([scene.masks[name] for name in ("Table", "Window", "Equipment")])


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _registration_metrics(
    *,
    condition: str,
    reference: SceneMasks,
    moving: SceneMasks,
    matrix: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    inlier_mask: np.ndarray,
    method: str,
) -> RegistrationResult:
    transformed = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), matrix)[:, 0]
    errors = np.linalg.norm(transformed - target_points, axis=1)
    inliers = np.asarray(inlier_mask, dtype=bool)
    inlier_errors = errors[inliers]
    warped_valid = cv2.warpPerspective(
        (moving.valid_scene * 255).astype(np.uint8),
        matrix,
        (reference.width, reference.height),
        flags=cv2.INTER_NEAREST,
    ) > 0
    warped_aoi = cv2.warpPerspective(
        (_union_aoi(moving) * 255).astype(np.uint8),
        matrix,
        (reference.width, reference.height),
        flags=cv2.INTER_NEAREST,
    ) > 0
    feature_matches = int(len(source_points))
    inlier_count = int(inliers.sum())
    inlier_ratio = float(inlier_count / feature_matches) if feature_matches else 0.0
    diagonal = float(np.hypot(reference.width, reference.height))
    median_error = float(np.median(inlier_errors)) if inlier_count else float("inf")
    p95_error = float(np.percentile(inlier_errors, 95)) if inlier_count else float("inf")
    valid_iou = _iou(warped_valid, reference.valid_scene)
    aoi_iou = _iou(warped_aoi, _union_aoi(reference))
    feature_gate = (
        feature_matches >= 40 and inlier_count >= 40 and inlier_ratio >= 0.30
        if method == "sift_ransac_homography"
        else feature_matches >= 8
    )
    passed = bool(
        feature_gate
        and median_error <= diagonal * 0.005
        and p95_error <= diagonal * 0.015
        and valid_iou >= 0.90
        and aoi_iou >= 0.70
    )
    return RegistrationResult(
        condition=condition,
        reference_image_id=reference.image_id,
        moving_image_id=moving.image_id,
        method=method,
        matrix=np.asarray(matrix, dtype=float),
        feature_matches=feature_matches,
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        median_reprojection_error_px=median_error,
        p95_reprojection_error_px=p95_error,
        valid_scene_iou=valid_iou,
        aoi_union_iou=aoi_iou,
        automatic_pass=passed,
    )


def _automatic_registration(condition: str, reference: SceneMasks, moving: SceneMasks) -> RegistrationResult:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=12)
    ref_points, ref_descriptors = detector.detectAndCompute(
        _read_gray(reference), (reference.valid_scene * 255).astype(np.uint8)
    )
    moving_points, moving_descriptors = detector.detectAndCompute(
        _read_gray(moving), (moving.valid_scene * 255).astype(np.uint8)
    )
    if ref_descriptors is None or moving_descriptors is None:
        raise RuntimeError(f"SIFT could not detect features for {condition}")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(moving_descriptors, ref_descriptors, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(good) < 4:
        raise RuntimeError(f"Too few feature matches for {condition}: {len(good)}")
    source = np.float32([moving_points[item.queryIdx].pt for item in good])
    target = np.float32([ref_points[item.trainIdx].pt for item in good])
    matrix, inliers = cv2.findHomography(
        source.reshape(-1, 1, 2),
        target.reshape(-1, 1, 2),
        cv2.RANSAC,
        8.0,
        maxIters=10000,
        confidence=0.999,
    )
    if matrix is None or inliers is None:
        raise RuntimeError(f"RANSAC homography failed for {condition}")
    return _registration_metrics(
        condition=condition,
        reference=reference,
        moving=moving,
        matrix=matrix,
        source_points=source,
        target_points=target,
        inlier_mask=inliers.ravel().astype(bool),
        method="sift_ransac_homography",
    )


def _manual_registration(
    condition: str,
    reference: SceneMasks,
    moving: SceneMasks,
    manual_config: dict[str, Any],
) -> RegistrationResult | None:
    entry = manual_config.get("registrations", {}).get(condition)
    if not entry:
        return None
    source = np.asarray(entry.get("moving_points", []), dtype=np.float32)
    target = np.asarray(entry.get("reference_points", []), dtype=np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 8:
        raise ValueError(f"Manual registration for {condition} requires at least 8 paired points")
    matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None:
        raise RuntimeError(f"Manual homography failed for {condition}")
    return _registration_metrics(
        condition=condition,
        reference=reference,
        moving=moving,
        matrix=matrix,
        source_points=source,
        target_points=target,
        inlier_mask=np.ones(len(source), dtype=bool),
        method="manual_control_point_homography",
    )


def _build_registrations(
    scenes: dict[str, SceneMasks], manual_config_file: Path | None
) -> dict[tuple[int, int], RegistrationResult]:
    manual = {"registrations": {}}
    if manual_config_file is not None and manual_config_file.is_file():
        manual = json.loads(manual_config_file.read_text(encoding="utf-8"))
    results: dict[tuple[int, int], RegistrationResult] = {}
    failures: list[str] = []
    for complexity, wwr in CONDITION_ORDER:
        condition = f"C{complexity}_WWR{wwr}"
        reference = scenes[f"1-C{complexity}W{wwr}"]
        moving = scenes[f"2-C{complexity}W{wwr}"]
        automatic = _automatic_registration(condition, reference, moving)
        result = automatic
        if not automatic.automatic_pass:
            fallback = _manual_registration(condition, reference, moving, manual)
            if fallback is not None:
                result = fallback
        if not result.automatic_pass:
            failures.append(
                f"{condition}: method={result.method}, inliers={result.inliers}, "
                f"ratio={result.inlier_ratio:.3f}, median={result.median_reprojection_error_px:.2f}, "
                f"p95={result.p95_reprojection_error_px:.2f}, "
                f"ValidSceneIoU={result.valid_scene_iou:.3f}, AOIIoU={result.aoi_union_iou:.3f}"
            )
        results[(complexity, wwr)] = result
    if failures:
        raise RuntimeError("Scene registration QC failed; formal heatmaps were not produced:\n" + "\n".join(failures))
    return results


def _prepare_events(run_root: Path, scenes: dict[str, SceneMasks]) -> pd.DataFrame:
    stage2 = run_root / "02_eye_stage2"
    fixation = read_table(stage2 / "01_fixation_event_level_data.xlsx")
    quality = read_table(stage2 / "02_trial_quality_control.xlsx")
    scene_map = read_table(run_root / "01_eye_stage1" / "06_scene_AOI_mapping_check.xlsx")
    keys = ["Participant", "GlobalTrialOrder"]
    require_columns(
        fixation,
        keys + ["FixationX", "FixationY", "FixationDuration", "ValidSceneHit"],
        "fixation event data",
    )
    require_columns(quality, keys + ["TrackingPassPrimary"], "trial quality control")
    require_columns(scene_map, keys + ["AOIImageID"], "scene AOI mapping check")
    quality = quality[keys + ["TrackingPassPrimary"]].drop_duplicates(keys)
    scene_map = scene_map[keys + ["AOIImageID"]].drop_duplicates(keys)
    events = fixation.merge(quality, on=keys, how="left", validate="many_to_one")
    events = events.merge(scene_map, on=keys, how="left", validate="many_to_one")
    events = events.loc[
        events["TrackingPassPrimary"].map(is_truthy) & events["ValidSceneHit"].map(is_truthy)
    ].copy()
    if events["AOIImageID"].isna().any():
        raise ValueError("AOIImageID is missing for QC-passed fixation events")
    parsed = events["AOIImageID"].astype(str).str.extract(IMAGE_ID_PATTERN)
    if parsed.isna().any(axis=None):
        bad = sorted(events.loc[parsed.isna().any(axis=1), "AOIImageID"].astype(str).unique())
        raise ValueError(f"Unrecognized AOIImageID values: {bad}")
    events["BlockFromAOIImageID"] = parsed["block"].astype(int)
    events["ComplexityFromAOIImageID"] = parsed["complexity"].astype(int)
    events["WWRFromAOIImageID"] = parsed["wwr"].astype(int)
    events["OriginalFixationX"] = pd.to_numeric(events["FixationX"], errors="coerce")
    events["OriginalFixationY"] = pd.to_numeric(events["FixationY"], errors="coerce")
    events["FixationDuration"] = pd.to_numeric(events["FixationDuration"], errors="coerce")
    if events[["OriginalFixationX", "OriginalFixationY"]].isna().any(axis=None):
        raise ValueError("QC-passed fixation events contain missing coordinates")
    events["RegisteredFixationX"] = events["OriginalFixationX"]
    events["RegisteredFixationY"] = events["OriginalFixationY"]
    events["RegistrationIncluded"] = True
    return events


def _register_events(
    events: pd.DataFrame,
    scenes: dict[str, SceneMasks],
    registrations: dict[tuple[int, int], RegistrationResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = events.copy()
    retention_rows = []
    for complexity, wwr in CONDITION_ORDER:
        result = registrations[(complexity, wwr)]
        reference = scenes[result.reference_image_id]
        selector = out["AOIImageID"].eq(result.moving_image_id)
        moving = out.loc[selector, ["OriginalFixationX", "OriginalFixationY"]].to_numpy(np.float32)
        transformed = cv2.perspectiveTransform(moving.reshape(-1, 1, 2), result.matrix)[:, 0]
        out.loc[selector, "RegisteredFixationX"] = transformed[:, 0]
        out.loc[selector, "RegisteredFixationY"] = transformed[:, 1]
        x = np.rint(transformed[:, 0]).astype(int)
        y = np.rint(transformed[:, 1]).astype(int)
        inside = (x >= 0) & (x < reference.width) & (y >= 0) & (y < reference.height)
        valid_indices = np.flatnonzero(inside)
        inside[valid_indices] &= reference.valid_scene[y[valid_indices], x[valid_indices]]
        out.loc[selector, "RegistrationIncluded"] = inside
        retention = float(inside.mean()) if len(inside) else 1.0
        retention_rows.append(
            {
                "Condition": result.condition,
                "Block2Fixations": int(len(inside)),
                "Block2FixationsRetained": int(inside.sum()),
                "Block2RetentionRatio": retention,
            }
        )
        if retention < 0.99:
            raise RuntimeError(
                f"{result.condition} retained only {retention:.3%} of Block 2 ValidScene fixations"
            )
    return out, pd.DataFrame(retention_rows)


def _density_map(
    events: pd.DataFrame,
    scene: SceneMasks,
    *,
    weight_column: str | None,
    scale: float = 0.25,
    sigma_fraction: float = 0.02,
) -> np.ndarray:
    grid_width = max(1, int(round(scene.width * scale)))
    grid_height = max(1, int(round(scene.height * scale)))
    weights = None
    if weight_column is not None:
        weights = pd.to_numeric(events[weight_column], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    density, _, _ = np.histogram2d(
        events["RegisteredFixationY"].to_numpy(float),
        events["RegisteredFixationX"].to_numpy(float),
        bins=(grid_height, grid_width),
        range=((0, scene.height), (0, scene.width)),
        weights=weights,
    )
    density = ndimage.gaussian_filter(
        density.astype(float), sigma=sigma_fraction * min(grid_width, grid_height), mode="constant"
    )
    mask = np.asarray(
        Image.fromarray((scene.valid_scene * 255).astype(np.uint8)).resize(
            (grid_width, grid_height), Image.Resampling.NEAREST
        )
    ) > 0
    density[~mask] = 0.0
    total = float(density.sum())
    return density / total if total else density


def _save_figure_bundle(fig: plt.Figure, basename: Path) -> dict[str, Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for extension in ("svg", "pdf", "png", "tiff"):
        path = basename.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[extension] = path
    plt.close(fig)
    return outputs


def _plot_aoi_overview(scenes: dict[str, SceneMasks], basename: Path) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 120 / 25.4), constrained_layout=False)
    for index, (complexity, wwr) in enumerate(CONDITION_ORDER):
        ax = axes.flat[index]
        scene = scenes[f"1-C{complexity}W{wwr}"]
        ax.imshow(_aoi_panel_image(scene))
        ax.set_title(f"({chr(97 + index)}) C{complexity}–WWR{wwr}", fontsize=7, pad=2)
        ax.set_axis_off()
    labels = {"Table": "Table", "Window": "Window", "Equipment": "Equipment (C1 only)"}
    legend = [
        Line2D([0], [0], color=HEX_AOI_COLORS[name], linewidth=2.2, label=labels[name])
        for name in AOI_OUTLINE_ORDER
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        fontsize=6,
        frameon=False,
    )
    fig.subplots_adjust(left=0.008, right=0.992, top=0.97, bottom=0.105, wspace=0.025, hspace=0.08)
    return _save_figure_bundle(fig, basename)


def _global_vmax(densities: dict[tuple[int, int], np.ndarray]) -> float:
    positive = np.concatenate([value[value > 0] for value in densities.values()])
    return float(np.quantile(positive, 0.995)) if len(positive) else 1.0


def _plot_density_overview(
    *,
    scenes: dict[str, SceneMasks],
    events: pd.DataFrame,
    basename: Path,
    weight_column: str | None,
    colorbar_label: str,
) -> tuple[dict[str, Path], pd.DataFrame]:
    densities: dict[tuple[int, int], np.ndarray] = {}
    counts = []
    for complexity, wwr in CONDITION_ORDER:
        selector = (
            events["ComplexityFromAOIImageID"].eq(complexity)
            & events["WWRFromAOIImageID"].eq(wwr)
            & events["RegistrationIncluded"].map(is_truthy)
        )
        condition_events = events.loc[selector]
        scene = scenes[f"1-C{complexity}W{wwr}"]
        densities[(complexity, wwr)] = _density_map(
            condition_events, scene, weight_column=weight_column
        )
        counts.append(
            {
                "Complexity": complexity,
                "WWR": wwr,
                "Participants": int(condition_events["ParticipantCode"].nunique()),
                "Trials": int(condition_events[["ParticipantCode", "GlobalTrialOrder"]].drop_duplicates().shape[0]),
                "Fixations": int(len(condition_events)),
                "FixationDurationMs": float(condition_events["FixationDuration"].sum()),
            }
        )
    vmax = _global_vmax(densities)
    cmap = _tobii_heatmap_cmap()
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 120 / 25.4), constrained_layout=True)
    for index, (complexity, wwr) in enumerate(CONDITION_ORDER):
        ax = axes.flat[index]
        scene = scenes[f"1-C{complexity}W{wwr}"]
        density = densities[(complexity, wwr)]
        ax.imshow(_masked_base(scene), extent=(0, scene.width, scene.height, 0))
        alpha = _tobii_heatmap_alpha(density, vmax)
        ax.imshow(
            density,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            alpha=alpha,
            extent=(0, scene.width, scene.height, 0),
            interpolation="bilinear",
        )
        ax.set_title(f"({chr(97 + index)}) C{complexity}–WWR{wwr}", fontsize=7, pad=2)
        ax.set_axis_off()
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=vmax), cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.022, pad=0.012)
    colorbar.set_label(colorbar_label, fontsize=6)
    colorbar.ax.tick_params(labelsize=5, length=2)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.015, hspace=0.02, wspace=0.02)
    return _save_figure_bundle(fig, basename), pd.DataFrame(counts)


def _deidentify_participants(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    participants = sorted(result["Participant"].astype(str).unique())
    codes = {participant: f"P{index:03d}" for index, participant in enumerate(participants, start=1)}
    result["ParticipantCode"] = result["Participant"].astype(str).map(codes)
    return result


def build_eye_scene_figures(
    run_root: str | Path,
    mapping_file: str | Path,
    output_dir: str | Path | None = None,
    *,
    manual_registration_file: str | Path | None = None,
    expected_participants: int | None = None,
    expected_trials: int | None = None,
    expected_fixations: int | None = None,
) -> dict[str, Path]:
    """Build the unified six-scene AOI plate and registered fixation heatmaps."""

    run_root = Path(run_root)
    mapping_file = Path(mapping_file)
    output = Path(output_dir) if output_dir is not None else run_root / "09_eye_figures"
    output.mkdir(parents=True, exist_ok=True)
    _, scenes = _canonical_mapping(mapping_file)
    registrations = _build_registrations(
        scenes,
        Path(manual_registration_file) if manual_registration_file else None,
    )
    events = _prepare_events(run_root, scenes)
    events, retention = _register_events(events, scenes, registrations)
    events = _deidentify_participants(events)
    actual_participants = int(events["ParticipantCode"].nunique())
    actual_trials = int(
        events[["ParticipantCode", "GlobalTrialOrder"]].drop_duplicates().shape[0]
    )
    mismatches = []
    if expected_participants is not None and actual_participants != expected_participants:
        mismatches.append(f"participants expected={expected_participants} actual={actual_participants}")
    if expected_trials is not None and actual_trials != expected_trials:
        mismatches.append(f"trials expected={expected_trials} actual={actual_trials}")
    if expected_fixations is not None and len(events) != expected_fixations:
        mismatches.append(f"fixations expected={expected_fixations} actual={len(events)}")
    if mismatches:
        raise RuntimeError(
            "Unexpected 60% QC figure sample: " + "; ".join(mismatches)
        )

    outputs: dict[str, Path] = {}
    aoi_outputs = _plot_aoi_overview(scenes, output / "FigureS_AOI_regions_unified")
    outputs.update({f"aoi_{key}": value for key, value in aoi_outputs.items()})
    event_outputs, condition_counts = _plot_density_overview(
        scenes=scenes,
        events=events,
        basename=output / "Figure6_fixation_event_density",
        weight_column=None,
        colorbar_label="Relative fixation-event density",
    )
    outputs.update({f"event_density_{key}": value for key, value in event_outputs.items()})
    duration_outputs, _ = _plot_density_overview(
        scenes=scenes,
        events=events,
        basename=output / "FigureS_fixation_duration_density",
        weight_column="FixationDuration",
        colorbar_label="Relative duration-weighted density",
    )
    outputs.update({f"duration_density_{key}": value for key, value in duration_outputs.items()})

    registration_frame = pd.DataFrame([result.as_record() for result in registrations.values()])
    registration_frame = registration_frame.merge(retention, on="Condition", how="left", validate="one_to_one")
    outputs["registration_qc"] = write_table(
        registration_frame, output / "Figure6_registration_qc.csv"
    )
    outputs["condition_counts"] = write_table(
        condition_counts, output / "Figure6_condition_counts.csv"
    )
    source_columns = [
        "ParticipantCode",
        "GlobalTrialOrder",
        "FixationIndex",
        "AOIImageID",
        "BlockFromAOIImageID",
        "ComplexityFromAOIImageID",
        "WWRFromAOIImageID",
        "OriginalFixationX",
        "OriginalFixationY",
        "RegisteredFixationX",
        "RegisteredFixationY",
        "FixationDuration",
        "AOICategory",
        "RegistrationIncluded",
    ]
    source_columns = [column for column in source_columns if column in events]
    outputs["source_data"] = write_table(
        events[source_columns], output / "Figure6_fixation_source_data.csv"
    )
    outputs["aoi_palette"] = write_table(
        pd.DataFrame(
            [{"AOICategory": name, "HexColor": HEX_AOI_COLORS[name]} for name in HEX_AOI_COLORS]
        ),
        output / "FigureS_AOI_palette.csv",
    )
    transforms = {
        "schema_version": 1,
        "reference_policy": "Block 1 is the reference canvas for each WWR x Complexity condition.",
        "registrations": {
            result.condition: {
                "reference_image_id": result.reference_image_id,
                "moving_image_id": result.moving_image_id,
                "method": result.method,
                "matrix": result.matrix.tolist(),
                "reference_image_sha256": file_sha256(scenes[result.reference_image_id].base_image),
                "moving_image_sha256": file_sha256(scenes[result.moving_image_id].base_image),
            }
            for result in registrations.values()
        },
    }
    transform_path = output / "Figure6_registration_transforms.json"
    transform_path.write_text(json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["registration_transforms"] = transform_path
    qa = pd.DataFrame(
        [
            {
                "FigureID": "FigureS_AOI_regions_unified",
                "CoreConclusion": "AOI categories use one fixed semantic palette across all six conditions.",
                "VisualEncoding": "Thick AOI outlines with a white contrast halo and 10% semantic-color fill.",
                "Participants": np.nan,
                "Trials": np.nan,
                "FixationsBeforeRegistration": np.nan,
                "FixationsPlotted": np.nan,
                "RegistrationPass": bool(registration_frame["RegistrationPass"].all()),
                "BlackBorderPolicy": "Pixels outside the confirmed ValidScene mask are rendered white.",
                "SourceData": outputs["aoi_palette"].name,
                "ExportPass": all(path.is_file() and path.stat().st_size > 0 for path in aoi_outputs.values()),
            },
            {
                "FigureID": "Figure6_fixation_event_density",
                "CoreConclusion": "QC-passed fixation-event density is directly comparable across six registered conditions.",
                "VisualEncoding": "Shared blue-green-yellow-red Tobii-style scale; red is the highest density.",
                "Participants": int(events["ParticipantCode"].nunique()),
                "Trials": int(events[["ParticipantCode", "GlobalTrialOrder"]].drop_duplicates().shape[0]),
                "FixationsBeforeRegistration": int(len(events)),
                "FixationsPlotted": int(events["RegistrationIncluded"].map(is_truthy).sum()),
                "RegistrationPass": bool(registration_frame["RegistrationPass"].all()),
                "BlackBorderPolicy": "Pixels outside the confirmed ValidScene mask are rendered white.",
                "SourceData": outputs["source_data"].name,
                "ExportPass": all(path.is_file() and path.stat().st_size > 0 for path in event_outputs.values()),
            },
            {
                "FigureID": "FigureS_fixation_duration_density",
                "CoreConclusion": "Duration weighting does not alter the registered six-condition spatial comparison contract.",
                "VisualEncoding": "Shared blue-green-yellow-red Tobii-style scale; red is the highest density.",
                "Participants": int(events["ParticipantCode"].nunique()),
                "Trials": int(events[["ParticipantCode", "GlobalTrialOrder"]].drop_duplicates().shape[0]),
                "FixationsBeforeRegistration": int(len(events)),
                "FixationsPlotted": int(events["RegistrationIncluded"].map(is_truthy).sum()),
                "RegistrationPass": bool(registration_frame["RegistrationPass"].all()),
                "BlackBorderPolicy": "Pixels outside the confirmed ValidScene mask are rendered white.",
                "SourceData": outputs["source_data"].name,
                "ExportPass": all(path.is_file() and path.stat().st_size > 0 for path in duration_outputs.values()),
            },
        ]
    )
    outputs["figure_qa"] = write_table(qa, output / "eye_scene_figure_qa.csv")
    summary = {
        "status": "complete",
        "run_id": run_root.name,
        "mapping_file_sha256": file_sha256(mapping_file),
        "primary_tracking_threshold": 0.60,
        "participants": int(events["ParticipantCode"].nunique()),
        "trials": int(events[["ParticipantCode", "GlobalTrialOrder"]].drop_duplicates().shape[0]),
        "valid_scene_fixations_before_registration": int(len(events)),
        "fixations_plotted": int(events["RegistrationIncluded"].map(is_truthy).sum()),
        "all_registrations_passed": bool(registration_frame["RegistrationPass"].all()),
        "palette": HEX_AOI_COLORS,
        "aoi_rendering": "thick_outline_white_contrast_halo_10pct_fill",
        "aoi_fill_alpha": AOI_FILL_ALPHA,
        "heatmap_colors_low_to_high": list(TOBII_HEATMAP_COLORS),
        "heatmap_peak_color": TOBII_HEATMAP_COLORS[-1],
    }
    summary_path = output / "eye_scene_figure_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs
