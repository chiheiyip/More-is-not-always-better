from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paper_analysis.teacher.eye import SceneMasks
from paper_analysis.teacher.eye_figures import (
    AOI_FILL_ALPHA,
    HEX_AOI_COLORS,
    TOBII_HEATMAP_COLORS,
    _aoi_panel_image,
    _density_map,
    _masked_base,
    _registration_metrics,
    _tobii_heatmap_alpha,
    _tobii_heatmap_cmap,
)


def _scene(tmp_path: Path, image_id: str, *, equipment: bool = True) -> SceneMasks:
    width, height = 100, 80
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    image[:10, :10] = 0
    image_path = tmp_path / f"{image_id}.png"
    Image.fromarray(image).save(image_path)
    valid = np.ones((height, width), dtype=bool)
    valid[:10, :10] = False
    table = np.zeros_like(valid)
    table[40:60, 20:45] = True
    window = np.zeros_like(valid)
    window[15:30, 55:90] = True
    equipment_mask = np.zeros_like(valid)
    if equipment:
        equipment_mask[35:50, 70:85] = True
    return SceneMasks(
        image_id=image_id,
        width=width,
        height=height,
        masks={"Table": table, "Window": window, "Equipment": equipment_mask},
        valid_scene=valid,
        valid_scene_reliable=True,
        projection_type="unknown",
        source_json=tmp_path / f"{image_id}.json",
        base_image=image_path,
        overlap_pixels=0,
    )


def test_locked_aoi_palette_matches_manuscript_contract() -> None:
    assert HEX_AOI_COLORS == {
        "Table": "#DC2D2D",
        "Window": "#EEC428",
        "Equipment": "#2E6ADC",
        "ValidScene": "#28AA5A",
    }


def test_valid_scene_mask_removes_black_border_without_changing_canvas(tmp_path: Path) -> None:
    scene = _scene(tmp_path, "1-C0W15", equipment=False)
    masked = _masked_base(scene)
    assert masked.shape == (80, 100, 3)
    assert np.array_equal(masked[0, 0], [255, 255, 255])
    assert np.array_equal(masked[20, 20], [180, 180, 180])
    panel = _aoi_panel_image(scene)
    assert panel.shape == masked.shape
    assert not scene.masks["Equipment"].any()
    assert AOI_FILL_ALPHA == 0.10
    assert np.array_equal(panel[50, 30], [184, 166, 166])
    assert np.array_equal(panel[40, 30], [220, 45, 45])
    assert np.array_equal(panel[41, 30], [220, 45, 45])


def test_tobii_heatmap_uses_red_for_peak_and_transparent_zero() -> None:
    cmap = _tobii_heatmap_cmap()
    peak_rgb = tuple(round(channel * 255) for channel in cmap(1.0)[:3])
    assert TOBII_HEATMAP_COLORS[-1] == "#D7191C"
    assert peak_rgb == (215, 25, 28)
    alpha = _tobii_heatmap_alpha(np.array([[0.0, 0.25, 1.0]]), vmax=1.0)
    assert alpha[0, 0] == 0.0
    assert 0.0 < alpha[0, 1] < alpha[0, 2] <= 0.82


def test_identity_registration_passes_locked_qc(tmp_path: Path) -> None:
    reference = _scene(tmp_path, "1-C1W45")
    moving = _scene(tmp_path, "2-C1W45")
    x, y = np.meshgrid(np.linspace(12, 90, 8), np.linspace(12, 70, 6))
    points = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
    result = _registration_metrics(
        condition="C1_WWR45",
        reference=reference,
        moving=moving,
        matrix=np.eye(3),
        source_points=points,
        target_points=points,
        inlier_mask=np.ones(len(points), dtype=bool),
        method="sift_ransac_homography",
    )
    assert result.automatic_pass
    assert result.inliers == 48
    assert result.valid_scene_iou == 1.0
    assert result.aoi_union_iou == 1.0


def test_density_map_is_normalized_and_clipped_to_valid_scene(tmp_path: Path) -> None:
    scene = _scene(tmp_path, "1-C0W75", equipment=False)
    events = pd.DataFrame(
        {
            "RegisteredFixationX": [25.0, 25.0, 75.0],
            "RegisteredFixationY": [50.0, 50.0, 20.0],
            "FixationDuration": [100.0, 200.0, 50.0],
        }
    )
    event_density = _density_map(events, scene, weight_column=None, scale=0.5)
    duration_density = _density_map(events, scene, weight_column="FixationDuration", scale=0.5)
    assert np.isclose(event_density.sum(), 1.0)
    assert np.isclose(duration_density.sum(), 1.0)
    assert not np.allclose(event_density, duration_density)
    assert event_density[0, 0] == 0.0
