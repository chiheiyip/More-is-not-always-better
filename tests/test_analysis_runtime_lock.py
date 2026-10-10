import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from paper_analysis.teacher import runtime_lock, aoi_lock
from paper_analysis.teacher.state import StageBlockedError, file_sha256


def test_python_lock_stops_changed_pillow_or_python(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    lock = {"python": "3.12.10", "system": "Windows", "machine": "AMD64", "packages": {"Pillow": "12.2.0"}}
    (tmp_path / "configs/analysis_python.lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(runtime_lock.platform, "python_version", lambda: "3.12.10")
    monkeypatch.setattr(runtime_lock.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_lock.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(runtime_lock.importlib.metadata, "version", lambda name: "12.2.0")
    assert runtime_lock.validate_python(tmp_path)["status"] == "passed"
    monkeypatch.setattr(runtime_lock.importlib.metadata, "version", lambda name: "10.3.0")
    with pytest.raises(StageBlockedError, match="Pillow: expected 12.2.0"):
        runtime_lock.validate_python(tmp_path)
    monkeypatch.setattr(runtime_lock.platform, "python_version", lambda: "3.12.11")
    with pytest.raises(StageBlockedError, match="python: expected 3.12.10"):
        runtime_lock.validate_python(tmp_path)


def test_equal_area_does_not_hide_different_mask():
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, True], [False, False]])
    assert a.sum() == b.sum()
    assert aoi_lock.mask_digest(a) != aoi_lock.mask_digest(b)
    assert aoi_lock.mask_digest(a) != aoi_lock.mask_digest(a.reshape(1, 4))


def test_aoi_freeze_rejects_one_pixel_and_source_change(tmp_path, monkeypatch):
    from paper_analysis.teacher.eye import load_scene_masks, scene_area_rows
    aoi = tmp_path / "中文.json"
    aoi.write_text(json.dumps({"aoi_image_width_px": 8, "aoi_image_height_px": 6,
        "aois": [{"class_name": "Table", "points": [[0, 0], [3, 0], [3, 3], [0, 3]]}]}))
    mapping = tmp_path / "mapping.xlsx"
    row = {"AOIFile": str(aoi), "AOIImageID": "scene1", "ImageWidth": 8, "ImageHeight": 6,
           "ProjectionType": "unknown", "ValidSceneConfirmed": False}
    pd.DataFrame([row]).to_excel(mapping, index=False)
    scene = load_scene_masks(pd.Series(row), tmp_path)
    areas = pd.DataFrame(scene_area_rows(scene))
    reference = tmp_path / "areas.xlsx"; areas.to_excel(reference, index=False)
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"git_commit": "reference-sha"}))
    monkeypatch.setattr(aoi_lock, "validate_python", lambda: {"lock_sha256": "runtime-sha"})
    config = {"scene_aoi_mapping": str(mapping), "eye": {}}
    target = tmp_path / "frozen.json"
    result = aoi_lock.freeze_aoi(config, target, reference, manifest)
    config["eye"] = {"aoi_pixel_lock": str(target), "aoi_pixel_lock_sha256": result["sha256"]}
    assert aoi_lock.verify_aoi(config, required=True)["scenes"] == 1
    with pytest.raises(StageBlockedError, match="never overwrite"):
        aoi_lock.freeze_aoi(config, target, reference, manifest)
    aoi.write_text(aoi.read_text() + " ")
    with pytest.raises(StageBlockedError, match="AOI source"):
        aoi_lock.verify_aoi(config)
    aoi.write_text(aoi.read_text().rstrip())
    archive = target.with_suffix(".npz"); archive.write_bytes(b"changed")
    with pytest.raises(StageBlockedError, match="mask archive"):
        aoi_lock.verify_aoi(config)
    areas.loc[0, "PixelArea"] += 1; areas.to_excel(reference, index=False)
    with pytest.raises(StageBlockedError, match="exactly reproduce"):
        aoi_lock.freeze_aoi(config, tmp_path / "different.json", reference, manifest)


def test_fresh_preflight_requires_frozen_reference_before_sources(tmp_path, monkeypatch):
    from paper_analysis.teacher.fresh import preflight
    monkeypatch.setattr(runtime_lock, "validate_analysis", lambda *a, **k: {})
    with pytest.raises(StageBlockedError, match="eye.aoi_pixel_lock"):
        preflight({"eye": {}}, tmp_path)


def test_r_mismatch_stops_before_inference_output(tmp_path, monkeypatch):
    from paper_analysis.teacher.r_runner import invoke_r
    def blocked(*a, **k):
        raise StageBlockedError("R environment lock mismatch")
    monkeypatch.setattr(runtime_lock, "validate_r", blocked)
    launcher = tmp_path / "Rscript.exe"; launcher.write_bytes(b"fake")
    output = tmp_path / "out"
    with pytest.raises(StageBlockedError, match="R environment"):
        invoke_r(str(launcher), tmp_path / "model.R", ["input.csv", str(output)])
    assert not output.exists()


def test_actual_registered_r_environments():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".r-env/Lib/R/bin/Rscript.exe").is_file():
        pytest.skip("Registered R environment unavailable")
    main = runtime_lock.validate_r(str(repo / "scripts/portable_rscript.cmd"), profile="primary")
    assert main["R"] == "4.4.2"
    independent = Path("C:/Program Files/R/R-4.5.3/bin/x64/Rscript.exe")
    if independent.is_file() and (repo / ".codex_tmp/r45-lib").is_dir():
        other = runtime_lock.validate_r(str(independent), profile="independent", library=str(repo / ".codex_tmp/r45-lib"))
        assert other["R"] == "4.5.3"
    with pytest.raises(StageBlockedError, match="R expected"):
        runtime_lock.validate_r(str(repo / "scripts/portable_rscript.cmd"), profile="independent")
