from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_analysis.intake.pipeline import build_manifests
from paper_analysis.stats.optimized import run_optimized_analysis


def test_intake_derives_cohort_from_scene_eye_record_id(tmp_path: Path) -> None:
    participants = tmp_path / "participants.csv"
    scenes = tmp_path / "scenes.csv"
    participants.write_text("participant_id,Experience,Gender\nP01,偶尔,Female\nP02,有时,Male\n", encoding="utf-8")
    scenes.write_text(
        "participant_id,scene_id,WWR,Complexity,eye_record_id\n"
        "P01,1,15,0,260203123456\nP02,1,15,0,260527123456\n",
        encoding="utf-8",
    )
    paths = build_manifests(participants, scenes, tmp_path / "out")
    out = pd.read_csv(paths["participants_standardized"]).set_index("participant_id")
    assert out.loc["P01", "DateBatch"] == "Initial"
    assert out.loc["P02", "DateBatch"] == "Supplement"
    assert bool(out.loc["P02", "SupplementFlag"]) is True


def test_optimized_analysis_keeps_grains_separate_and_flags_unstable_synthetic_fit(tmp_path: Path) -> None:
    rows = []
    for participant_index, (participant, record, experience, gender) in enumerate([
        ("P01", "260203120000", "Low", "Female"),
        ("P02", "260204120000", "High", "Male"),
        ("P03", "260527120000", "Low", "Female"),
        ("P04", "260528120000", "High", "Male"),
    ]):
        for scene_id, (wwr, complexity) in enumerate([(15, 0), (45, 0), (75, 0), (15, 1), (45, 1), (75, 1)], start=1):
            for aoi, offset in [("table", 0.0), ("window", 0.1)]:
                rows.append({
                    "participant_id": participant, "scene_id": scene_id, "eye_record_id": record,
                    "WWR": wwr, "Complexity": complexity, "ExperienceGroup": experience, "Gender": gender,
                    "Age": 22, "block": 1 if scene_id <= 3 else 2, "position": (scene_id - 1) % 3 + 1,
                        "class_name": aoi, "visited": not (aoi == "window" and scene_id in {2, 5}), "FCR": 0.2 + offset + wwr / 1000 + participant_index * 0.01,
                        "TFD_ms": 500 + wwr + offset + participant_index * 3, "TTFF_ms": 100 + wwr + participant_index * 2, "attention_share": 0.3 + offset + participant_index * 0.01,
                        "q_S1": 4 + wwr / 100 + participant_index * 0.02, "q_S2": 4 + wwr / 100 + participant_index * 0.02, "q_S3": 4 + wwr / 100 + participant_index * 0.02,
                        "q_S4": 4 + wwr / 100 + participant_index * 0.02, "q_S5": 4 + wwr / 100 + participant_index * 0.02,
                        "eeg_F_theta": 1 + wwr / 100 + participant_index * 0.02, "eeg_F_alpha": 1 + wwr / 100 + participant_index * 0.02, "eeg_F_beta": 1 + wwr / 100 + participant_index * 0.02,
                        "eeg_P_theta": 1 + wwr / 100 + participant_index * 0.02, "eeg_P_alpha": 1 + wwr / 100 + participant_index * 0.02, "eeg_P_beta": 1 + wwr / 100 + participant_index * 0.02,
                        "eeg_O_theta": 1 + wwr / 100 + participant_index * 0.02, "eeg_O_alpha": 1 + wwr / 100 + participant_index * 0.02, "eeg_O_beta": 1 + wwr / 100 + participant_index * 0.02,
                })
    pre = tmp_path / "pre.csv"
    pd.DataFrame(rows).to_csv(pre, index=False)
    qc = pd.DataFrame({"participant_id": [f"P{i:02d}" for i in range(1, 5) for _ in range(6)], "scene_id": list(range(1, 7)) * 4, "excluded_from_analysis": False})
    qc_path = tmp_path / "qc.csv"
    qc.to_csv(qc_path, index=False)
    outputs = run_optimized_analysis(pre, qc_path, tmp_path / "optimized")
    audit = pd.read_json(outputs["adversarial_audit_json"])
    registry = pd.read_csv(outputs["data_registry"]).set_index("dataset")
    diagnostics = pd.read_csv(outputs["model_diagnostics"])
    assert audit.loc[0, "status"] == "fail"
    assert registry.loc["questionnaire_scene_all", "n_trials"] == 24
    assert registry.loc["eeg_scene_qc_passed", "n_trials"] == 24
    assert diagnostics["status"].eq("unstable_nonfinite_robust_covariance").any()
