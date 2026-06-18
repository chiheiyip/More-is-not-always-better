#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from more_is_not_always_better.discovery import (  # noqa: E402
    build_participants_from_roots,
    build_scene_manifest_from_eye_root,
    load_questionnaire_long_from_wjx,
    summarize_roots,
)
from paper_analysis.diagnostics.pipeline import run_diagnostics  # noqa: E402
from paper_analysis.eeg.contract import validate_eeg_scene_summary  # noqa: E402
from paper_analysis.eeg.pipeline import run_eeg_pipeline  # noqa: E402
from paper_analysis.eye_tracking.pipeline import run_eye_pipeline  # noqa: E402
from paper_analysis.figures import run_figure_pipeline  # noqa: E402
from paper_analysis.fusion.pipeline import run_fusion_pipeline  # noqa: E402
from paper_analysis.intake.pipeline import build_manifests  # noqa: E402
from paper_analysis.questionnaire.pipeline import run_questionnaire_pipeline  # noqa: E402
from paper_analysis.reporting.pipeline import build_paper_outputs  # noqa: E402
from paper_analysis.stats.models import run_statistical_models  # noqa: E402


DEFAULT_QUESTIONNAIRE = r"E:\26\补\VR+EEG实验问卷-补-原始数据-2026-06-14.xlsx"
DEFAULT_EYE_ROOT = r"E:\26\补\眼动数据"
DEFAULT_EEG_ROOT = r"E:\26\补\脑电数据"
DEFAULT_EEGLAB_ROOT = r"D:\Program Files\MATLAB\eeglab"
DEFAULT_OUTPUTS_ROOT = "outputs/realdata_full"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real raw-data pipeline from questionnaire, eye tracking, and EEG inputs.")
    parser.add_argument("--questionnaire_xlsx", default=DEFAULT_QUESTIONNAIRE)
    parser.add_argument("--eye_root", default=DEFAULT_EYE_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--eeglab_root", default=DEFAULT_EEGLAB_ROOT)
    parser.add_argument("--outputs_root", default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--participants", default="", help="Comma-separated participant IDs to include.")
    parser.add_argument("--max-participants", type=int, default=None, help="Limit to the first N trimodal participants for smoke runs.")
    parser.add_argument("--eye_alias_csv", default=None)
    parser.add_argument("--eeg_scene_csv", default=None, help="Use an existing scene-level EEG CSV instead of exporting from raw .set/.fdt files.")
    parser.add_argument("--skip-eeg-export", action="store_true", help="Stop before EEG/fusion unless --eeg_scene_csv is supplied.")
    parser.add_argument("--matlab_command", default="matlab")
    parser.add_argument("--expected-scenes-per-subject", type=int, default=12)
    parser.add_argument("--bin-size-ms", type=int, default=2000)
    parser.add_argument("--duration-tolerance-s", type=float, default=10.0)
    parser.add_argument("--model-config", default="configs/model_families.json")
    parser.add_argument("--eeg-qc-config", default="configs/eeg_qc.json")
    parser.add_argument("--figure-contracts", default="configs/figure_contracts.json")
    parser.add_argument("--reviewer-map", default="configs/reviewer_response_map.json")
    parser.add_argument("--skip-questionnaire-significance", action="store_true")
    parser.add_argument("--skip-questionnaire-reliability", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-reporting", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--cleanup-scratch", action="store_true", help="Remove intermediate raw-intake files after a successful run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run_realdata_all(args)
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


def run_realdata_all(args: argparse.Namespace) -> dict[str, Any]:
    outputs_root = Path(args.outputs_root)
    scratch_dir = outputs_root / "_raw_intake"
    _assert_output_not_inside_raw_inputs(outputs_root, [args.questionnaire_xlsx, args.eye_root, args.eeg_root])

    selected = _parse_participants(args.participants)
    planned_steps = _planned_steps(args)
    if args.dry_run:
        return {
            "status": "dry_run",
            "outputs_root": str(outputs_root),
            "scratch_dir": str(scratch_dir),
            "selected_participants": selected,
            "max_participants": args.max_participants,
            "steps": planned_steps,
            "raw_inputs_read_only": True,
        }

    outputs_root.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    preflight = summarize_roots(
        eye_root=args.eye_root,
        eeg_root=args.eeg_root,
        eye_alias_csv=args.eye_alias_csv,
        questionnaire_xlsx=args.questionnaire_xlsx,
    )
    participants = build_participants_from_roots(
        eye_root=args.eye_root,
        eeg_root=args.eeg_root,
        questionnaire_xlsx=args.questionnaire_xlsx,
        eye_alias_csv=args.eye_alias_csv,
    )
    participants = _select_participants(participants, selected=selected, max_participants=args.max_participants)
    participants_csv = scratch_dir / "participants_raw.csv"
    scene_csv = scratch_dir / "scene_manifest_raw.csv"
    questionnaire_long_csv = scratch_dir / "questionnaire_long_raw.csv"
    participants.to_csv(participants_csv, index=False, encoding="utf-8-sig")

    scene = build_scene_manifest_from_eye_root(
        args.eye_root,
        participants_csv,
        out_csv=scene_csv,
        eye_alias_csv=args.eye_alias_csv,
    )
    active_ids = _active_participant_ids(participants)
    questionnaire_long = load_questionnaire_long_from_wjx(args.questionnaire_xlsx, participants=active_ids)
    questionnaire_long.to_csv(questionnaire_long_csv, index=False, encoding="utf-8-sig")

    intake = build_manifests(participants_csv, scene_csv, outputs_root / "01_sample_qc")
    questionnaire = run_questionnaire_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        questionnaire_long=questionnaire_long_csv,
        outdir=outputs_root / "02_questionnaire",
        with_significance=not args.skip_questionnaire_significance,
        skip_reliability=args.skip_questionnaire_reliability,
    )
    eye = run_eye_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        outdir=outputs_root / "03_eye_tracking",
    )

    eeg_scene_csv = _prepare_eeg_scene_csv(args, outputs_root)
    if eeg_scene_csv is None:
        summary = _summary_base(args, outputs_root, preflight, participants, scene, questionnaire, eye)
        summary.update({"status": "partial_no_eeg_scene_csv", "fusion_run": False})
        _write_run_summary(summary, outputs_root)
        _cleanup_scratch(args, scratch_dir)
        return summary

    eeg_contract = validate_eeg_scene_summary(eeg_scene_csv)
    if eeg_contract["status"] == "error":
        raise SystemExit("EEG scene CSV failed contract validation: " + "; ".join(eeg_contract["errors"]))

    eeg = run_eeg_pipeline(
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        eeg_scene_csv=eeg_scene_csv,
        outdir=outputs_root / "04_eeg",
        eeg_qc_config=args.eeg_qc_config,
    )
    fusion = run_fusion_pipeline(
        questionnaire_long=questionnaire["questionnaire_long"],
        eye_aoi_trial_long=eye["eye_aoi_trial_long"],
        eeg_trial_long=eeg["eeg_trial_long"],
        participants_csv=intake["participants_standardized"],
        scene_manifest_csv=intake["scene_manifest_standardized"],
        outdir=outputs_root / "05_multimodal_fusion",
        expected_scenes_per_subject=args.expected_scenes_per_subject,
        bin_size_ms=args.bin_size_ms,
        duration_tolerance_s=args.duration_tolerance_s,
    )

    stats = None
    diagnostics = None
    reporting = None
    figures = None
    if not args.skip_models:
        stats = run_statistical_models(fusion["analysis_master_long"], args.model_config, outputs_root / "06_models")
    if not args.skip_diagnostics:
        diagnostics = run_diagnostics(fusion["analysis_master_long"], intake["participants_standardized"], outputs_root / "06_robustness")
    if not args.skip_reporting:
        if stats is None:
            raise SystemExit("Reporting requires model outputs. Remove --skip-models or add --skip-reporting.")
        reporting = build_paper_outputs(
            model_results_csv=stats["model_results"],
            diagnostics_dir=outputs_root / "06_robustness",
            reviewer_map=args.reviewer_map,
            outdir=outputs_root / "07_paper_tables",
        )
    if not args.skip_figures:
        figures = run_figure_pipeline(outputs_root=outputs_root, figure_contracts_config=args.figure_contracts, outdir=outputs_root / "10_figures")

    summary = _summary_base(args, outputs_root, preflight, participants, scene, questionnaire, eye)
    summary.update({
        "status": "complete",
        "eeg_scene_csv": str(eeg_scene_csv),
        "eeg_contract_status": eeg_contract["status"],
        "eeg_trial_rows": _row_count(eeg["eeg_trial_long"]),
        "fusion_run": True,
        "fusion_pre_qc_rows": _row_count(fusion["analysis_master_long_pre_qc"]),
        "fusion_kept_rows": _row_count(fusion["analysis_master_long"]),
        "models_run": stats is not None,
        "diagnostics_run": diagnostics is not None,
        "reporting_run": reporting is not None,
        "figures_run": figures is not None,
        "outputs": _stringify_outputs({
            "intake": intake,
            "questionnaire": questionnaire,
            "eye": eye,
            "eeg": eeg,
            "fusion": fusion,
            "stats": stats or {},
            "diagnostics": diagnostics or {},
            "reporting": reporting or {},
            "figures": figures or {},
        }),
    })
    _write_run_summary(summary, outputs_root)
    _cleanup_scratch(args, scratch_dir)
    return summary


def _prepare_eeg_scene_csv(args: argparse.Namespace, outputs_root: Path) -> Path | None:
    if args.eeg_scene_csv:
        return Path(args.eeg_scene_csv)
    if args.skip_eeg_export:
        return None
    eeg_outdir = outputs_root / "04_eeg_raw_export"
    script = Path(__file__).resolve().parent / "run_eeg_from_raw.py"
    cmd = [
        sys.executable,
        str(script),
        "--eeg_root",
        str(args.eeg_root),
        "--outdir",
        str(eeg_outdir),
        "--eeglab_root",
        str(args.eeglab_root),
        "--matlab_command",
        str(args.matlab_command),
    ]
    subprocess.run(cmd, check=True)
    return eeg_outdir / "summary" / "all_subjects_scene_level.csv"


def _select_participants(participants: pd.DataFrame, selected: list[str], max_participants: int | None) -> pd.DataFrame:
    out = participants.copy()
    complete_mask = (
        out.get("has_eeg_raw", False).fillna(False)
        & out.get("has_eye_raw", False).fillna(False)
        & out.get("has_questionnaire", False).fillna(False)
    )
    if selected:
        missing = sorted(set(selected) - set(out["participant_id"].astype(str)))
        if missing:
            raise SystemExit(f"Requested participants not found: {missing}")
        out = out.loc[out["participant_id"].astype(str).isin(selected)].copy()
    elif max_participants is not None:
        matched = out.loc[complete_mask, "participant_id"].astype(str).head(max_participants)
        out = out.loc[out["participant_id"].astype(str).isin(set(matched))].copy()
    else:
        out = out.loc[complete_mask].copy()
    if out.empty:
        raise SystemExit("No participants available after filtering.")
    required = ["has_eeg_raw", "has_eye_raw", "has_questionnaire"]
    missing_required = out.loc[~out[required].fillna(False).all(axis=1), ["participant_id"] + required]
    if not missing_required.empty:
        raise SystemExit("All selected participants must have questionnaire, eye, and EEG raw data. Missing rows: " + missing_required.to_dict("records").__repr__())
    out["exclude"] = False
    out["ExcludeReason"] = ""
    return out.reset_index(drop=True)


def _summary_base(
    args: argparse.Namespace,
    outputs_root: Path,
    preflight: dict[str, Any],
    participants: pd.DataFrame,
    scene: pd.DataFrame,
    questionnaire: dict[str, Path],
    eye: dict[str, Path],
) -> dict[str, Any]:
    return {
        "outputs_root": str(outputs_root),
        "participants": int(len(participants)),
        "participant_ids": participants["participant_id"].astype(str).tolist(),
        "scene_rows": int(len(scene)),
        "questionnaire_rows": _row_count(questionnaire["questionnaire_long"]),
        "eye_metric_rows": _row_count(eye["eye_aoi_trial_long"]),
        "preflight": preflight,
        "raw_inputs": {
            "questionnaire_xlsx": str(args.questionnaire_xlsx),
            "eye_root": str(args.eye_root),
            "eeg_root": str(args.eeg_root),
        },
        "raw_inputs_read_only": True,
    }


def _write_run_summary(summary: dict[str, Any], outputs_root: Path) -> None:
    summary_json = outputs_root / "realdata_run_summary.json"
    summary_csv = outputs_root / "realdata_run_summary.csv"
    safe = _json_safe(summary)
    summary_json.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in safe.items()}
    pd.DataFrame([flat]).to_csv(summary_csv, index=False, encoding="utf-8-sig")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _cleanup_scratch(args: argparse.Namespace, scratch_dir: Path) -> None:
    if args.cleanup_scratch and scratch_dir.exists():
        shutil.rmtree(scratch_dir)


def _parse_participants(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _active_participant_ids(participants: pd.DataFrame) -> list[str]:
    exclude = participants.get("exclude", pd.Series(False, index=participants.index)).astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return participants.loc[~exclude, "participant_id"].astype(str).tolist()


def _planned_steps(args: argparse.Namespace) -> list[str]:
    steps = ["preflight", "participants", "scene_manifest", "questionnaire", "eye"]
    if args.eeg_scene_csv:
        steps.extend(["validate_eeg_scene_csv", "eeg", "fusion"])
    elif args.skip_eeg_export:
        steps.append("stop_before_eeg")
    else:
        steps.extend(["matlab_eeg_export", "validate_eeg_scene_csv", "eeg", "fusion"])
    if not args.skip_models:
        steps.append("models")
    if not args.skip_diagnostics:
        steps.append("diagnostics")
    if not args.skip_reporting:
        steps.append("paper_outputs")
    if not args.skip_figures:
        steps.append("figures")
    return steps


def _row_count(path: str | Path) -> int:
    try:
        return int(pd.read_csv(path, encoding="utf-8-sig").shape[0])
    except pd.errors.EmptyDataError:
        return 0


def _stringify_outputs(groups: dict[str, dict[str, Path]]) -> dict[str, dict[str, str]]:
    return {group: {name: str(path) for name, path in outputs.items()} for group, outputs in groups.items()}


def _assert_output_not_inside_raw_inputs(outputs_root: Path, raw_inputs: list[str | Path]) -> None:
    out = _resolved(outputs_root)
    raw_dirs = []
    for value in raw_inputs:
        path = Path(value)
        if path.suffix == "":
            raw_dirs.append(_resolved(path))
    for raw in raw_dirs:
        if out == raw or raw in out.parents:
            raise SystemExit(f"outputs_root must not be inside a raw input directory: {outputs_root}")


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


if __name__ == "__main__":
    main()
