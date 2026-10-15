"""Verify revised EEG, retained eye main results and rebuilt joint results."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_analysis.teacher.fresh import dump, git_sha
from paper_analysis.teacher.hf_qc_revision import read_json, write_csv
from paper_analysis.teacher.reproducibility import compare_tables, seal_files, verify_seal, verify_complete_bootstrap_repeat
from paper_analysis.teacher.runtime_lock import validate_analysis
from paper_analysis.teacher.r_runner import invoke_r
from paper_analysis.teacher.state import StageBlockedError, file_sha256


def verify(config, run):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise StageBlockedError("Verification must use clean committed code")
    manifest = read_json(run / "run_manifest.json")
    if manifest["status"] != "analysis_complete_pending_delivery":
        raise StageBlockedError("Revised analysis must finish before complete verification")
    if not read_json(run / "methodological_gate.json")["adopt_NoHF"]:
        raise StageBlockedError("No-HF methodological gate failed")
    output = run / "complete_verification"
    if output.exists():
        raise StageBlockedError("Use a new verification output, preserving prior evidence")
    output.mkdir()
    environment = validate_analysis(config, REPO, require_matlab=True)
    bootstrap = verify_complete_bootstrap_repeat(run / "bootstrap_repeat/repeat_execution.json")
    checks = []
    def compare(name, left, right, keys, *, model=True):
        a = pd.read_csv(left, float_precision="round_trip")
        b = pd.read_csv(right, float_precision="round_trip")
        result = compare_tables(a, b, keys, model=model, exact_numeric=True)
        write_csv(result, output / f"{name}_comparison.csv")
        checks.append({"name": name, "passed": bool(result.passed.all()), "rows": len(a),
                       "inputs": seal_files([left, right])})
        if not result.passed.all():
            raise StageBlockedError("Revised repetition differs: " + name)
    # Retain the existing main-model repeated-process evidence.
    prior = read_json(run / "numerical_verification/summary.json")
    if not prior["passed"]:
        raise StageBlockedError("Main EEG repeated-process checks failed")
    source = Path(config["hf_qc_revision"]["source_run"])
    old_proof = read_json(source / "reproducibility_verification.json")
    if old_proof["status"] != "passed" or old_proof["scopes"]["eye"]["status"] != "passed":
        raise StageBlockedError("Eye main-result reuse proof failed")
    verify_seal(old_proof["inputs"])
    verify_seal(old_proof["evidence"])
    eye_files = [p for folder in ["01_eye_stage1", "02_eye_stage2", "04_eye_stage3"]
                 for p in (source / folder).glob("*.csv")]
    eye_seal = seal_files(eye_files)
    if not eye_seal:
        raise StageBlockedError("No registered eye outputs to reuse")
    # Repeat relative 1–40 models; absolute outcomes are identical to the already
    # independently repeated revised 1–45 inputs/results and are explicitly reused.
    from paper_analysis.teacher.eeg_denominator import run_models
    c = read_json(run / "resolved_config.json")
    main_dir = run / "10_eeg_denominator_sensitivity/main_1_45"
    main = {p.stem: pd.read_csv(p) for p in main_dir.glob("*.csv") if p.stem != "outcomes"}
    sensitivity = run / "10_eeg_denominator_sensitivity/sensitivity_1_40"
    frames = {trim: pd.read_csv(sensitivity / f"trim_{trim}s/input.csv", float_precision="round_trip") for trim in [0, 5, 10, 15]}
    run_models(frames, c, REPO, output / "sensitivity_repeat", run / "audit_models/analysis_contract.json", reuse_absolute=main)
    for name in ["family_coefficients", "family_factors"]:
        compare("sensitivity_"+name, sensitivity / f"{name}.csv", output / f"sensitivity_repeat/{name}.csv",
                ["onset_trim_s", "outcome", "model", "term", "family_id"])
    # Independent FDT -> ROI waveform -> SciPy Welch checks for the extra cache.
    psd = read_json(run / "10_eeg_denominator_sensitivity/PSD_provenance.json")
    gap_cache = Path(psd["component_caches"][-1])
    gap = output / "waveform_check"
    (gap / "10_eeg_denominator_sensitivity").mkdir(parents=True)
    dump(gap / "resolved_config.json", c)
    dump(gap / "run_manifest.json", {"git_commit": manifest["code_sha"]})
    dump(gap / "10_eeg_denominator_sensitivity/PSD_provenance.json", {"cache_dir": str(gap_cache),
        "sources": read_json(gap_cache / "cache_manifest.json")["sources"]})
    spec = importlib.util.spec_from_file_location("waveform_verifier", REPO / "scripts/verify_fresh_waveform_psd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(gap, sample_count=12)
    # Fresh scene-level joint fits from the exact revised intersection.
    scene = output / "scene_joint_repeat"
    invoke_r(c["rscript"], REPO / "analysis/r/eeg_crossmodal_analysis.R", [
        str(run / "06_eeg_primary/eeg_crossmodal_model_input.csv"), str(scene), "O_theta_relative,O_alpha_relative"], required=True)
    compare("scene_joint", run / "06_eeg_primary/19b_eeg_crossmodal_CR2.csv", scene / "19b_eeg_crossmodal_CR2.csv",
            ["outcome", "model", "term", "EyePredictor"])
    # Fresh time-bin joint fits, with all eligibility rules applied again.
    from paper_analysis.teacher.complete import _build_sync_outputs
    sync = output / "sync_joint_repeat"
    sync.mkdir()
    (sync / "02_eye_stage2").mkdir()
    shutil.copyfile(source / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx", sync / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx")
    _build_sync_outputs(Path(config["outputs_root"]), sync, c)
    compare("synchronized_joint", run / "08_synchronized_crossmodal/timebin_model_results.csv",
            sync / "08_synchronized_crossmodal/timebin_model_results.csv", ["onset_trim_s", "outcome", "term"])
    # Common-eye sensitivity was also refitted twice by the production revision.
    eye_repeat = pd.read_csv(run / "eye_common_sensitivity/repeat_comparison.csv")
    if not eye_repeat.passed.all():
        raise StageBlockedError("Eye common sensitivity repeat failed")
    verify_seal(eye_seal)
    result = {"status": "passed", "verification_sha": git_sha(REPO), "calculation_sha": manifest["code_sha"],
        "environment": environment, "comparison_mode": "exact_numeric", "checks": checks,
        "scopes": {"eye": {"status": "passed", "mode": "verified unchanged main outputs reused; revised common sensitivity actually refitted twice"},
                   "eeg": {"status": "passed", "mode": "revised main and sensitivity model repetition, full integration verification, new waveform spot checks and actual bootstrap repeats"},
                   "joint": {"status": "passed", "mode": "actual repeated scene and synchronized fits"}},
        "eye_reuse": eye_seal, "full_bootstrap_repeat": bootstrap,
        "evidence": seal_files([p for p in output.rglob("*") if p.is_file()])}
    dump(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(read_json(args.config), args.run_root), ensure_ascii=False))
