"""Revised exact-joint sensitivity and actual bootstrap repeat verification."""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .fresh import dump, git_sha, unique
from .hf_qc_revision import write_csv, TRIAL_KEYS
from .r_runner import invoke_r
from .reproducibility import compare_tables, seal_files, verify_seal
from .runtime_lock import validate_r
from .state import StageBlockedError, file_sha256
from paper_analysis.utils.io import is_truthy


def eye_common_sensitivity(config, source, run, repo):
    out = run / "eye_common_sensitivity"
    original = source / "02_eye_stage2/eye_stage2_model_input.csv"
    data = pd.read_csv(original, encoding="utf-8-sig")
    unique(data, ["Participant", "GlobalTrialOrder"])
    qc = pd.read_csv(config["eeg"]["trial_file"], encoding="utf-8-sig")
    keys = qc.loc[~qc.bad_eeg_quality.map(is_truthy) & ~qc.eeg_subject_quality_exclusion.map(is_truthy), TRIAL_KEYS]
    keys = keys.rename(columns={"participant_id": "Participant", "scene_id": "GlobalTrialOrder"})
    common = data.loc[data.IncludedPrimary.map(is_truthy)].merge(keys, on=["Participant", "GlobalTrialOrder"], validate="one_to_one")
    common["IncludeEEGTrialValid"] = True
    write_csv(common, out / "EEG_valid_common_sample_model_input.csv")
    write_csv(pd.DataFrame([{"Participants": common.Participant.nunique(), "Trials": len(common)}]), out / "exact_common_sample_counts.csv")
    for folder in [out / "reference", out / "repeat"]:
        invoke_r(config["rscript"], repo / "analysis/r/eye_eeg_common_sensitivity.R",
                 [str(out / "EEG_valid_common_sample_model_input.csv"), str(folder)], required=True)
    name = "16_EEG_valid_common_sample_sensitivity.csv"
    old = pd.read_csv(out / "reference" / name)
    new = pd.read_csv(out / "repeat" / name)
    comparison = compare_tables(old, new, ["outcome", "model", "term"], exact_numeric=True)
    write_csv(comparison, out / "repeat_comparison.csv")
    if not comparison.passed.all():
        raise StageBlockedError("Eye EEG-common sensitivity does not repeat")
    shutil.copyfile(out / "reference" / name, out / name)
    dump(out / "reuse_provenance.json", {"eye_main": "unchanged; independently verified source reused", "eye_source": str(original),
        "eye_source_sha256": file_sha256(original), "EEG_QC": config["eeg"]["trial_file"],
        "EEG_QC_sha256": file_sha256(config["eeg"]["trial_file"]), "sensitivity": "new fits, repeated independently"})


def bootstrap_repeat(config, run, repo):
    """Actually refit all 80,000 draws; draw replay alone is insufficient."""
    out = run / "bootstrap_repeat"
    out.mkdir(parents=True)
    os.environ.pop("PAPER_ANALYSIS_R_REUSE_ROOT", None)
    scripts = seal_files([repo / "analysis/r" / name for name in
                          ["eeg_primary_analysis.R", "common.R", "runtime_profile.R", "runtime-versions.lock.json"]])
    environment = validate_r(config["rscript"], repo=repo, profile="primary")
    def job(trim):
        source = run / ("06_eeg_primary" if trim == 10 else f"07_eeg_window_robustness/trim_{trim}s")
        input_path = source / ("eeg_primary_model_input.csv" if trim == 10 else "input.csv")
        names = ["07_eeg_cluster_bootstrap_5000.csv", "08_eeg_bootstrap_failures.csv", "bootstrap_draw_provenance.json"]
        inputs = seal_files([input_path, *[source / n for n in names]])
        dest = out / f"trim_{trim}s"
        dest.mkdir()
        shutil.copyfile(input_path, dest / "input.csv")
        eeg = config["eeg"]
        arguments = [str(dest / "input.csv"), str(dest), ",".join(f"{m}_relative" for m in eeg["core_metrics"]),
            ",".join(f"log10_{m}_absolute" for m in eeg["core_metrics"]), "5000",
            ",".join(f"{m}_relative" for m in eeg["secondary_metrics"]),
            ",".join(f"{m}_relative" for m in eeg["supplemental_metrics"])]
        print(f"BOOTSTRAP REPEAT START {trim}s: 20000 actual fits", flush=True)
        invoke_r(config["rscript"], repo / "analysis/r/eeg_primary_analysis.R", arguments, required=True)
        verify_seal(inputs)
        for name in names:
            if file_sha256(source / name) != file_sha256(dest / name):
                raise StageBlockedError(f"Bootstrap repeat bytes differ: {trim}s {name}")
        table = pd.read_csv(dest / names[0])
        all_successful = bool(table.successful.eq(5000).all() & table.failed_replicates.eq(0).all())
        if not all_successful:
            raise StageBlockedError("Bootstrap has failed draws; retain diagnostics, do not label complete reproduction")
        record = {"status": "passed", "trim_s": trim, "fits_requested": 20000, "all_successful_5000": all_successful,
            "max_absolute_difference": 0, "reference": str(source), "repeat": str(dest), "input_seal": inputs,
            "outputs": seal_files([dest / n for n in names])}
        dump(dest / "job.json", record)
        print(f"BOOTSTRAP REPEAT PASS {trim}s", flush=True)
        return record
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = list(executor.map(job, [0, 5, 10, 15]))
    verify_seal(scripts)
    dump(out / "repeat_execution.json", {"status": "passed", "code_sha": git_sha(repo), "environment": environment,
        "scripts": scripts, "jobs": jobs, "fits_requested": 80000})
