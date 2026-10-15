"""Create source-backed writing summaries without refitting any model.

The copied eye metrics retain every original CSV field string except the
explicit EEG-eligibility link, which is rebuilt from the revised model keys.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_analysis.teacher.fresh import dump, git_sha, unique
from paper_analysis.teacher.hf_qc_revision import read_json, write_csv
from paper_analysis.teacher.state import StageBlockedError, file_sha256


def revised_eye_link(source, target, eligible):
    data = source.read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    header = rows[0]
    participant, trial, flag = [header.index(c) for c in ["Participant", "GlobalTrialOrder", "IncludeEEGTrialValid"]]
    seen = set()
    for row in rows[1:]:
        key = (row[participant], int(row[trial]))
        if key in seen: raise StageBlockedError("Duplicate eye trial key")
        seen.add(key)
        row[flag] = "True" if key in eligible else "False"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream, lineterminator="\r\n" if b"\r\n" in data else "\n").writerows(rows)
    before = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    after = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig"))))
    for a, b in zip(before, after):
        if [v for i, v in enumerate(a) if i != flag] != [v for i, v in enumerate(b) if i != flag]:
            raise StageBlockedError("An eye measurement field changed during link update")
    return {"source_sha256": file_sha256(source), "output_sha256": file_sha256(target),
        "only_changed_field": "IncludeEEGTrialValid", "other_field_strings_exactly_equal": True}


def prepare(config, run):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise StageBlockedError("Writing-source preparation requires committed code")
    source = Path(config["hf_qc_revision"]["source_run"])
    out = run / "writing_sources"
    out.mkdir(exist_ok=False)
    models = pd.read_csv(run / "audit_models/NoHF_1_45/trim_0s/input.csv", float_precision="round_trip")
    unique(models, ["Participant", "GlobalTrialOrder"])
    eligible = set(models[["Participant", "GlobalTrialOrder"]].itertuples(index=False, name=None))
    eye_input = source / "02_eye_stage2/eye_stage2_model_input.csv"
    proof = revised_eye_link(eye_input, out / "eye_trial_metrics_revised_EEG_eligibility.csv", eligible)
    eye = pd.read_csv(out / "eye_trial_metrics_revised_EEG_eligibility.csv", encoding="utf-8-sig")
    primary = eye.loc[eye.IncludedPrimary.eq(True)]
    common = primary.loc[primary.IncludeEEGTrialValid.eq(True)]
    pd.testing.assert_frame_equal(common[["Participant", "GlobalTrialOrder"]].sort_values(["Participant", "GlobalTrialOrder"]).reset_index(drop=True),
        pd.read_csv(run / "eye_common_sensitivity/EEG_valid_common_sample_model_input.csv")[["Participant", "GlobalTrialOrder"]].sort_values(["Participant", "GlobalTrialOrder"]).reset_index(drop=True))
    scene = pd.read_csv(run / "06_eeg_primary/eeg_crossmodal_model_input.csv")
    unique(scene, ["Participant", "GlobalTrialOrder"])
    if set(scene[["Participant", "GlobalTrialOrder"]].itertuples(index=False, name=None)) != set(common[["Participant", "GlobalTrialOrder"]].itertuples(index=False, name=None)):
        raise StageBlockedError("Scene joint and eye common trial identities differ")
    sync = pd.read_excel(run / "08_synchronized_crossmodal/synchronized_crossmodal_sample_and_source.xlsx").iloc[0]
    write_csv(pd.DataFrame([
        {"sample": "eye_main", "participants": primary.Participant.nunique(), "trials": len(primary), "status": "verified_reuse"},
        {"sample": "eeg_common", "participants": models.Participant.nunique(), "trials": len(models), "status": "revised_NoHF"},
        {"sample": "exact_joint", "participants": common.Participant.nunique(), "trials": len(common), "status": "rebuilt_from_exact_keys"},
        {"sample": "synchronized", "participants": int(sync.Participants), "trials": int(sync.Trials), "status": "verified_clock_and_revised_QC"}
    ]), out / "publication_sample_facts.csv")
    temporal = pd.read_csv(run / "audit_models/NoHF_1_45/family_coefficients.csv")
    rows = []
    for family in ["temporal_relative", "temporal_absolute"]:
        scale = "relative" if family.endswith("relative") else "absolute"
        for band in ["theta", "alpha", "beta"]:
            for term in ["Block", "PositionWithinBlockCentered"]:
                selected = temporal.loc[temporal.family_id.eq(family) & temporal.outcome.str.endswith(f"{band}_{scale}") & temporal.term.eq(term)]
                if len(selected) != 12: raise StageBlockedError("Incomplete writing temporal summary")
                rows.append({"family_id": family, "band": band, "term": term, "tests": len(selected),
                    "detected_joint": int(selected.joint_q.lt(.05).sum()), "estimate_min": selected.estimate.min(),
                    "estimate_max": selected.estimate.max(), "joint_q_min": selected.joint_q.min(), "joint_q_max": selected.joint_q.max()})
    write_csv(pd.DataFrame(rows), out / "temporal_writing_summary.csv")
    powers = pd.read_csv(run / "10_eeg_denominator_sensitivity/paired_psd_integrals.csv")
    write_csv(pd.DataFrame([{"metric": name, "records": len(powers), "mean": powers[name].mean(),
        "median": powers[name].median(), "min": powers[name].min(), "max": powers[name].max()}
        for name in ["percent_40_45", "denominator_reduction_percent", "relative_increase_percent"]]), out / "denominator_writing_summary.csv")
    keys = ["onset_trim_s", "outcome", "term"]
    old_sync = pd.read_csv(source / "08_synchronized_crossmodal/timebin_model_results.csv")
    new_sync = pd.read_csv(run / "08_synchronized_crossmodal/timebin_model_results.csv")
    unique(old_sync, keys); unique(new_sync, keys)
    comparison = old_sync.merge(new_sync, on=keys, how="outer", validate="one_to_one", suffixes=("_original", "_revised"), indicator=True)
    if not comparison._merge.eq("both").all(): raise StageBlockedError("Synchronized test family identities differ")
    valid = comparison.p_fdr_bh_parallel_original.notna() & comparison.p_fdr_bh_parallel_revised.notna()
    comparison["parallel_classification_flip"] = (comparison.p_fdr_bh_parallel_original.lt(.05) != comparison.p_fdr_bh_parallel_revised.lt(.05)).where(valid)
    write_csv(comparison.drop(columns="_merge"), out / "synchronized_Original_vs_NoHF.csv")
    # The existing six primary models and their registered inferences are reused.
    main = pd.read_csv(source / "02_eye_stage2/12_CR2_robust_results.csv")
    selected = main.loc[main["p.value.BH"].lt(.05) & main.term.ne("(Intercept)")]
    sensitivity = pd.read_csv(run / "eye_common_sensitivity/16_EEG_valid_common_sample_sensitivity.csv")
    direction = selected[["outcome", "term", "estimate"]].merge(sensitivity[["outcome", "term", "estimate"]],
        on=["outcome", "term"], how="left", validate="one_to_one", suffixes=("_main", "_revised_common"))
    direction["same_direction"] = (direction.estimate_main * direction.estimate_revised_common).gt(0)
    write_csv(direction, out / "eye_primary_findings_revised_common_direction.csv")
    dump(out / "source_provenance.json", {"code_sha": git_sha(REPO), "scientific_calculation_sha": read_json(run / "run_manifest.json")["code_sha"],
        "new_model_fits": 0, "eye_link": proof, "eligible_model_input_sha256": file_sha256(run / "audit_models/NoHF_1_45/trim_0s/input.csv"),
        "note": "Writing summaries and EEG linkage only; original eye measurements and primary QC are preserved"})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(read_json(args.config), args.run_root))
