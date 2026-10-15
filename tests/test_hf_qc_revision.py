import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_analysis.teacher.hf_qc_revision import (
    HF, PRE_FLAG, PRE_REASONS, common_membership, impact_tables,
    qc_profiles, reconstruct_pair, methodological_gate, build_audit_qc,
)
from paper_analysis.teacher.state import StageBlockedError


def policy():
    return {"policy": "robust", "robust_metrics": [HF, "rms_mean_uV", "peak_to_peak_uV"],
            "robust_k": 3.5, "robust_min_n": 4, "min_segment_duration_s": 1,
            "nan_fraction_threshold": .2, "flat_fraction_threshold": .2,
            "bad_scene_fraction_threshold": .3}


def test_shipped_formal_profile_removes_only_hf_and_retains_historical_profile():
    import json
    repo = Path(__file__).resolve().parents[1]
    original = json.loads((repo / "configs/eeg_qc.json").read_text(encoding="utf-8"))
    revised = json.loads((repo / "configs/eeg_qc_no_hf.json").read_text(encoding="utf-8"))
    assert revised == qc_profiles(original)["NoHF"]
    assert HF in original["robust_metrics"]


def features():
    rows = []
    for trim in [0, 5, 10, 15]:
        for participant in ["甲", "乙", "丙", "丁"]:
            for trial in range(1, 11):
                # All medians/MADs are deliberately zero: strict > matters.
                rows.append({"participant_id": participant, "scene_id": trial, "onset_trim_s": trim,
                    HF: 0., "rms_mean_uV": 0., "peak_to_peak_uV": 0., "analysis_dur_s": 30.,
                    "segment_valid_duration": True, "nan_fraction": 0., "flat_fraction": 0., "near_boundary": 0})
    return pd.DataFrame(rows)


def test_removing_only_hf_keeps_thresholds_and_separates_participant_propagation():
    data = features()
    # A: four HF-only -> participant excluded. B: three HF-only -> exactly30%,
    # only those trials excluded. C: HF and RMS overlap -> both exclude.
    data.loc[data.participant_id.eq("甲") & data.scene_id.le(4), HF] = 1
    data.loc[data.participant_id.eq("乙") & data.scene_id.le(3), HF] = 1
    c = data.participant_id.eq("丙") & data.scene_id.le(4)
    data.loc[c, [HF, "rms_mean_uV"]] = 1
    tables, thresholds = reconstruct_pair(data, policy())
    a = tables["Original"].loc[lambda d: d.participant_id.eq("甲") & d.onset_trim_s.eq(0)]
    assert a.bad_eeg_quality.sum() == 10 and a[PRE_FLAG].sum() == 4
    assert a.loc[a.scene_id.gt(4), PRE_REASONS].eq("").all()
    b = tables["Original"].loc[lambda d: d.participant_id.eq("乙")]
    assert not b.eeg_subject_quality_exclusion.any()
    impact = impact_tables(tables)
    overlap = impact["HF_QC_flag_overlap_by_variant"]
    assert overlap.n_hf_only.eq(7).all() and overlap.n_hf_overlap.eq(4).all()
    people = impact["HF_QC_participant_impact"]
    assert people.loc[people.participant_id.eq("甲"), "category"].eq("A_original_excluded_nohf_retained").all()
    assert people.loc[people.participant_id.eq("丙"), "category"].eq("B_both_excluded").all()
    assert not tables["NoHF"].loc[lambda d: d.participant_id.isin(["甲", "乙"]), "bad_eeg_quality"].any()
    assert len(thresholds) == 4 * (3 + 2)


def test_duration_nan_flat_remain_active_boundary_remains_record_only():
    data = features()
    data.loc[data.scene_id.eq(1), "analysis_dur_s"] = .5
    data.loc[data.scene_id.eq(2), "nan_fraction"] = .21
    data.loc[data.scene_id.eq(3), "flat_fraction"] = .21
    data.loc[data.scene_id.eq(4), "near_boundary"] = 1
    result, _ = reconstruct_pair(data, policy())
    for d in result.values():
        assert d.loc[d.scene_id.le(3), PRE_FLAG].all()
        assert not d.loc[d.scene_id.eq(4), PRE_FLAG].any()
    overlap = impact_tables(result)["HF_QC_flag_overlap_by_variant"]
    assert overlap.n_boundary_flag.eq(0).all()
    assert overlap.n_boundary_status_recorded.eq(4).all()


@pytest.mark.parametrize("change", ["duplicate", "missing_window", "missing_trial_variant"])
def test_invalid_trial_grids_fail(change):
    data = features()
    if change == "duplicate":
        data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    elif change == "missing_window":
        data = data.loc[data.onset_trim_s.ne(15)]
    else:
        data = data.iloc[1:]
    with pytest.raises(StageBlockedError):
        reconstruct_pair(data, policy())


def test_original_policy_is_not_mutated_or_silently_assumed():
    original = policy()
    before = copy.deepcopy(original)
    actual = qc_profiles(original)
    assert original == before
    actual["NoHF"]["robust_metrics"].append("anything")
    assert original == before
    with pytest.raises(StageBlockedError):
        qc_profiles({"policy": "legacy_0_4"})


def test_full_qc_stage_does_not_inherit_the_old_common_membership_column(tmp_path):
    from paper_analysis.teacher.fresh import dump
    source = tmp_path / "source"
    data = features()
    data.loc[data.participant_id.eq("甲") & data.scene_id.le(4), HF] = 1
    original, _ = reconstruct_pair(data, policy())
    folder = source / "00_eeg_trial_qc"
    folder.mkdir(parents=True)
    original["Original"].to_csv(folder / "eeg_onset_sensitivity_trial_long.csv", index=False)
    dump(folder / "QC_policy.json", policy())
    run = tmp_path / "run"
    build_audit_qc({}, source, run)
    revised = pd.read_csv(run / "qc/NoHF/eeg_onset_sensitivity_trial_long.csv")
    assert revised.CommonQCIncluded.all()
    verified = __import__("json").loads((run / "qc/verification.json").read_text())
    assert verified["independent_QC_match"]


@pytest.mark.parametrize("mean_rms,independent_pass,expected", [(1e-7, True, True), (1., True, False), (1e-7, False, False)])
def test_formal_gate_uses_methodology_not_p_values(tmp_path, monkeypatch, mean_rms, independent_pass, expected):
    from paper_analysis.teacher.fresh import dump
    monkeypatch.setattr("paper_analysis.teacher.hf_qc_revision.git_sha", lambda _: "committed")
    proof = tmp_path / "proof"
    dump(proof / "audit_completion.json", {"numeric_target_matched": True})
    dump(proof / "mean_waveform_scale_check.json", {"mean": mean_rms, "channels": 8.})
    dump(tmp_path / "qc/verification.json", {"independent_QC_match": independent_pass, "remaining_thresholds_exactly_equal": True})
    dump(tmp_path / "qc/Original/QC_policy.json", policy())
    dump(tmp_path / "qc/NoHF/QC_policy.json", qc_profiles(policy())["NoHF"])
    config = {"hf_qc_revision": {"hf_formula_proof": str(proof), "cancellation_evidence": {
        "formula_verified_field": "numeric_target_matched", "mean_waveform_rms_field": "mean", "mean_channel_rms_field": "channels"}}}
    decision = methodological_gate(config, tmp_path, tmp_path)
    assert decision["adopt_NoHF"] is expected
    assert decision["selection_uses_significance"] is False


def test_psd_reuse_preserves_historical_key_and_requires_exact_actual_environment():
    from paper_analysis.teacher.hf_qc_formal import verify_cache_implementation
    from paper_analysis.teacher.eeg_denominator import cache_identity
    epochs = pd.DataFrame([{"Participant": "甲", "GlobalTrialOrder": 1, "onset_trim_s": 0,
                            "view_start_s": 1., "view_end_s": 61.}])
    methods = {"implementation.m": "exact-code-hash", "native_runtime": "current-lock-document-hash"}
    historical = {"lock_sha256": "historical-lock-document-hash", "snapshot": {"version": "fixed", "function": "same-sha", "threads": 10}}
    current = {"lock_sha256": "current-lock-document-hash", "snapshot": copy.deepcopy(historical["snapshot"])}
    sources = [{"participant": "甲", "set_sha256": "a", "fdt_sha256": "b"}]
    key = cache_identity(sources, epochs, dict(methods, native_runtime=historical["lock_sha256"]))
    manifest = {"sources": sources, "cache_key": key}
    proof = verify_cache_implementation(manifest, epochs, methods, current, historical)
    assert proof["snapshots_exactly_equal"] and proof["original_cache_identity_verified"]
    current["snapshot"]["function"] = "changed"
    with pytest.raises(StageBlockedError, match="runtime snapshots differ"):
        verify_cache_implementation(manifest, epochs, methods, current, historical)
    with pytest.raises(StageBlockedError, match="implementation/epoch"):
        verify_cache_implementation(manifest, epochs, dict(methods, **{"implementation.m": "changed"}), historical, historical)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / ".r-env/Lib/R/bin/x64/Rscript.exe").is_file(),
                    reason="Registered primary R environment not installed")
def test_real_r_revised_exact_eye_common_sample_repeats(tmp_path):
    from paper_analysis.teacher.hf_qc_joint import eye_common_sensitivity
    repo = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(1015)
    rows = []
    outcomes = ["TableShare", "WindowShare", "RawCompetition", "LogTableEnrichment", "LogWindowEnrichment", "AdjustedCompetition"]
    for i in range(36):
        intercept = rng.normal(size=6)
        for trial in range(1, 13):
            row = {"Participant": f"P{i:02}", "GlobalTrialOrder": trial, "IncludedPrimary": True,
                "WWR": [15, 45, 75][(trial-1)%3], "Complexity": ["C0", "C1"][(trial-1)//3%2],
                "Gender": ["Female", "Male"][i%2], "OrderGroup": ["new order2", "order1", "order2"][i%3],
                "ExperienceGroup": ["High", "Low"][(i//2)%2], "Block": (trial-1)//6+1,
                "PositionWithinBlockCentered": (trial-1)%6-2.5}
            for j, name in enumerate(outcomes):
                row[name] = intercept[j] + rng.normal(scale=.1)
            rows.append(row)
    source = tmp_path / "source"
    path = source / "02_eye_stage2/eye_stage2_model_input.csv"
    path.parent.mkdir(parents=True)
    data = pd.DataFrame(rows)
    data.to_csv(path, index=False)
    qc = data[["Participant", "GlobalTrialOrder"]].rename(columns={"Participant": "participant_id", "GlobalTrialOrder": "scene_id"})
    qc["bad_eeg_quality"] = False
    qc["eeg_subject_quality_exclusion"] = False
    qc.loc[0, "bad_eeg_quality"] = True
    qc.to_csv(tmp_path / "qc.csv", index=False)
    config = {"rscript": str(repo / "scripts/portable_rscript.cmd"), "eeg": {"trial_file": str(tmp_path / "qc.csv")}}
    run = tmp_path / "run"
    eye_common_sensitivity(config, source, run, repo)
    counts = pd.read_csv(run / "eye_common_sensitivity/exact_common_sample_counts.csv")
    assert counts.Trials.iloc[0] == 431
    result = pd.read_csv(run / "eye_common_sensitivity/repeat_comparison.csv")
    assert result.passed.all()
