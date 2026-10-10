import pandas as pd
import pytest
from paper_analysis.teacher.historical_eye import compare_frames
from paper_analysis.teacher.state import StageBlockedError


def test_pair_by_keys_keep_missing_and_investigate_border_flip():
    before = pd.DataFrame({"outcome": ["a", "b"], "term": ["t", "t"],
                           "estimate": [1.0, -.1], "p.value.BH": [.0500001, None]})
    after = pd.DataFrame({"outcome": ["b", "a"], "term": ["t", "t"],
                          "estimate": [-.1, 1.0 + 1e-10], "p.value.BH": [None, .0499999]})
    rows = compare_frames(before, after, ["outcome", "term"], "model")
    p = next(r for r in rows if r["field"] == "p.value.BH")
    assert p["beyond_tolerance"] == 0
    assert p["significance_flips"] == 1
    assert p["missing_mismatches"] == 0
    after.loc[0, "p.value.BH"] = .2
    assert next(r for r in compare_frames(before, after, ["outcome", "term"], "model")
                if r["field"] == "p.value.BH")["missing_mismatches"] == 1


def test_duplicate_key_or_removed_family_member_rejected():
    data = pd.DataFrame({"outcome": ["a", "b"], "value": [1, 2]})
    with pytest.raises(StageBlockedError, match="sample/term keys"):
        compare_frames(data, data.iloc[:1], ["outcome"], "model")
    with pytest.raises(StageBlockedError, match="duplicate"):
        compare_frames(data, pd.concat([data, data.iloc[:1]]), ["outcome"], "model")


def test_historical_r_platform_is_not_a_package_version():
    from paper_analysis.teacher.historical_eye import historical_r_packages
    text = "Platform: x86_64-w64-mingw32/x64\r\r\nloaded via a namespace (and not attached):\r\r\n[1] lme4_2.0-6 broom.mixed_0.2.9.7"
    assert historical_r_packages(text) == [("lme4", "2.0-6"), ("broom.mixed", "0.2.9.7")]


def test_equal_infinite_df_remains_equal_and_maximum_is_finite():
    data = pd.DataFrame({"outcome": ["a", "b"], "df": [float("inf"), 10.0]})
    assert compare_frames(data, data.copy(), ["outcome"], "model")[0]["max_abs_difference"] == 0


def test_empty_C0_labels_require_scene_proof_and_keep_block_starts_missing():
    from paper_analysis.teacher.historical_eye import factor_encoding_check
    old = pd.DataFrame({"Participant": ["a"]*3, "GlobalTrialOrder": [1,2,3], "Block": [1,1,2],
                        "SceneID": [1,2,3], "AOIImageID": ["1-C0W15","1-C1W15","2-C0W15"], "Complexity": [None,"C1",None],
                        "PreviousWWR": [None,15,None], "PreviousComplexity": [None,None,None]})
    new = old.copy(); new["Complexity"] = ["C0","C1","C0"]; new["PreviousComplexity"] = [None,"C0",None]
    checks = factor_encoding_check(old,new)
    assert [c["canonical_mismatches"] for c in checks] == [0,0]
    assert [c["historical_empty_baseline_labels"] for c in checks] == [2,1]
    new.loc[2,"PreviousComplexity"] = "C0"
    assert factor_encoding_check(old,new)[1]["canonical_mismatches"] == 1


def test_statistics_refit_never_copies_inside_source(tmp_path):
    from paper_analysis.teacher.historical_eye import refit_statistics
    with pytest.raises(StageBlockedError, match="new sibling"):
        refit_statistics(tmp_path, tmp_path / "recursive", tmp_path)
