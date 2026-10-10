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
