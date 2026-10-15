import csv
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("hf_writing", Path(__file__).parents[1] / "scripts/prepare_hf_qc_writing_sources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_revised_eye_link_preserves_lexical_measurements(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(('\ufeffParticipant,GlobalTrialOrder,IncludeEEGTrialValid,value,note\r\n'
                       '甲,1,False,0.10000000000000001,"comma, and text"\r\n'
                       '乙,2,True,,missing\r\n').encode("utf-8"))
    target = tmp_path / "derived.csv"
    proof = module.revised_eye_link(source, target, {("甲", 1)})
    rows = list(csv.reader(target.open(encoding="utf-8-sig", newline="")))
    assert rows[1] == ["甲", "1", "True", "0.10000000000000001", "comma, and text"]
    assert rows[2] == ["乙", "2", "False", "", "missing"]
    assert proof["other_field_strings_exactly_equal"]
    assert b"\r\n" in target.read_bytes()
    assert "0.10000000000000001" in source.read_text(encoding="utf-8-sig")


def test_revised_eye_link_rejects_duplicate_keys(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("Participant,GlobalTrialOrder,IncludeEEGTrialValid\n甲,1,False\n甲,1,True\n", encoding="utf-8-sig")
    with pytest.raises(module.StageBlockedError, match="Duplicate"):
        module.revised_eye_link(source, tmp_path / "derived.csv", set())
