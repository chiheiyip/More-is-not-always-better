from pathlib import Path
import subprocess
import shutil

import pytest

from paper_analysis.teacher.r_runner import invoke_r


def test_unicode_inputs_and_optional_files_are_exactly_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "Rscript")
    repo = tmp_path / "repo"
    scripts = repo / "analysis" / "r"
    scripts.mkdir(parents=True)
    script = scripts / "analysis.R"
    script.write_text("source('common.R')", encoding="utf-8")
    (scripts / "common.R").write_text("# common", encoding="utf-8")
    source = tmp_path / "原始.csv"
    source.write_bytes("姓名,功率\n参与者,0.123456789\n".encode("utf-8-sig"))
    extra = tmp_path / "补充.csv"
    extra.write_bytes(b"optional,exact\n1,2\n")
    output = tmp_path / "输出"
    def run(command, *, cwd, stdout, stderr, check):
        assert command[1:] == ["analysis.R", "input_0.csv", "out", "5000", "input_3.csv", ""]
        assert (cwd / "input_0.csv").read_bytes() == source.read_bytes()
        assert (cwd / "input_3.csv").read_bytes() == extra.read_bytes()
        assert (cwd / "common.R").is_file()
        (cwd / "out" / "result.csv").write_bytes(b"beta,p\n0.1,0.02\n")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    assert invoke_r("Rscript", script, [str(source), str(output), "5000", str(extra), ""])
    assert (output / "result.csv").read_bytes() == b"beta,p\n0.1,0.02\n"


def test_failed_r_does_not_publish_partial_results(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "Rscript")
    scripts = tmp_path / "repo" / "analysis" / "r"
    scripts.mkdir(parents=True)
    script = scripts / "analysis.R"
    script.write_text("stop('failed')", encoding="utf-8")
    source = tmp_path / "input.csv"
    source.write_bytes(b"value\n1\n")
    output = tmp_path / "out"
    def run(command, *, cwd, **kwargs):
        (cwd / "out" / "partial.csv").write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        invoke_r("Rscript", script, [str(source), str(output)])
    assert not (output / "partial.csv").exists()
