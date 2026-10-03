from pathlib import Path
import subprocess
import shutil
import json
import sys
import pandas as pd

import pytest

from paper_analysis.teacher.r_runner import invoke_r, _reuse_r
from paper_analysis.teacher.state import file_sha256


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


@pytest.mark.parametrize("alter", [None, "data", "method", "output", "threshold"])
def test_registered_reuse_requires_equal_inputs_methods_and_hashes(tmp_path, monkeypatch, alter):
    repo = tmp_path / "repo"
    scripts = repo / "analysis" / "r"
    scripts.mkdir(parents=True)
    script = scripts / "eye_stage2_analysis.R"
    old = 'x <- utils::read.csv(args[[1]], check.names = FALSE)\nwrite_csv_utf8(x, file.path(outdir, "results.csv"))\n'
    new = old.replace('utils::read.csv(args[[1]], check.names = FALSE)', 'read_teacher_csv(args[[1]])')
    script.write_text(new + ("# changed model\n" if alter == "method" else ""), encoding="utf-8")
    (scripts / "common.R").write_text('# original\nread_teacher_csv <- function(path) {}', encoding="utf-8")
    root = tmp_path / "results"
    source = root / "12_teacher_analysis" / "02_eye_stage2"
    source.mkdir(parents=True)
    output = root / "teacher_runs" / "current" / "02_eye_stage2"
    output.mkdir(parents=True)
    original_config = root / "previous.json"
    original_config.write_text(json.dumps({"eye": {"primary_tracking_threshold": .6}}))
    (source / "run_manifest.json").write_text(json.dumps({"status": "complete", "git_commit": "previous", "config_path": str(original_config)}))
    (source / "input.csv").write_text("Participant,Complexity,value\np1,,0.123\np2,C1,0.321\n")
    current = output / "input.csv"
    current.write_text("Participant,Complexity,value\np1,C0,0.123\np2,C1," + ("0.322" if alter == "data" else "0.321") + "\n")
    for name in ["results.csv", "plot.png", "R_session_info.txt"]:
        (source / name).write_bytes(b"registered exact output")
    old += 'safe_png(file.path(outdir, "plot.png"), {})\n'
    script.write_text(new + 'safe_png(file.path(outdir, "plot.png"), {})\n' + ("# changed model\n" if alter == "method" else ""))
    index = pd.DataFrame([{"RelativePath": p.relative_to(root).as_posix(), "SHA256": file_sha256(p)} for p in source.iterdir()])
    if alter == "output":
        (source / "results.csv").write_bytes(b"tampered")
    monkeypatch.setenv("PAPER_ANALYSIS_R_REUSE_ROOT", str(root / "12_teacher_analysis"))
    monkeypatch.setattr(pd, "read_excel", lambda _: index)
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kw: (old if "eye_stage2" in command[-1] else "# original\n").encode())
    reused = _reuse_r(script, [str(current), str(output), ".7" if alter == "threshold" else ".6"], output)
    assert reused == (alter is None)
    assert (output / "results.csv").exists() == (alter is None)


def test_real_r_reads_chinese_identifiers_without_locale_transcoding(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    launcher = repo / "scripts" / "portable_rscript.cmd"
    if sys.platform != "win32" or not (repo / ".r-env/Lib/R/bin/Rscript.exe").exists():
        pytest.skip("portable R is unavailable")
    scripts = tmp_path / "repo" / "analysis" / "r"
    scripts.mkdir(parents=True)
    shutil.copyfile(repo / "analysis" / "r" / "common.R", scripts / "common.R")
    script = scripts / "read_input.R"
    script.write_text('args <- commandArgs(TRUE)\nsource("common.R")\nx <- read_teacher_csv(args[1])\nstopifnot(names(x)[1] == "Participant", nrow(x) == 2, length(unique(x$Participant)) == 2)\ndir.create(args[2], showWarnings=FALSE)\nwrite_csv_utf8(x, file.path(args[2], "roundtrip.csv"))\n', encoding="utf-8")
    source = tmp_path / "原始.csv"
    source.write_text("Participant,value\n参与者甲,0.123\n参与者乙,0.321\n", encoding="utf-8-sig")
    output = tmp_path / "输出"
    assert invoke_r(str(launcher), script, [str(source), str(output)])
    pd.testing.assert_frame_equal(pd.read_csv(source), pd.read_csv(output / "roundtrip.csv"))
