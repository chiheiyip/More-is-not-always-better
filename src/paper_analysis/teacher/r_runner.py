"""Run teacher R scripts with exact inputs through ASCII-only Windows paths."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import json
import os
import re
from pathlib import Path

import pandas as pd

from paper_analysis.teacher.state import StageBlockedError, file_sha256


def _reuse_r(script: Path, arguments: list[str], output: Path, environment: dict | None = None) -> bool:
    """Reuse registered R results only after checking model inputs and methods."""
    reuse_root = os.environ.get("PAPER_ANALYSIS_R_REUSE_ROOT")
    if not reuse_root:
        return False
    root = Path(reuse_root)
    try:
        run = next(p for p in output.parents if p.parent.name == "teacher_runs")
        source = root / output.relative_to(run)
        if environment is not None:
            runtime_record = source / f"{script.stem}_runtime_lock.json"
            if not runtime_record.is_file() or json.loads(runtime_record.read_text(encoding="utf-8")) != environment:
                return False
        manifest_folder = next(p for p in [source, *source.parents] if (p / "run_manifest.json").is_file())
        manifest = json.loads((manifest_folder / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            return False
        original_config = json.loads(Path(manifest["config_path"]).read_text(encoding="utf-8"))
        # Scalar inference settings must match; input-dependent metric lists
        # are also checked against the actual previous model input columns.
        if script.name == "eye_stage2_analysis.R" and float(arguments[2]) != original_config["eye"]["primary_tracking_threshold"]:
            return False
        if script.name == "eye_stage3_analysis.R":
            if set(arguments[2].split(",")) != set(manifest["arguments"]["triggers"]) or int(arguments[5]) != original_config["stage3"]["bootstrap_iterations"]:
                return False
        if script.name == "eeg_primary_analysis.R" and int(arguments[4]) != original_config["eeg"]["bootstrap_iterations"]:
            return False
        if script.name not in {"eye_stage2_analysis.R", "eye_stage3_analysis.R", "eeg_primary_analysis.R"}:
            return False
        repo = script.resolve().parents[2]
        old_commit = manifest["git_commit"]
        old_script = subprocess.check_output(["git", "show", f"{old_commit}:analysis/r/{script.name}"], cwd=repo).decode("utf-8")
        current_script = script.read_text(encoding="utf-8")
        normalized = re.sub(r"read_teacher_csv\(([^\n]+?)\)", r"utils::read.csv(\1, check.names = FALSE)", current_script)
        if normalized.strip() != old_script.replace("\r\n", "\n").strip():
            return False
        old_common = subprocess.check_output(["git", "show", f"{old_commit}:analysis/r/common.R"], cwd=repo).decode("utf-8").replace("\r\n", "\n").strip()
        common_prefix = (script.parent / "common.R").read_text(encoding="utf-8").split("read_teacher_csv <- function(path)")[0].strip()
        if common_prefix != old_common:
            return False
        index = pd.read_excel(root.parent / "结果文件总索引.xlsx")
        registered = {str(row.RelativePath).replace("\\", "/"): row.SHA256 for row in index.itertuples()}
        def verify_registered(path):
            relative = path.relative_to(root.parent).as_posix()
            return path.is_file() and registered.get(relative) == file_sha256(path)
        comparisons = []
        for position, argument in enumerate(arguments):
            if position == 1 or not argument or not Path(argument).is_file():
                continue
            previous = source / Path(argument).name
            if not verify_registered(previous):
                return False
            before, after = pd.read_csv(previous), pd.read_csv(argument)
            # Empty historical C0 is the same categorical baseline; block
            # starts keep missing previous values rather than becoming C0.
            for table in (before, after):
                if "Complexity" in table:
                    table["Complexity"] = table["Complexity"].fillna("C0")
                if {"PreviousWWR", "PreviousComplexity"}.issubset(table):
                    table.loc[table.PreviousWWR.notna() & table.PreviousComplexity.isna(), "PreviousComplexity"] = "C0"
            pd.testing.assert_frame_equal(before, after, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
            comparisons.append({"previous": str(previous), "previous_sha256": file_sha256(previous),
                                "current": argument, "current_sha256": file_sha256(argument), "rows": len(after)})
        names = set(re.findall(r'file.path\(outdir,\s*"([^"]+)"\)', old_script))
        names.add("R_session_info.txt")
        artifacts = [source / name for name in sorted(names) if (source / name).is_file()]
        if not comparisons or len(artifacts) < 3 or not all(verify_registered(p) for p in artifacts):
            return False
        if script.name == "eeg_primary_analysis.R":
            from paper_analysis.teacher.eeg import _metric_columns
            old_input = pd.read_csv(source / Path(arguments[0]).name)
            eeg_config = original_config["eeg"]
            relative, absolute = _metric_columns(old_input, eeg_config["core_metrics"])
            absolute = [f"log10_{name}" for name in absolute]
            secondary, _ = _metric_columns(old_input, eeg_config["secondary_metrics"])
            supplemental, _ = _metric_columns(old_input, eeg_config["supplemental_metrics"])
            if [arguments[i] for i in (2, 3, 5, 6)] != [",".join(v) for v in (relative, absolute, secondary, supplemental)]:
                return False
        records = [{"path": p.name, "sha256": file_sha256(p)} for p in artifacts]
        for artifact in artifacts:
            shutil.copyfile(artifact, output / artifact.name)
        proof = {"status": "reused", "source": str(source), "historical_git_commit": old_commit,
                 "method_check": "R inference code identical after replacing only the UTF-8 input reader",
                 "input_check": "all fields and rows equal after explicit C0 baseline relabel; absolute tolerance 1e-12",
                 "input_comparisons": comparisons, "arguments": arguments, "outputs": records}
        (output / f"{script.stem}_reuse.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except (OSError, ValueError, KeyError, StopIteration, AssertionError, subprocess.CalledProcessError):
        return False


def invoke_r(rscript: str, script: Path, arguments: list[str], *, required: bool = True) -> bool:
    executable = str(Path(rscript).resolve()) if Path(rscript).exists() else shutil.which(rscript)
    if not executable:
        if required:
            raise StageBlockedError(f"Rscript is unavailable ({rscript}); restore the registered R environment.")
        return False
    from .runtime_lock import validate_r
    environment = validate_r(executable)
    # Teacher scripts share the contract: input file, output directory, then
    # optional scalar arguments or additional input files.
    output = Path(arguments[1])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{script.stem}_runtime_lock.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    if _reuse_r(script, arguments, output, environment):
        return True
    scratch = script.resolve().parents[2] / ".codex_tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="teacher_r_", dir=scratch) as temporary:
        stage = Path(temporary)
        for dependency in script.parent.glob("*.R"):
            shutil.copyfile(dependency, stage / dependency.name)
        # The executing process receives the same settings validated in preflight.
        from .runtime_lock import REPO as runtime_repo
        shutil.copyfile(runtime_repo / "analysis/r/runtime_profile.R", stage / "runtime_profile.R")
        shutil.copyfile(runtime_repo / "analysis/r/runtime-versions.lock.json", stage / "runtime-versions.lock.json")
        staged_script = stage / script.name
        prefix = ('source("runtime_profile.R")\n'
                  'configure_locked_runtime("runtime-versions.lock.json", "' + environment["profile"] + '")\n')
        staged_script.write_text(prefix + staged_script.read_text(encoding="utf-8"), encoding="utf-8")
        staged_arguments = list(arguments)
        staged_arguments[1] = "out"
        (stage / "out").mkdir()
        for index, value in enumerate(arguments):
            if index == 1 or not value:
                continue
            source = Path(value)
            if source.is_file():
                name = f"input_{index}{source.suffix}"
                shutil.copyfile(source, stage / name)
                staged_arguments[index] = name
        log = output / f"{script.stem}_execution.log"
        with log.open("w", encoding="utf-8") as handle:
            from .runtime_lock import locked_r_environment
            subprocess.run([executable, script.name, *staged_arguments], cwd=stage, env=locked_r_environment(),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        for artifact in (stage / "out").iterdir():
            if artifact.is_dir():
                shutil.copytree(artifact, output / artifact.name, dirs_exist_ok=True)
            else:
                shutil.copyfile(artifact, output / artifact.name)
    return True
