"""Run teacher R scripts with exact inputs through ASCII-only Windows paths."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from paper_analysis.teacher.state import StageBlockedError


def invoke_r(rscript: str, script: Path, arguments: list[str], *, required: bool = True) -> bool:
    executable = str(Path(rscript).resolve()) if Path(rscript).exists() else shutil.which(rscript)
    if not executable:
        if required:
            raise StageBlockedError(f"Rscript is unavailable ({rscript}); restore the registered R environment.")
        return False
    # Teacher scripts share the contract: input file, output directory, then
    # optional scalar arguments or additional input files.
    output = Path(arguments[1])
    output.mkdir(parents=True, exist_ok=True)
    scratch = script.resolve().parents[2] / ".codex_tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="teacher_r_", dir=scratch) as temporary:
        stage = Path(temporary)
        for dependency in script.parent.glob("*.R"):
            shutil.copyfile(dependency, stage / dependency.name)
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
            subprocess.run([executable, script.name, *staged_arguments], cwd=stage,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        for artifact in (stage / "out").iterdir():
            if artifact.is_dir():
                shutil.copytree(artifact, output / artifact.name, dirs_exist_ok=True)
            else:
                shutil.copyfile(artifact, output / artifact.name)
    return True
