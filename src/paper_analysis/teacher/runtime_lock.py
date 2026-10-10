"""Fail closed on changes to the formally verified calculation environment."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from .state import StageBlockedError, file_sha256

REPO = Path(__file__).resolve().parents[3]


def validate_python(repo: Path = REPO) -> dict:
    path = Path(repo) / "configs/analysis_python.lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    actual = {"python": platform.python_version(), "system": platform.system(),
              "machine": platform.machine(), "packages": {}}
    differences = []
    for key in ("python", "system", "machine"):
        if actual[key] != lock[key]:
            differences.append(f"{key}: expected {lock[key]}, found {actual[key]}")
    for name, expected in lock["packages"].items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        actual["packages"][name] = version
        if version != expected:
            differences.append(f"{name}: expected {expected}, found {version}")
    if differences:
        raise StageBlockedError("Analysis environment lock mismatch; use Python 3.12.10 and "
                               "requirements-analysis.lock.txt. " + "; ".join(differences))
    return {"status": "passed", "lock_sha256": file_sha256(path), **actual}


def validate_r(rscript: str, *, profile: str | None = None,
               library: str | None = None, repo: Path = REPO) -> dict:
    """Probe the actual launcher/library selection before inference or reuse."""
    repo = Path(repo)
    lock_path = repo / "analysis/r/runtime-versions.lock.json"
    env = os.environ.copy()
    if library:
        env["R_LIBS_USER"] = library
    try:
        result = subprocess.run([str(rscript), str(repo / "analysis/r/verify_runtime_lock.R"),
                                 str(lock_path), profile or "auto"],
                                cwd=repo, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise StageBlockedError(f"R environment lock mismatch: {detail}") from exc
    marker = "ANALYSIS_RUNTIME_LOCK="
    records = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(records) != 1:
        raise StageBlockedError("R environment probe did not return one verified snapshot")
    return {"lock_sha256": file_sha256(lock_path), **json.loads(records[0])}


def validate_analysis(config: dict, repo: Path = REPO, *, require_r: bool = True) -> dict:
    snapshot = {"python": validate_python(repo)}
    if require_r:
        if not config.get("rscript"):
            raise StageBlockedError("Locked R launcher must be specified in rscript")
        snapshot["r_primary"] = validate_r(config["rscript"], repo=repo, profile="primary")
    fresh = config.get("fresh", {})
    if fresh.get("verification_rscript"):
        snapshot["r_independent"] = validate_r(fresh["verification_rscript"], repo=repo,
                                              profile="independent",
                                              library=fresh.get("verification_r_library"))
    return snapshot
