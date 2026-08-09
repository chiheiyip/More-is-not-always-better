from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_METHOD_DEPENDENCIES = {
    "eye-stage1": (
        "src/paper_analysis/teacher/eye.py",
        "src/paper_analysis/teacher/contracts.py",
    ),
    "eye-stage2": (
        "src/paper_analysis/teacher/eye.py",
        "src/paper_analysis/teacher/contracts.py",
        "analysis/r/eye_stage2_analysis.R",
        "analysis/r/common.R",
    ),
    "eye-stage3-plan": (
        "src/paper_analysis/teacher/eye.py",
        "src/paper_analysis/teacher/contracts.py",
    ),
    "eye-stage3-run": (
        "src/paper_analysis/teacher/eye.py",
        "src/paper_analysis/teacher/contracts.py",
        "analysis/r/eye_stage3_analysis.R",
        "analysis/r/common.R",
    ),
    "eeg-order": (
        "src/paper_analysis/teacher/eeg.py",
        "src/paper_analysis/teacher/contracts.py",
        "analysis/r/eeg_order_analysis.R",
        "analysis/r/common.R",
    ),
    "eeg-primary": (
        "src/paper_analysis/teacher/eeg.py",
        "src/paper_analysis/teacher/contracts.py",
        "analysis/r/eeg_primary_analysis.R",
        "analysis/r/eeg_crossmodal_analysis.R",
        "analysis/r/common.R",
    ),
}


class StageBlockedError(RuntimeError):
    """Raised when a mandatory manual or data-quality gate has not passed."""


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_fingerprint(
    stage: str,
    config: dict[str, Any],
    input_paths: list[str | Path],
) -> str:
    payload: dict[str, Any] = {
        "stage": stage,
        "config": config,
        "inputs": [],
    }
    for record in input_hash_records(input_paths):
        payload["inputs"].append(record)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def input_hash_records(input_paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in sorted({str(Path(path).resolve()) for path in input_paths if path}):
        path = Path(value)
        records.append({
            "path": value,
            "exists": path.exists(),
            "sha256": file_sha256(path) if path.is_file() else None,
        })
    return records


def write_input_hashes(
    outdir: str | Path, input_paths: list[str | Path]
) -> Path:
    target = Path(outdir) / "input_hashes.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(input_hash_records(input_paths), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def git_commit(cwd: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip()


def method_contract_hash(repo_root: str | Path) -> str:
    root = Path(repo_root)
    files = [
        *sorted((root / "src" / "paper_analysis").rglob("*.py")),
        *sorted((root / "analysis" / "r").glob("*.R")),
        *sorted((root / "matlab" / "eeg_bandpower_pipeline").glob("*.m")),
        root / "scripts" / "run_teacher_analysis.py",
        root / "scripts" / "build_teacher_candidate_inputs.py",
        root / "scripts" / "run_eeg_from_raw.py",
        root / "scripts" / "repair_eeg_clock_cache_encoding.py",
        root / "scripts" / "run_clock_synchronized_fusion.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stage_method_contract_hash(repo_root: str | Path, stage: str) -> str:
    root = Path(repo_root)
    dependencies = STAGE_METHOD_DEPENDENCIES.get(stage)
    if dependencies is None:
        return method_contract_hash(root)
    digest = hashlib.sha256()
    for relative in dependencies:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def legacy_stage_methods_unchanged(
    repo_root: str | Path,
    stage: str,
    source_commit: str,
) -> bool:
    dependencies = STAGE_METHOD_DEPENDENCIES.get(stage)
    if not dependencies or not source_commit or source_commit == "unavailable":
        return False
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", source_commit, "--", *dependencies],
            cwd=Path(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return result.returncode == 0


def new_run_directory(outputs_root: str | Path, run_id: str | None = None) -> Path:
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(outputs_root) / "teacher_runs" / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_run_manifest(
    outdir: str | Path,
    *,
    stage: str,
    fingerprint: str,
    config_path: str | Path,
    arguments: dict[str, Any],
    repo_root: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "stage": stage,
        "stage_fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "arguments": arguments,
        "git_commit": git_commit(repo_root),
        "method_contract_hash": method_contract_hash(repo_root),
        "stage_method_contract_hash": stage_method_contract_hash(
            repo_root, stage
        ),
        "python": sys.version,
        "python_packages": {
            name: _package_version(name)
            for name in (
                "numpy", "pandas", "scipy", "statsmodels",
                "matplotlib", "openpyxl", "Pillow",
                "python-docx",
            )
        },
        "platform": platform.platform(),
        **(extra or {}),
    }
    path = Path(outdir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def write_approval_template(
    path: str | Path,
    *,
    fingerprint: str,
    stage: str,
    notes: list[str] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "approved": False,
        "stage": stage,
        "stage_fingerprint": fingerprint,
        "approved_by": "",
        "approved_at": "",
        "approved_triggers": [],
        "notes": notes or [],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def require_approval(
    path: str | Path,
    *,
    fingerprint: str,
    stage: str,
) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise StageBlockedError(
            f"{stage} requires approval file: {target}. "
            "Create it from the generated template and preserve stage_fingerprint."
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageBlockedError(f"Approval file is not valid JSON: {target}") from exc
    if not payload.get("approved"):
        raise StageBlockedError(f"Approval is not granted in {target}")
    if payload.get("stage") != stage:
        raise StageBlockedError(
            f"Approval stage mismatch in {target}: {payload.get('stage')!r} != {stage!r}"
        )
    if payload.get("stage_fingerprint") != fingerprint:
        raise StageBlockedError(
            f"Approval fingerprint is stale for {stage}; rerun review and approve again."
        )
    return payload
