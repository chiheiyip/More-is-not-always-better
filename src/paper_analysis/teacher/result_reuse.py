"""Verify a completed frozen-sample sensitivity run before integrating it."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from paper_analysis.teacher.eeg_denominator import METHOD_FILES, verify_cache
from paper_analysis.teacher.state import file_sha256


def reuse_sensitivity(config, *, outputs_root: Path, destination: Path, repo: Path) -> bool:
    pointer = outputs_root / "eeg_denominator_sensitivity_latest.json"
    if destination.exists() or not pointer.is_file():
        return False
    try:
        source = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_root"])
        manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["config"]["denominator_sensitivity"] != config["denominator_sensitivity"]:
            return False
        if Path(manifest["config"]["rscript"]).resolve() != Path(config["rscript"]).resolve():
            return False
        records = {str(Path(r["path"]).resolve()): r for r in manifest["input_code_hashes"]}
        checks = []
        for relative in METHOD_FILES:
            current = repo / relative
            previous = source / "code_snapshot" / relative
            if not previous.is_file() or file_sha256(previous) != records[str(current.resolve())]["sha256"]:
                return False
            before = previous.read_text(encoding="utf-8").rstrip()
            after = current.read_text(encoding="utf-8").rstrip()
            if relative == "analysis/r/common.R":
                # eeg_audit.R retains its own UTF-8 reader and never calls
                # the added teacher reader. Every inference function is exact.
                after = after.split("read_teacher_csv <- function(path)")[0].rstrip()
            if before != after:
                return False
            checks.append({"path": relative, "check": "same inference source after newline normalization", "current_sha256": file_sha256(current)})
        for key in ("historical_package", "candidate_factor_reference"):
            path = Path(config["denominator_sensitivity"][key])
            if file_sha256(path) != records[str(path.resolve())]["sha256"]:
                return False
        for waveform in manifest["psd_cache"]["sources"]:
            for kind in ("set", "fdt"):
                if file_sha256(waveform[f"{kind}_path"]) != waveform[f"{kind}_sha256"]:
                    return False
        cache = manifest["psd_cache"]
        verify_cache(cache["cache_dir"], cache["cache_key"])
        outputs = json.loads((source / "output_hashes.json").read_text(encoding="utf-8"))
        if not outputs or not all((source / r["path"]).is_file() and file_sha256(source / r["path"]) == r["sha256"] for r in outputs):
            return False
        shutil.copytree(source, destination)
        proof = {"status": "reused", "source_run": str(source), "analysis_git_sha": manifest["git_sha"],
                 "checks": checks, "output_files_verified": len(outputs),
                 "source_waveforms_verified": len(cache["sources"]),
                 "cache": "complete spectra and integral output hashes verified",
                 "configuration": "same denominators, frozen historical sample, QC, factor reference and PSD parameters"}
        (destination / "reuse_verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except (OSError, ValueError, KeyError):
        return False
