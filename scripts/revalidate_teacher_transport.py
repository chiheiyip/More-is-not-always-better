"""Revalidate completed stages after a change confined to R reuse checks.

This retains the original execution commit and records a new validation commit.
Changes to data preparation, inference, or the R invocation itself are rejected.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from paper_analysis.teacher.complete import STAGE_FOLDERS, _input_hashes_current
from paper_analysis.teacher.state import STAGE_METHOD_DEPENDENCIES, file_sha256, git_commit, stage_method_contract_hash


def without_reuse(source):
    parsed = ast.parse(source)
    parsed.body = [node for node in parsed.body if not isinstance(node, ast.FunctionDef) or node.name != "_reuse_r"]
    return ast.dump(parsed, include_attributes=False)


def revalidate(run: Path):
    for stage, folder in STAGE_FOLDERS.items():
        directory = run / folder
        path = directory / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        current = stage_method_contract_hash(REPO, stage)
        if manifest.get("stage_method_contract_hash") == current:
            continue
        changed = subprocess.check_output(["git", "diff", "--name-only", manifest["git_commit"], "--", *STAGE_METHOD_DEPENDENCIES[stage]], cwd=REPO, text=True).splitlines()
        if changed != ["src/paper_analysis/teacher/r_runner.py"]:
            raise ValueError(f"{stage}: changes extend beyond R reuse checks: {changed}")
        previous = subprocess.check_output(["git", "show", manifest["git_commit"] + ":src/paper_analysis/teacher/r_runner.py"], cwd=REPO).decode("utf-8")
        if without_reuse(previous) != without_reuse((REPO / changed[0]).read_text(encoding="utf-8")):
            raise ValueError(f"{stage}: R invocation or input staging changed")
        if not _input_hashes_current(directory / "input_hashes.json"):
            raise ValueError(f"{stage}: recorded inputs changed")
        proof = {"previous_contract": manifest.get("stage_method_contract_hash"), "current_contract": current,
                 "validation_git_sha": git_commit(REPO), "changed_source": changed,
                 "check": "AST identical except the _reuse_r cache validation function; all recorded input hashes unchanged",
                 "output_hashes": [{"path": p.relative_to(directory).as_posix(), "sha256": file_sha256(p)}
                                   for p in sorted(directory.rglob("*")) if p.is_file() and p != path]}
        manifest["stage_method_contract_hash"] = current
        manifest["transport_revalidation"] = proof
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(stage, "revalidated", len(proof["output_hashes"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    revalidate(parser.parse_args().run)
