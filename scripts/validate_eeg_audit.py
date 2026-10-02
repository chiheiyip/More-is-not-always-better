"""Run real-input code validation without replacing or promoting any results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from run_teacher_analysis import _resolve_paths
from paper_analysis.teacher.eeg_audit import run_eeg_audit, CORE, TRIMS, sample_hash
from paper_analysis.teacher.state import file_sha256


def inventory(roots: list[Path]) -> dict[str, dict]:
    result = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                result[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def compare_history(out: Path, historical: Path) -> dict:
    new = pd.read_csv(out/"coefficients.csv")
    new = new.loc[new.model.eq("Model1") & new.outcome.isin([f"{m}_relative" for m in CORE])]
    comparisons = []; sample_records = []
    for trim in TRIMS:
        folder = historical/"onset_window_models"/f"trim_{trim}s"
        path = folder/"eeg_cr2_robust_results.csv"
        old = pd.read_csv(path)
        old = old.loc[old.model.eq("Model1") & old.outcome.isin([f"{m}_relative" for m in CORE])]
        old["onset_trim_s"] = trim
        merged = old.merge(new.loc[new.onset_trim_s.eq(trim)], on=["outcome", "model", "term", "onset_trim_s"], suffixes=("_historical", "_validation"), how="outer", indicator=True, validate="one_to_one")
        for col in ("estimate", "std.error", "df", "p.value"):
            merged[f"{col}_difference"] = merged[f"{col}_validation"] - merged[f"{col}_historical"]
        comparisons.append(merged)
        old_input = pd.read_csv(folder/"eeg_onset_order_model_input.csv")
        new_input = pd.read_csv(out/f"trim_{trim}s"/"input.csv")
        keys_equal = sample_hash(old_input) == sample_hash(new_input)
        old0 = old_input.Complexity.isna()
        confirmed_c0 = bool(old_input.loc[old0,"Cond"].astype(str).str.contains("C0").all()) if "Cond" in old_input else False
        # The old export's blank C0 was an empty-string R factor reference.
        if old0.any() and not confirmed_c0:
            raise ValueError("Historical blank complexity cannot be verified as C0")
        old_input.loc[old0,"Complexity"] = "C0"
        fields = ["Participant", "GlobalTrialOrder", "WWR", "Complexity", "Gender", "ExperienceGroup", "OrderGroup", "Block", "PositionWithinBlockCentered", *[f"{m}_relative" for m in CORE]]
        left=old_input[fields].sort_values(["Participant","GlobalTrialOrder"]).reset_index(drop=True)
        right=new_input[fields].sort_values(["Participant","GlobalTrialOrder"]).reset_index(drop=True)
        equal = keys_equal and len(left)==len(right)
        differences=[]
        if equal:
            for col in fields:
                if pd.api.types.is_numeric_dtype(left[col]):
                    match=np.allclose(left[col],right[col],rtol=1e-12,atol=1e-14,equal_nan=True)
                else: match=left[col].astype(str).equals(right[col].astype(str))
                if not match:differences.append(col)
        sample_records.append({"onset_trim_s":trim,"historical_trials":len(old_input),"validation_trials":len(new_input),
                               "same_sample_keys":keys_equal,"historical_blank_c0":int(old0.sum()),
                               "normalized_inputs_equal":equal and not differences,"different_fields":";".join(differences),
                               "historical_input_sha256":file_sha256(folder/"eeg_onset_order_model_input.csv"),
                               "historical_coefficients_sha256":file_sha256(path)})
    comparison=pd.concat(comparisons,ignore_index=True)
    comparison.to_csv(out/"historical_coefficient_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(sample_records).to_csv(out/"historical_sample_comparison.csv",index=False,encoding="utf-8-sig")
    beta = comparison["estimate_difference"].abs().max()
    p_difference = comparison["p.value_difference"].abs().max()
    concordant = bool(comparison._merge.eq("both").all() and np.isfinite(beta) and beta <= 1e-8
                      and np.isfinite(p_difference) and p_difference <= 1e-6
                      and all(r["normalized_inputs_equal"] for r in sample_records))
    return {"comparison_status":"concordant_within_numerical_tolerance" if concordant else "requires_investigation",
            "beta_atol":1e-8,"raw_p_atol":1e-6,
            "matched_rows":int(comparison._merge.eq("both").sum()),"unmatched_rows":int(comparison._merge.ne("both").sum()),
            "max_abs_beta_difference":float(beta) if np.isfinite(beta) else None,
            "max_abs_raw_p_difference":float(p_difference) if np.isfinite(p_difference) else None,
            "all_sample_keys_equal":all(r["same_sample_keys"] for r in sample_records),
            "all_normalized_inputs_equal":all(r["normalized_inputs_equal"] for r in sample_records)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--historical-dir",type=Path,required=True)
    parser.add_argument("--protect",type=Path,action="append",default=[])
    args=parser.parse_args()
    out=args.outdir.resolve()
    protected=[p.resolve() for p in args.protect]
    for p in protected:
        if not p.is_dir():raise ValueError(f"Protected directory missing: {p}")
        if out==p or p in out.parents:raise ValueError("Validation output must be outside protected directories")
    before=inventory(protected)
    config=_resolve_paths(json.loads(args.config.read_text(encoding="utf-8")),args.config.resolve())
    try:
        run_eeg_audit(config,config_path=args.config.resolve(),outdir=out,repo_root=ROOT)
        comparison=compare_history(out,args.historical_dir.resolve())
        (out/"historical_comparison_summary.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    finally:
        after=inventory(protected)
        changed=[p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)]
        if out.is_dir():
            (out/"protected_directory_check.json").write_text(json.dumps({"unchanged":not changed,"files_checked":len(before),"changed_files":changed,"method":"exact path, size and mtime_ns inventory; direct inputs additionally SHA-256 verified by runner"},indent=2),encoding="utf-8")
            records=[{"path":p.relative_to(out).as_posix(),"sha256":file_sha256(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name!="output_hashes.json"]
            (out/"output_hashes.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        if changed:raise RuntimeError("Protected directory changed during validation")
    print(json.dumps(comparison,indent=2))
    status=json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))["status"]
    return 0 if status=="validated" and comparison["comparison_status"]=="concordant_within_numerical_tolerance" else 3


if __name__=="__main__":raise SystemExit(main())
