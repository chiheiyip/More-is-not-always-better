"""Fresh eye calculations in the recorded historical environment, then compare."""
from __future__ import annotations

import io
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .aoi_lock import freeze_aoi, verify_aoi
from .runtime_lock import validate_analysis
from .state import StageBlockedError, file_sha256, git_commit, write_input_hashes

PRIMARY_FILES = ("09_familyA_primary_models.csv", "10_familyB_primary_models.csv",
                 "11_WWR_posthoc_Holm.csv", "12_CR2_robust_results.csv")


def historical_r_packages(session: str) -> list[tuple[str, str]]:
    # Platform strings such as x86_64-w64-mingw32 also contain underscores.
    # Only the explicitly declared namespace section lists package versions.
    marker = "loaded via a namespace (and not attached):"
    if marker not in session:
        raise StageBlockedError("Historical R namespace/version section missing")
    return re.findall(r"([A-Za-z][A-Za-z0-9.]+)_([0-9][A-Za-z0-9.+-]*)", session.split(marker, 1)[1])


def compare_frames(before: pd.DataFrame, after: pd.DataFrame, keys: list[str], name: str) -> list[dict]:
    for frame in (before, after):
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise StageBlockedError(f"Missing/duplicate comparison keys: {name}")
    if set(before.columns) != set(after.columns):
        raise StageBlockedError(f"Comparison fields differ: {name}")
    left = before.set_index(keys).sort_index()
    right = after.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise StageBlockedError(f"Comparison sample/term keys differ: {name}")
    records = []
    for field in before.columns:
        if field in keys:
            continue
        a, b = left[field], right[field]
        missing = int((a.isna() != b.isna()).sum())
        valid = a.notna() & b.notna()
        numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        if numeric and not pd.api.types.is_bool_dtype(a):
            x, y = a[valid].to_numpy(dtype=float), b[valid].to_numpy(dtype=float)
            tolerance = 1e-6 if field.startswith("p.value") else (1e-8 if field in {
                "estimate", "std.error", "std.error.CR2", "std.error.likelihood", "SE", "df",
                "conf.low", "conf.high", "statistic", "z.ratio", "t.ratio"} else 1e-12)
            if field in {"PixelArea", "ValidScenePixels"}:
                tolerance = 0.0
            exact = np.equal(x, y)
            with np.errstate(invalid="ignore"):
                delta = np.where(exact, 0.0, np.abs(x - y))
            passed = np.isclose(x, y, rtol=0, atol=tolerance)
            records.append({"file": name, "field": field, "rows": len(a), "missing_mismatches": missing,
                "unequal_values": int((~exact).sum()), "max_abs_difference": float(delta.max()) if len(delta) else 0,
                "tolerance": tolerance, "beyond_tolerance": int((~passed).sum()),
                "significance_flips": int(((x < .05) != (y < .05)).sum()) if field.startswith("p.value") else 0,
                "direction_flips": int(((np.sign(x) != np.sign(y)) & (x != 0) & (y != 0)).sum()) if field == "estimate" else 0})
        else:
            changed = int((a[valid].astype(str) != b[valid].astype(str)).sum())
            records.append({"file": name, "field": field, "rows": len(a), "missing_mismatches": missing,
                            "unequal_values": changed, "beyond_tolerance": changed,
                            "max_abs_difference": None, "tolerance": 0,
                            "significance_flips": 0, "direction_flips": 0})
    return records


def factor_encoding_check(before: pd.DataFrame, after: pd.DataFrame) -> list[dict]:
    keys = ["Participant", "GlobalTrialOrder"]
    a = before.sort_values(keys).set_index(keys)
    b = after.sort_values(keys).set_index(keys)
    if not a.index.equals(b.index):
        raise StageBlockedError("Factor encoding sample keys differ")
    previous_scene = a.groupby([a.index.get_level_values("Participant"), a.Block], sort=False).SceneID.shift()
    records = []
    for field, scene in [("Complexity", a.SceneID), ("PreviousComplexity", previous_scene)]:
        value = a[field].copy()
        baseline = value.isna() & scene.astype(str).str.contains("C0", regex=False)
        if field == "PreviousComplexity":
            baseline &= a.PreviousWWR.notna()
        value = value.astype(object)
        value.loc[baseline] = "C0"
        missing = int((value.isna() != b[field].isna()).sum())
        valid = value.notna() & b[field].notna()
        mismatches = int((value[valid].astype(str) != b[field][valid].astype(str)).sum()) + missing
        records.append({"field": field, "historical_empty_baseline_labels": int(baseline.sum()),
                        "canonical_mismatches": mismatches,
                        "proof": "C0 verified from current/previous within-block SceneID; true block-start missing stays missing"})
    return records


def prepare(config: dict, package: Path, out: Path, repo: Path) -> Path:
    environment = validate_analysis(config, repo)
    if out.exists():
        raise StageBlockedError("Historical eye verification requires a new output directory")
    with zipfile.ZipFile(package) as archive:
        members = [n for n in archive.namelist() if any(f"__{s}__" in n for s in
                   ("01_eye_stage1", "02_eye_stage2", "03_eye_stage3_plan", "04_eye_stage3"))]
        old_manifest = json.loads(archive.read(next(n for n in members if "01_eye_stage1__run_manifest.json" in n)))
        # The package is read for environment/source provenance now, never for model outcomes.
        versions = {k.lower(): v for k, v in environment["python"]["packages"].items()}
        if any(versions.get(k.lower()) != v for k, v in old_manifest["python_packages"].items()):
            raise StageBlockedError("Historical package Python libraries do not match the committed lock")
        if not old_manifest["python"].startswith(environment["python"]["python"] + " "):
            raise StageBlockedError("Historical Python patch version differs")
        session = archive.read(next(n for n in members if "02_eye_stage2__R_session_info.txt" in n)).decode("utf-8")
        rversion = re.search(r"R version ([0-9.]+)", session).group(1)
        packages = historical_r_packages(session)
        current = environment["r_primary"]
        if current["R"] != rversion or any(current["packages"].get(k) != v.replace("-", ".") for k, v in packages):
            raise StageBlockedError("R or loaded historical statistical packages differ")
        reference = out / "reference"; reference.mkdir(parents=True)
        for token, name in [("01_eye_stage1__08_AOI_area_report.xlsx", "areas.xlsx"),
                            ("01_eye_stage1__run_manifest.json", "run_manifest.json"),
                            ("01_eye_stage1__input_hashes.json", "input_hashes.json")]:
            (reference / name).write_bytes(archive.read(next(n for n in members if token in n)))
    # Prove shared original eye measurements/annotations unchanged before calculation.
    source_records = json.loads((reference / "input_hashes.json").read_text(encoding="utf-8"))
    root = Path(config["eye"]["raw_root"]).resolve()
    original = [r for r in source_records if Path(r["path"]).resolve().is_relative_to(root)]
    if not original:
        raise StageBlockedError("No historical original eye source hashes found")
    for record in original:
        source = Path(record["path"])
        if not source.is_file() or file_sha256(source) != record["sha256"]:
            raise StageBlockedError(f"Historical original source changed: {source}")
    lock = freeze_aoi(config, out / "frozen_aoi/aoi_pixel_lock.json", reference / "areas.xlsx", reference / "run_manifest.json")
    config = json.loads(json.dumps(config))
    config["eye"]["aoi_pixel_lock"] = lock["path"]
    config["eye"]["aoi_pixel_lock_sha256"] = lock["sha256"]
    config["eye"]["provisional_user_authorization"] = True
    for key, stage in [("stage1_dir", "01_eye_stage1"), ("stage2_dir", "02_eye_stage2"), ("stage3_plan_dir", "03_eye_stage3_plan")]:
        config["eye"][key] = str(out / stage)
    path = out / "config.local.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    proof = {"historical_package": str(package.resolve()), "package_sha256": file_sha256(package),
             "historical_calculation_sha": old_manifest["git_commit"], "verification_sha": git_commit(repo),
             "environment": environment, "historical_recorded_R_packages": len(packages),
             "original_eye_source_hashes_verified": len(original), "config_sha256": file_sha256(path),
             "original_eye_sources": original,
             "scope": "fresh eye stages 1/2/3; current EEG sample used only for common-sample supplements",
             "historical_transitive_python_versions": "not fully recorded; eight recorded libraries match exactly"}
    (out / "environment_and_source_proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(config_path: Path, out: Path, repo: Path):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise StageBlockedError("Real-data recalculation requires clean committed code")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    proof = json.loads((out / "environment_and_source_proof.json").read_text(encoding="utf-8"))
    if proof["verification_sha"] != git_commit(repo) or proof["config_sha256"] != file_sha256(config_path):
        raise StageBlockedError("Prepared verification code/config changed")
    validate_analysis(config, repo); verify_aoi(config, required=True)
    os.environ.pop("PAPER_ANALYSIS_R_REUSE_ROOT", None)
    from .eye import run_eye_stage1, run_eye_stage2, run_eye_stage3_plan, run_eye_stage3, _eye_stage_input_paths
    from .complete import _approve_stage3
    sources = _eye_stage_input_paths(Path(config["participant_information"]), Path(config["trial_order_mapping"]),
                                   Path(config["scene_aoi_mapping"]), Path(config["eye"]["raw_root"]), Path(config["eye"]["aoi_root"]))
    from .fresh import stage, verify_inputs
    from .state import input_hash_records
    records = input_hash_records(sources)
    verify_inputs(proof["original_eye_sources"])
    verify_inputs(records)
    fingerprint = hashlib.sha256((file_sha256(config_path) + git_commit(repo) +
                  json.dumps(records, sort_keys=True, ensure_ascii=False)).encode("utf-8")).hexdigest()
    for folder, function in [("01_eye_stage1", run_eye_stage1), ("02_eye_stage2", run_eye_stage2),
                             ("03_eye_stage3_plan", run_eye_stage3_plan)]:
        stage(out, folder, fingerprint, lambda f=function, s=folder: f(config, config_path=config_path,
              outdir=out/s, repo_root=repo), resume=True)
    _approve_stage3(out / "03_eye_stage3_plan")
    stage(out, "04_eye_stage3", fingerprint, lambda: run_eye_stage3(config, config_path=config_path,
          outdir=out/"04_eye_stage3", repo_root=repo), resume=True)
    verify_inputs(records)
    verify_inputs(proof["original_eye_sources"])
    (out / "fresh_calculation_complete.json").write_text(json.dumps({"status": "complete", "sha": git_commit(repo),
         "input_hashes": records, "config_sha256": file_sha256(config_path)}, ensure_ascii=False, indent=2), encoding="utf-8")


def compare(package: Path, out: Path):
    if not (out / "fresh_calculation_complete.json").is_file():
        raise StageBlockedError("Freeze fresh calculation before reading historical model outputs")
    proof = json.loads((out / "environment_and_source_proof.json").read_text(encoding="utf-8"))
    if file_sha256(package) != proof["package_sha256"]:
        raise StageBlockedError("Historical package changed")
    for seal in (out / "stage_seals").glob("*.json"):
        for record in json.loads(seal.read_text(encoding="utf-8"))["files"]:
            if file_sha256(out / record["path"]) != record["sha256"]:
                raise StageBlockedError(f"Fresh output changed after freeze: {record['path']}")
    results = []
    inventory = []
    encoding = []
    with zipfile.ZipFile(package) as archive:
        def read(stage, name, excel=False):
            data = io.BytesIO(archive.read(next(n for n in archive.namelist() if f"__{stage}__{name}" in n)))
            return pd.read_excel(data) if excel else pd.read_csv(data)
        for name in PRIMARY_FILES:
            keys = ["outcome", "contrast"] if "posthoc" in name else ["outcome", "model", "term"]
            results.extend(compare_frames(read("02_eye_stage2", name), pd.read_csv(out/"02_eye_stage2"/name), keys, name))
        for name, extra in [("13_tracking_threshold_sensitivity_models.csv", ["Threshold"]),
                            ("14_block1_sensitivity_models.csv", []),
                            ("15_leave_one_participant_out.csv", ["ExcludedParticipant"]),
                            ("16_EEG_valid_common_sample_sensitivity.csv", [])]:
            results.extend(compare_frames(read("02_eye_stage2", name), pd.read_csv(out/"02_eye_stage2"/name),
                                          ["outcome", "model", "term", *extra], name))
        name = "06_trial_level_eye_tracking_data.xlsx"
        historical_trials = read("02_eye_stage2", name, True)
        current_trials = pd.read_excel(out/"02_eye_stage2"/name)
        encoding = factor_encoding_check(historical_trials, current_trials)
        results.extend(compare_frames(historical_trials, current_trials,
                       ["Participant", "GlobalTrialOrder"], name))
        name = "08_AOI_area_report.xlsx"
        results.extend(compare_frames(read("01_eye_stage1", name, True), pd.read_excel(out/"01_eye_stage1"/name),
                       ["AOIImageID", "AOICategory"], name))
        for member in archive.namelist():
            if "__04_eye_stage3__" not in member or not member.endswith(".csv") or "model_input" in member:
                continue
            name = member.split("__04_eye_stage3__", 1)[1]
            current = out / "04_eye_stage3" / name
            old = pd.read_csv(io.BytesIO(archive.read(member)))
            new = pd.read_csv(current) if current.is_file() else None
            record = {"file": name, "historical_rows": len(old), "current_rows": len(new) if new is not None else None,
                      "historical_failed_rows": int(old.status.eq("fit_failed").sum()) if "status" in old else None,
                      "current_failed_rows": int(new.status.eq("fit_failed").sum()) if new is not None and "status" in new else None,
                      "comparison": "source/diagnostic inventory"}
            if new is not None and set(old.columns) == set(new.columns) and len(old) == len(new):
                keys = [k for k in ["outcome", "model", "term", "AOI", "Metric", "Outcome", "EyePredictor"] if k in old]
                if len(old) == 0:
                    record["comparison"] = "both empty; no valid inference"
                elif keys:
                    try:
                        rows = compare_frames(old, new, keys, name)
                    except StageBlockedError as exc:
                        record["comparison"] = str(exc)
                    else:
                        results.extend(rows); record["comparison"] = "all fields paired"
            elif new is None:
                record["comparison"] = "not produced under current evidence triggers; do not claim verified"
            else:
                record["comparison"] = "schema/row count changed; inspect original diagnostics and current outputs"
            inventory.append(record)
    table = pd.DataFrame(results)
    table["scope"] = "eye_standalone"
    verified_encoding = [r["field"] for r in encoding if r["canonical_mismatches"] == 0]
    table.loc[table.file.eq("06_trial_level_eye_tracking_data.xlsx") & table.field.isin(verified_encoding), "scope"] = "historical_empty_C0_encoding"
    table.loc[table.file.eq("16_EEG_valid_common_sample_sensitivity.csv") |
              (table.file.eq("06_trial_level_eye_tracking_data.xlsx") & table.field.str.startswith("IncludeEEG")), "scope"] = "current_eeg_common_sample"
    table.to_csv(out / "historical_primary_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory).to_csv(out / "historical_supplemental_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(encoding).to_csv(out / "factor_encoding_check.csv", index=False, encoding="utf-8-sig")
    sample = pd.read_excel(out / "02_eye_stage2/06_trial_level_eye_tracking_data.xlsx")
    primary = sample.loc[sample.IncludedPrimary.eq(True)]
    standalone = table.loc[table.scope.eq("eye_standalone")]
    main = table.loc[table.file.isin(PRIMARY_FILES)]
    areas = table.loc[table.file.eq("08_AOI_area_report.xlsx") & table.field.isin(["PixelArea", "ValidScenePixels"])]
    summary = {"participants": int(primary.Participant.nunique()), "trials": len(primary),
               "missing_mismatches": int(standalone.missing_mismatches.sum()),
               "beyond_tolerance": int(standalone.beyond_tolerance.sum()),
               "significance_flips": int(standalone.significance_flips.sum()), "direction_flips": int(standalone.direction_flips.sum()),
               "scope": "standalone eye comparisons; current EEG membership/common-sample models separately labelled",
               "supplemental_inventory": "historical_supplemental_inventory.csv; differences/unverified rows are not counted as passed"}
    summary["primary_model_comparison"] = {k: int(main[k].sum()) for k in
        ("missing_mismatches", "unequal_values", "beyond_tolerance", "significance_flips", "direction_flips")}
    summary["pixel_count_unequal_values"] = int(areas.unequal_values.sum())
    summary["factor_encoding_check"] = encoding
    (out / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# 眼动历史环境复算核查", "",
              f"历史包：{package.name}；历史计算 SHA：{proof['historical_calculation_sha']}。",
              f"本次复算 SHA：{proof['verification_sha']}；Python 和八个历史已登记库版本全部匹配；"
              f"核对历史已加载 R 包 {proof['historical_recorded_R_packages']} 个。",
              "历史未登记全部 Python 间接依赖，不能宣称完整恢复旧操作系统及每个间接依赖。", "",
              f"原始眼动文件/标注/底图哈希通过核对 {proof['original_eye_source_hashes_verified']} 个。"
              "本次从原始 CSV 重算，没有复用历史试次指标或模型输出。",
              f"主分析：{summary['participants']} 人、{summary['trials']} 试次。",
              f"AOI 像素计数不一致：{summary['pixel_count_unequal_values']} 项。", "",
              "主模型、CR2 与 WWR 配对表逐项核对：",
              f"- 缺失位置变化：{summary['primary_model_comparison']['missing_mismatches']} 项。",
              f"- 非逐位相等字段值：{summary['primary_model_comparison']['unequal_values']} 项。",
              f"- 超出验收容差：{summary['primary_model_comparison']['beyond_tolerance']} 项。",
              f"- 方向翻转：{summary['primary_model_comparison']['direction_flips']} 项；"
              f"p/q 跨越 0.05：{summary['primary_model_comparison']['significance_flips']} 项。", "",
              "统计容差：系数/SE/CI/df 绝对差 1e-8，p/q 绝对差 1e-6；AOI 整数像素计数要求完全相等。"
              "容差内显著性翻转仍单列；浮点值是否逐位相等也单独报告。", "",
              "旧表的 C0 使用空基线标签，读取 Excel 后呈缺失；新表明确使用 C0。"
              "factor_encoding_check.csv 用当前/同 Block 前序 SceneID 验证其语义是否相同；"
              "原始空值差异保留在完整对照表，不静默修改分析输入。", "",
              "完整字段对照见 historical_primary_comparison.csv；阈值、Block1、逐人剔除和补充模型均重新计算。"
              "Stage 3 旧失败、空表、当前未触发及模型字段变化见 historical_supplemental_inventory.csv，"
              "这些状态不计作已验证相同。", "",
              "与当前 EEG 名单相关的 IncludeEEG 字段及共同样本模型独立标记为 current_eeg_common_sample。"
              "历史环境不会恢复旧 EEG 纳入名单；本次不重新计算 EEG，也不替换原正式交付。", "",
              "库环境和计算复现的核对，不自动证明稿件中所有理论解释正确。"]
    (out / "眼动历史环境复算核查.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def refit_statistics(source: Path, out: Path, repo: Path):
    """Reuse sealed same-source Python tables; refit every eye R model."""
    from .fresh import verify_inputs
    from .r_runner import invoke_r
    if out.exists() or out.resolve().is_relative_to(source.resolve()):
        raise StageBlockedError("R refit requires a new sibling run directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise StageBlockedError("R refit requires clean committed code")
    original = json.loads((source / "environment_and_source_proof.json").read_text(encoding="utf-8"))
    complete = json.loads((source / "fresh_calculation_complete.json").read_text(encoding="utf-8"))
    config = json.loads((source / "config.local.json").read_text(encoding="utf-8"))
    environment = validate_analysis(config, repo)
    if environment["python"] != original["environment"]["python"]:
        raise StageBlockedError("Cannot reuse Python tables under a changed Python environment")
    verify_aoi(config, required=True); verify_inputs(complete["input_hashes"])
    source_seals = {}
    for seal in (source / "stage_seals").glob("*.json"):
        source_seals[seal.name] = file_sha256(seal)
        for record in json.loads(seal.read_text(encoding="utf-8"))["files"]:
            if file_sha256(source / record["path"]) != record["sha256"]:
                raise StageBlockedError(f"Cannot reuse changed source artifact: {record['path']}")
    out.mkdir(parents=True)
    for folder in ("01_eye_stage1", "02_eye_stage2", "03_eye_stage3_plan", "04_eye_stage3", "reference", "frozen_aoi"):
        shutil.copytree(source / folder, out / folder)
    for key, folder in [("stage1_dir", "01_eye_stage1"), ("stage2_dir", "02_eye_stage2"), ("stage3_plan_dir", "03_eye_stage3_plan")]:
        config["eye"][key] = str(out / folder)
    config["eye"]["aoi_pixel_lock"] = str(out / "frozen_aoi/aoi_pixel_lock.json")
    config_path = out / "config.local.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    original.update(verification_sha=git_commit(repo), environment=environment,
                    config_sha256=file_sha256(config_path), reused_python_source=str(source),
                    python_calculation_sha=complete["sha"], reused_stage_seals=source_seals)
    (out / "environment_and_source_proof.json").write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ.pop("PAPER_ANALYSIS_R_REUSE_ROOT", None)
    print("Refitting Stage 2 with locked historical R locale", flush=True)
    stage2 = out / "02_eye_stage2"
    invoke_r(config["rscript"], repo / "analysis/r/eye_stage2_analysis.R",
             [str(stage2 / "eye_stage2_model_input.csv"), str(stage2), str(config["eye"]["primary_tracking_threshold"])])
    stage3 = out / "04_eye_stage3"
    metadata = json.loads((stage3 / "run_manifest.json").read_text(encoding="utf-8"))
    triggers = metadata["arguments"]["triggers"]
    print("Refitting Stage 3 with locked historical R locale", flush=True)
    invoke_r(config["rscript"], repo / "analysis/r/eye_stage3_analysis.R",
             [str(stage3 / "eye_stage3_model_input.csv"), str(stage3), ",".join(triggers),
              str(stage3 / "eye_stage3_AOI_model_input.csv"), str(stage3 / "eye_stage3_boundary_model_input.csv"),
              str(config["stage3"]["bootstrap_iterations"])])
    from .eye import _package_r_csv_outputs
    _package_r_csv_outputs(stage2)
    _package_r_csv_outputs(stage3)
    for folder in (stage2, stage3):
        stage_manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        stage_manifest["R_refit"] = {"git_commit": git_commit(repo), "source_python_git_commit": complete["sha"],
                                       "runtime_lock": environment["r_primary"]}
        (folder / "run_manifest.json").write_text(json.dumps(stage_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep original Python stage manifests; record the new R calculation separately.
    (out / "r_refit_provenance.json").write_text(json.dumps({"source_python_sha": complete["sha"],
        "r_calculation_sha": git_commit(repo), "environment": environment, "source_stage_seals": source_seals,
        "reuse": "verified Python trial/AOI/boundary inputs only; all R models refitted"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "stage_seals").mkdir()
    for folder in ("01_eye_stage1", "02_eye_stage2", "03_eye_stage3_plan", "04_eye_stage3"):
        records = [{"path": p.relative_to(out).as_posix(), "sha256": file_sha256(p)} for p in sorted((out/folder).rglob("*")) if p.is_file()]
        (out / "stage_seals" / f"{folder}.json").write_text(json.dumps({"files": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_inputs(complete["input_hashes"])
    (out / "fresh_calculation_complete.json").write_text(json.dumps({"status": "complete", "sha": git_commit(repo),
        "source_python_sha": complete["sha"], "input_hashes": complete["input_hashes"],
        "config_sha256": file_sha256(config_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
