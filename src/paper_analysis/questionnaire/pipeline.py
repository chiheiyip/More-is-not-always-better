from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paper_analysis.questionnaire.significance import (
    ipq_subject_level_outputs,
    item_level_lmm,
    wwr_polynomial_contrasts,
)
from paper_analysis.utils.io import assert_unique, read_table, require_columns, write_table, write_text
from paper_analysis.utils.markdown import dataframe_to_markdown


S_ITEMS = ["S1", "S2", "S3", "S4", "S5"]
AFFORD4_ITEMS = ["S1", "S2", "S3", "S4"]
B_ITEMS = ["B1", "B2", "B3"]
IPQ_ITEMS = [f"IPQ{i}" for i in range(1, 7)]
DERIVED_ITEMS = ["S5_7", "Bmean", "Afford4", "IPQ_mean"]
QUESTION_ITEMS = S_ITEMS + B_ITEMS + ["IPQ"] + IPQ_ITEMS + DERIVED_ITEMS


def run_questionnaire_pipeline(
    participants_csv: str | Path,
    scene_manifest_csv: str | Path,
    outdir: str | Path = "outputs/02_questionnaire",
    questionnaire_wide: str | Path | None = None,
    questionnaire_long: str | Path | None = None,
    with_significance: bool = True,
    afford4_min_items: int = 3,
    wwr_levels: tuple[float, float, float] = (15.0, 45.0, 75.0),
    skip_reliability: bool = False,
) -> dict[str, Path]:
    if questionnaire_wide is None and questionnaire_long is None:
        raise ValueError("Provide questionnaire_wide or questionnaire_long")
    participants = read_table(participants_csv)
    scene = read_table(scene_manifest_csv)
    if questionnaire_long is not None:
        long = read_table(questionnaire_long)
    else:
        long = wide_to_long(read_table(questionnaire_wide))

    long = standardize_questionnaire_long(long)
    merged = attach_scene_and_participants(long, scene, participants)
    merged = add_questionnaire_composites(merged, afford4_min_items=afford4_min_items)
    qc = questionnaire_qc(merged)
    scale_qc = questionnaire_scale_qc(merged)
    b_qc = b_item_c1_qc(merged)
    reliability = questionnaire_reliability(merged) if not skip_reliability else pd.DataFrame([{"scale": "all", "status": "skipped"}])
    s_desc = item_descriptives(merged, S_ITEMS + [c for c in ["S5_7", "Afford4"] if c in merged.columns])
    b_desc = item_descriptives(merged, B_ITEMS + [c for c in ["Bmean"] if c in merged.columns])
    ipq_desc = ipq_descriptives(merged)
    paper_md = questionnaire_markdown(s_desc, b_desc, reliability, scale_qc, b_qc, ipq_desc)

    outdir = Path(outdir)
    outputs: dict[str, Path] = {
        "questionnaire_long": write_table(merged, outdir / "questionnaire_long.csv"),
        "questionnaire_qc": write_table(qc, outdir / "questionnaire_qc.csv"),
        "questionnaire_scale_qc": write_table(scale_qc, outdir / "questionnaire_scale_qc.csv"),
        "questionnaire_b_item_qc": write_table(b_qc, outdir / "questionnaire_b_item_qc.csv"),
        "questionnaire_reliability": write_table(reliability, outdir / "questionnaire_reliability.csv"),
        "s_items_descriptives": write_table(s_desc, outdir / "s_items_descriptives.csv"),
        "b_items_descriptives": write_table(b_desc, outdir / "b_items_descriptives.csv"),
        "ipq_descriptives": write_table(ipq_desc, outdir / "ipq_descriptives.csv"),
        "questionnaire_paper_tables": write_text(paper_md, outdir / "questionnaire_paper_tables.md"),
    }
    if with_significance:
        model_results, model_diagnostics, marginal_means = item_level_lmm(merged)
        outputs.update({
            "questionnaire_item_model_results": write_table(model_results, outdir / "questionnaire_item_model_results.csv"),
            "questionnaire_item_model_diagnostics": write_table(model_diagnostics, outdir / "questionnaire_item_model_diagnostics.csv"),
            "questionnaire_item_pairwise_or_marginal_means": write_table(marginal_means, outdir / "questionnaire_item_pairwise_or_marginal_means.csv"),
            "questionnaire_wwr_polynomial_contrasts": write_table(wwr_polynomial_contrasts(merged, levels=wwr_levels), outdir / "questionnaire_wwr_polynomial_contrasts.csv"),
        })
        ipq_outputs = ipq_subject_level_outputs(merged)
        for name, df in ipq_outputs.items():
            outputs[name] = write_table(df, outdir / f"{name}.csv")
    return outputs


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    require_columns(wide, ["participant_id"], "questionnaire wide table")
    id_cols = [c for c in wide.columns if not _parse_item_scene(c)]
    rows: list[dict] = []
    for _, row in wide.iterrows():
        base = {c: row[c] for c in id_cols}
        by_scene: dict[int, dict] = {}
        for col in wide.columns:
            parsed = _parse_item_scene(col)
            if not parsed:
                continue
            item, scene_id = parsed
            by_scene.setdefault(scene_id, {}).update({item: row[col]})
        for scene_id, values in by_scene.items():
            rows.append({**base, "scene_id": scene_id, **values})
    return pd.DataFrame(rows)


def standardize_questionnaire_long(long: pd.DataFrame) -> pd.DataFrame:
    require_columns(long, ["participant_id", "scene_id"], "questionnaire long table")
    out = long.copy()
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["scene_id"] = pd.to_numeric(out["scene_id"], errors="coerce").astype("Int64")
    for item in QUESTION_ITEMS:
        if item in out.columns:
            out[item] = pd.to_numeric(out[item], errors="coerce")
    assert_unique(out, ["participant_id", "scene_id"], "questionnaire long table")
    return out


def attach_scene_and_participants(long: pd.DataFrame, scene: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    scene_cols = [c for c in ["participant_id", "scene_id", "WWR", "WWR_numeric", "Complexity", "Cond", "block", "position", "round", "condition_id"] if c in scene.columns]
    part_cols = [c for c in ["participant_id", "ExperienceGroup", "ExperienceRaw", "Experience", "SportFreqGroup", "Gender", "Age", "RecruitmentBatch", "SupplementFlag"] if c in participants.columns]
    out = long.merge(scene[scene_cols], on=["participant_id", "scene_id"], how="left")
    if part_cols:
        out = out.merge(participants[part_cols].drop_duplicates("participant_id"), on="participant_id", how="left")
    return out


def add_questionnaire_composites(df: pd.DataFrame, afford4_min_items: int = 3) -> pd.DataFrame:
    out = df.copy()
    if all(c in out.columns for c in AFFORD4_ITEMS):
        afford = out[AFFORD4_ITEMS].apply(pd.to_numeric, errors="coerce")
        out["Afford4_n_valid"] = afford.notna().sum(axis=1).astype(int)
        out["Afford4"] = afford.mean(axis=1, skipna=True).where(out["Afford4_n_valid"] >= int(afford4_min_items))
    if all(c in out.columns for c in B_ITEMS):
        bvals = out[B_ITEMS].apply(pd.to_numeric, errors="coerce")
        out["Bmean_n_valid"] = bvals.notna().sum(axis=1).astype(int)
        out["Bmean"] = bvals.mean(axis=1, skipna=True).where(out["Bmean_n_valid"] > 0)
    if "S5" in out.columns:
        s5 = pd.to_numeric(out["S5"], errors="coerce")
        if s5.max(skipna=True) > 7 and s5.max(skipna=True) <= 9:
            out["S5_7"] = 1.0 + (s5 - 1.0) * (6.0 / 8.0)
    if all(c in out.columns for c in IPQ_ITEMS):
        subject_ipq = out.groupby("participant_id", dropna=False)[IPQ_ITEMS].mean(numeric_only=True)
        subject_ipq["IPQ_n_valid"] = subject_ipq[IPQ_ITEMS].notna().sum(axis=1)
        subject_ipq["IPQ_mean"] = subject_ipq[IPQ_ITEMS].mean(axis=1, skipna=True)
        out = out.merge(subject_ipq[["IPQ_n_valid", "IPQ_mean"]].reset_index(), on="participant_id", how="left", suffixes=("", "_subject"))
    return out


def questionnaire_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in [c for c in QUESTION_ITEMS + ["Afford4_n_valid", "Bmean_n_valid"] if c in df.columns]:
        values = pd.to_numeric(df[item], errors="coerce")
        rows.append({
            "item": item,
            "n_observations": int(values.notna().sum()),
            "n_missing": int(values.isna().sum()),
            "mean": float(values.mean()) if values.notna().any() else None,
            "sd": float(values.std()) if values.notna().sum() > 1 else None,
            "min": float(values.min()) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
        })
    return pd.DataFrame(rows)


def questionnaire_scale_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in [c for c in S_ITEMS + B_ITEMS + IPQ_ITEMS if c in df.columns]:
        values = pd.to_numeric(df[item], errors="coerce")
        max_v = float(values.max()) if values.notna().any() else np.nan
        min_v = float(values.min()) if values.notna().any() else np.nan
        note = "within_1_7_expected_range"
        if item == "S5" and pd.notna(max_v) and max_v > 7 and max_v <= 9:
            note = "S5_detected_9_point_scale_S5_7_created"
        elif pd.notna(min_v) and (min_v < 1 or max_v > 7):
            note = "outside_1_7_expected_range_review_required"
        rows.append({"item": item, "min": min_v, "max": max_v, "status": note})
    return pd.DataFrame(rows)


def b_item_c1_qc(df: pd.DataFrame) -> pd.DataFrame:
    if not any(c in df.columns for c in B_ITEMS):
        return pd.DataFrame([{"status": "no_b_items"}])
    complexity = _complexity_is_c1(df)
    has_b = df[[c for c in B_ITEMS if c in df.columns]].notna().any(axis=1)
    bad = df.loc[(complexity == False) & has_b].copy()  # noqa: E712 - need nullable comparison semantics
    if bad.empty:
        return pd.DataFrame([{"status": "pass", "n_c0_rows_with_b_values": 0}])
    cols = [c for c in ["participant_id", "scene_id", "WWR", "Complexity", "Cond"] + B_ITEMS if c in bad.columns]
    out = bad[cols].copy()
    out.insert(0, "status", "c0_row_has_b_values_review_required")
    return out


def item_descriptives(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    group_cols = [c for c in ["WWR", "Complexity", "ExperienceGroup"] if c in df.columns]
    rows: list[dict] = []
    for item in [i for i in items if i in df.columns]:
        if group_cols:
            grouped = df.groupby(group_cols, dropna=False)[item]
            for keys, values in grouped:
                keys = keys if isinstance(keys, tuple) else (keys,)
                row = {"item": item, **dict(zip(group_cols, keys))}
                row.update(_summary(values, df.loc[values.index, "participant_id"] if "participant_id" in df.columns else None))
                rows.append(row)
        else:
            rows.append({"item": item, **_summary(df[item], df["participant_id"] if "participant_id" in df.columns else None)})
    return pd.DataFrame(rows)


def ipq_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in IPQ_ITEMS + ["IPQ_mean", "IPQ"] if c in df.columns]
    if not cols:
        return pd.DataFrame([{"status": "no_ipq_items"}])
    subject = df.groupby("participant_id", dropna=False)[cols].mean(numeric_only=True).reset_index() if "participant_id" in df.columns else df[cols]
    rows = []
    for col in cols:
        rows.append({"item": col, **_summary(subject[col], subject["participant_id"] if "participant_id" in subject.columns else None)})
    return pd.DataFrame(rows)


def questionnaire_reliability(df: pd.DataFrame) -> pd.DataFrame:
    scales = [
        ("S1_S4_Afford4_candidate", AFFORD4_ITEMS, "supplementary_construct_only"),
        ("S1_S5_all_s_items", S_ITEMS, "diagnostic_not_default_composite"),
        ("B1_B3_c1_only", B_ITEMS, "supplementary_c1_only"),
        ("IPQ1_IPQ6_subject_level", IPQ_ITEMS, "subject_level_only"),
    ]
    rows = []
    for scale, items, policy in scales:
        available = [c for c in items if c in df.columns]
        if len(available) < 2:
            rows.append({"scale": scale, "items": ",".join(available), "k_items": len(available), "valid_rows": 0, "cronbach_alpha": np.nan, "status": "insufficient_items", "policy": policy})
            continue
        data = df.copy()
        if scale.startswith("B1_B3"):
            data = data.loc[_complexity_is_c1(data).fillna(False)]
        if scale.startswith("IPQ") and "participant_id" in data.columns:
            data = data.groupby("participant_id", dropna=False)[available].mean(numeric_only=True).reset_index()
        x = data[available].apply(pd.to_numeric, errors="coerce").dropna(how="any")
        alpha = _cronbach_alpha(x)
        status = _alpha_status(alpha, len(x), len(available))
        rows.append({"scale": scale, "items": ",".join(available), "k_items": len(available), "valid_rows": int(len(x)), "cronbach_alpha": alpha, "status": status, "policy": policy})
    return pd.DataFrame(rows)


def questionnaire_markdown(s_desc: pd.DataFrame, b_desc: pd.DataFrame, reliability: pd.DataFrame, scale_qc: pd.DataFrame, b_qc: pd.DataFrame, ipq_desc: pd.DataFrame) -> str:
    return "\n".join([
        "# Questionnaire Paper Tables",
        "",
        "## Method Notes",
        "S1-S5 remain item-level primary outcomes. Afford4 is a supplementary S1-S4 candidate construct only when reliability and missingness are acceptable. B items are C1-only supplementary checks. IPQ is treated at participant level and is not interpreted as scene-level WWR/Complexity evidence.",
        "",
        "## S Items",
        dataframe_to_markdown(s_desc) if not s_desc.empty else "No S item data.",
        "",
        "## B Items",
        dataframe_to_markdown(b_desc) if not b_desc.empty else "No B item data.",
        "",
        "## IPQ Subject-Level Descriptives",
        dataframe_to_markdown(ipq_desc) if not ipq_desc.empty else "No IPQ data.",
        "",
        "## Reliability Diagnostics",
        dataframe_to_markdown(reliability) if not reliability.empty else "Reliability diagnostics skipped.",
        "",
        "## Scale QC",
        dataframe_to_markdown(scale_qc) if not scale_qc.empty else "No scale QC rows.",
        "",
        "## B Item C1-Only QC",
        dataframe_to_markdown(b_qc) if not b_qc.empty else "No B item QC rows.",
        "",
    ])


def _summary(values: pd.Series, subject_ids: pd.Series | None = None) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    n_obs = int(valid.shape[0])
    n_subjects = int(subject_ids.loc[numeric.notna()].astype(str).nunique()) if subject_ids is not None and n_obs else n_obs
    ci_low, ci_high = _ci95(valid)
    return {
        "n": n_subjects,
        "n_subjects": n_subjects,
        "n_obs": n_obs,
        "mean": float(valid.mean()) if n_obs else None,
        "sd": float(valid.std(ddof=1)) if n_obs > 1 else None,
        "median": float(valid.median()) if n_obs else None,
        "min": float(valid.min()) if n_obs else None,
        "max": float(valid.max()) if n_obs else None,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "skewness": float(stats.skew(valid, bias=False)) if n_obs > 2 else None,
        "kurtosis": float(stats.kurtosis(valid, fisher=True, bias=False)) if n_obs > 3 else None,
        "shapiro_p": _shapiro_p(valid),
    }


def _ci95(values: pd.Series) -> tuple[float | None, float | None]:
    n = int(values.shape[0])
    if n < 2:
        return None, None
    mean = float(values.mean())
    se = float(stats.sem(values, nan_policy="omit"))
    if not math.isfinite(se):
        return None, None
    delta = float(stats.t.ppf(0.975, n - 1) * se)
    return mean - delta, mean + delta


def _shapiro_p(values: pd.Series) -> float | None:
    n = int(values.shape[0])
    if n < 3 or n > 5000:
        return None
    try:
        return float(stats.shapiro(values).pvalue)
    except Exception:
        return None


def _cronbach_alpha(items: pd.DataFrame) -> float:
    if items.shape[0] < 3 or items.shape[1] < 2:
        return np.nan
    variances = items.var(axis=0, ddof=1)
    total = items.sum(axis=1)
    total_var = total.var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return np.nan
    k = items.shape[1]
    return float((k / (k - 1)) * (1 - variances.sum() / total_var))


def _alpha_status(alpha: float, n_rows: int, k_items: int) -> str:
    if k_items < 2 or n_rows < 3 or not np.isfinite(alpha):
        return "insufficient"
    if alpha >= 0.70:
        return "acceptable_diagnostic"
    if alpha >= 0.60:
        return "check_before_composite"
    return "low_do_not_force_composite"


def _complexity_is_c1(df: pd.DataFrame) -> pd.Series:
    if "Complexity" in df.columns:
        comp = df["Complexity"].astype(str).str.strip().str.lower()
        return comp.isin({"1", "1.0", "c1", "high", "true"})
    if "Cond" in df.columns:
        return df["Cond"].astype(str).str.strip().str.upper().eq("C1")
    return pd.Series(pd.NA, index=df.index, dtype="boolean")


def _parse_item_scene(name: str) -> tuple[str, int] | None:
    text = str(name).strip()
    item_pat = r"S[1-5]|B[1-3]|IPQ[1-6]|IPQ"
    match = re.match(rf"^({item_pat})[_\-\. ]?(?:scene)?(\d{{1,2}})$", text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), int(match.group(2))
    match = re.match(rf"^(?:scene)?(\d{{1,2}})[_\-\. ]?({item_pat})$", text, re.IGNORECASE)
    if match:
        return match.group(2).upper(), int(match.group(1))
    return None
