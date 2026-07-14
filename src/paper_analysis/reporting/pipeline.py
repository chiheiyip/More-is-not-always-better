from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_analysis.utils.io import read_table, write_table, write_text
from paper_analysis.utils.markdown import dataframe_to_markdown
from paper_analysis.reporting.experiment_results import build_experiment_result_package


def build_paper_outputs(
    model_results_csv: str | Path,
    diagnostics_dir: str | Path,
    reviewer_map: str | Path = "configs/reviewer_response_map.json",
    data_availability_config: str | Path = "configs/data_availability.json",
    figure_contracts_config: str | Path = "configs/figure_contracts.json",
    outdir: str | Path = "outputs/07_paper_tables",
) -> dict[str, Path]:
    models = read_table(model_results_csv)
    model_dir = Path(model_results_csv).parent
    diagnostics_dir = Path(diagnostics_dir)
    outdir = Path(outdir)
    response_dir = outdir.parent / "08_reviewer_response"
    data_package_dir = outdir.parent / "09_data_package"
    paper_tables = build_model_table(models)
    claim_strength = claim_strength_table(models, diagnostics_dir)
    reviewer_index = reviewer_response_index(reviewer_map, claim_strength)
    reviewer_matrix = reviewer_issue_matrix(reviewer_index)
    figure_contracts = figure_contract_index(figure_contracts_config)
    source_data = source_data_index(figure_contracts)
    data_availability = data_availability_index(data_availability_config)
    data_statement = data_availability_statement(data_availability)
    sample_flow = _read_optional(model_dir / "modality_sample_flow.csv")
    metric_samples = _read_optional(model_dir / "metric_sample_summary.csv")
    summary = paper_summary_markdown(paper_tables, claim_strength, figure_contracts, data_availability, sample_flow, metric_samples)
    outputs = {
        "table_model_results": write_table(paper_tables, outdir / "table_model_results.csv"),
        "claim_strength_table": write_table(claim_strength, outdir / "claim_strength_table.csv"),
        "figure_contracts_index": write_table(figure_contracts, outdir / "figure_contracts_index.csv"),
        "source_data_index": write_table(source_data, outdir / "source_data_index.csv"),
        "paper_results_summary": write_text(summary, outdir / "paper_results_summary.md"),
        "modality_sample_flow": write_table(sample_flow, outdir / "modality_sample_flow.csv"),
        "eye_metric_sample_summary": write_table(metric_samples.loc[metric_samples.get("grain", pd.Series(dtype=str)).astype(str).str.startswith("eye")].copy() if not metric_samples.empty else metric_samples, outdir / "eye_metric_sample_summary.csv"),
        "eye_analysis_scope_note": write_text(_eye_scope_note(sample_flow), outdir / "eye_analysis_scope_note.md"),
        "response_evidence_index": write_table(reviewer_index, response_dir / "response_evidence_index.csv"),
        "reviewer_issue_matrix": write_table(reviewer_matrix, response_dir / "reviewer_issue_matrix.csv"),
        "data_availability_index": write_table(data_availability, data_package_dir / "data_availability_index.csv"),
        "data_availability_statement": write_text(data_statement, data_package_dir / "data_availability_statement.md"),
    }
    outputs.update(build_experiment_result_package(outputs_root=outdir.parent, outdir=outdir, audit_dir=outdir.parent / "11_audit"))
    return outputs


def build_model_table(models: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in [
        "grain", "family", "scope", "hypothesis_block", "interpretation_tier",
        "outcome", "term", "estimate", "std_error", "effect_scale", "p_value",
        "p_fdr_bh", "ci_low", "ci_high", "model_type", "n_obs", "n_subjects",
        "n_trials", "formula", "status",
    ] if c in models.columns]
    return models[keep].copy() if keep else pd.DataFrame()


def claim_strength_table(models: pd.DataFrame, diagnostics_dir: Path) -> pd.DataFrame:
    rows = []
    nonlinear_path = diagnostics_dir / "nonlinear_wwr_sensitivity.csv"
    nonlinear = read_table(nonlinear_path) if nonlinear_path.exists() else pd.DataFrame()
    rows.append({"claim_id": "C1_WWR_NONLINEAR", "support_level": "exploratory", "reason": "Only three WWR levels; use trend/planned contrasts, not optimality language."})
    rows.append({"claim_id": "C2_COMPLEXITY_PROCESSING", "support_level": _term_support(models, "Complexity"), "reason": "Requires convergence between questionnaire, EEG, and eye metrics."})
    rows.append({"claim_id": "C3_EXPERIENCE_MODERATION", "support_level": _term_support(models, "ExperienceGroup"), "reason": "Interpret as moderation only if interaction terms are stable after balance and covariates."})
    if not nonlinear.empty:
        rows.append({"claim_id": "C1_WWR_NONLINEAR_DIAGNOSTIC", "support_level": "see_diagnostics", "reason": str(nonlinear.get("claim_strength", pd.Series([""])).iloc[0])})
    return pd.DataFrame(rows)


def reviewer_response_index(reviewer_map: str | Path, claim_strength: pd.DataFrame) -> pd.DataFrame:
    path = Path(reviewer_map)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("reviewer_issues", [])
    else:
        rows = []
    if not rows:
        rows = [
            {"issue_id": "R1_SAMPLE_BALANCE", "action": "supplement_low_experience_and_report_balance", "evidence_file": "outputs/01_sample_qc/group_balance_before_after.csv"},
            {"issue_id": "R1_ORDER_FATIGUE", "action": "model_block_position_trial_order", "evidence_file": "outputs/06_robustness/order_fatigue_effects.csv"},
            {"issue_id": "R1_NONLINEARITY", "action": "soften_claim_and_report_planned_contrasts", "evidence_file": "outputs/06_robustness/nonlinear_wwr_sensitivity.csv"},
        ]
    out = pd.DataFrame(rows)
    if "claim_id" in out.columns:
        out = out.merge(claim_strength[["claim_id", "support_level"]], on="claim_id", how="left")
    return out


def reviewer_issue_matrix(reviewer_index: pd.DataFrame) -> pd.DataFrame:
    out = reviewer_index.copy()
    if out.empty:
        return pd.DataFrame(columns=["issue_id", "reviewer_concern", "action", "evidence_file", "response_readiness"])
    if "evidence_file" not in out.columns:
        out["evidence_file"] = ""
    out["evidence_file"] = out["evidence_file"].fillna("")
    out["evidence_mapped"] = out["evidence_file"].astype(str).str.len() > 0
    out["response_readiness"] = out["evidence_mapped"].map({True: "ready_to_draft_response", False: "AUTHOR_INPUT_NEEDED"})
    if "support_level" in out.columns:
        weak = out["support_level"].fillna("").astype(str).str.contains("unsupported|exploratory", case=False, regex=True)
        out.loc[weak, "response_readiness"] = "draft_with_bounded_claim"
    return out


def figure_contract_index(figure_contracts_config: str | Path) -> pd.DataFrame:
    data = _load_json(figure_contracts_config)
    rows = []
    for row in data.get("figures", []):
        out = row.copy()
        out["source_data"] = _join_list(out.get("source_data"))
        out["export_targets"] = _join_list(out.get("export_targets"))
        out["panel_map"] = _join_panel_map(out.get("panel_map"))
        out["statistics_note"] = out.get("statistics_note", "")
        out["image_integrity_note"] = out.get("image_integrity_note", "")
        out["contract_status"] = data.get("figure_contract_status", "")
        out["backend_policy"] = data.get("backend_policy", "")
        out["journal_export_contract"] = json.dumps(data.get("journal_export_contract", {}), ensure_ascii=False)
        rows.append(out)
    return pd.DataFrame(rows)


def source_data_index(figure_contracts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if figure_contracts.empty or "source_data" not in figure_contracts.columns:
        return pd.DataFrame(columns=["figure_id", "source_file", "source_role"])
    for _, figure in figure_contracts.iterrows():
        for source_file in str(figure.get("source_data", "")).split(";"):
            source_file = source_file.strip()
            if not source_file:
                continue
            rows.append({
                "figure_id": figure.get("figure_id"),
                "source_file": source_file,
                "source_role": figure.get("manuscript_role"),
                "core_conclusion": figure.get("core_conclusion"),
            })
    return pd.DataFrame(rows)


def data_availability_index(data_availability_config: str | Path) -> pd.DataFrame:
    data = _load_json(data_availability_config)
    rows = []
    for row in data.get("datasets", []):
        out = row.copy()
        out["derived_outputs"] = _join_list(out.get("derived_outputs"))
        out["statement_status"] = data.get("statement_status", "")
        out["repository_policy"] = data.get("repository_policy", "")
        rows.append(out)
    return pd.DataFrame(rows)


def data_availability_statement(data_availability: pd.DataFrame) -> str:
    if data_availability.empty:
        return "\n".join(["# Data Availability", "", "AUTHOR_INPUT_NEEDED"])
    public = data_availability.loc[data_availability["access_route"].astype(str).str.contains("public", case=False, na=False), "label"].tolist()
    restricted = data_availability.loc[data_availability["access_route"].astype(str).str.contains("restricted|controlled", case=False, regex=True, na=False), "label"].tolist()
    placeholders = data_availability.loc[
        data_availability["identifier"].astype(str).eq("AUTHOR_INPUT_NEEDED") |
        data_availability["repository_target"].astype(str).eq("AUTHOR_INPUT_NEEDED")
    ]
    lines = [
        "# Data Availability",
        "",
        "Processed analysis tables, figure source data, and code required to reproduce the reported analyses will be deposited in a DOI-backed public repository before submission. Repository name and identifier are AUTHOR_INPUT_NEEDED.",
    ]
    if public:
        lines.extend(["", "Public/de-identified data package:", ", ".join(public) + "."])
    if restricted:
        lines.extend([
            "",
            "Restricted or de-identification-dependent source data:",
            ", ".join(restricted) + ". Access conditions, consent constraints, and request review route are AUTHOR_INPUT_NEEDED.",
        ])
    if not placeholders.empty:
        lines.extend(["", "Unresolved repository fields:"])
        lines.extend(f"- {row.dataset_id}: repository_target={row.repository_target}, identifier={row.identifier}" for row in placeholders.itertuples())
    return "\n".join(lines) + "\n"


def paper_summary_markdown(
    table: pd.DataFrame,
    claim_strength: pd.DataFrame,
    figure_contracts: pd.DataFrame | None = None,
    data_availability: pd.DataFrame | None = None,
    sample_flow: pd.DataFrame | None = None,
    metric_samples: pd.DataFrame | None = None,
) -> str:
    return "\n".join([
        "# Paper Results Summary",
        "",
        "## Model Results",
        dataframe_to_markdown(table) if not table.empty else "No model results available.",
        "",
        "## Modality-specific samples",
        dataframe_to_markdown(sample_flow) if sample_flow is not None and not sample_flow.empty else "No modality sample flow available.",
        "",
        "Eye-tracking inference uses eye-specific metric availability and does not inherit EEG exclusions. The trimodal intersection is used only for aligned multimodal analyses.",
        "",
        "## Eye metric availability",
        dataframe_to_markdown(metric_samples.loc[metric_samples.get("grain", pd.Series(dtype=str)).astype(str).str.startswith("eye")]) if metric_samples is not None and not metric_samples.empty else "No eye metric availability table available.",
        "",
        "## Claim Strength",
        dataframe_to_markdown(claim_strength) if not claim_strength.empty else "No claim strength diagnostics available.",
        "",
        "## Figure Contracts",
        dataframe_to_markdown(figure_contracts) if figure_contracts is not None and not figure_contracts.empty else "No figure contracts available.",
        "",
        "## Data Availability",
        dataframe_to_markdown(data_availability) if data_availability is not None and not data_availability.empty else "No data availability index available.",
        "",
    ])


def _term_support(models: pd.DataFrame, term_fragment: str) -> str:
    if models.empty or "term" not in models.columns:
        return "unsupported_no_model"
    hit = models["term"].astype(str).str.contains(term_fragment, case=False, regex=False)
    if not hit.any():
        return "unsupported_no_term"
    p_col = "p_fdr_bh" if "p_fdr_bh" in models.columns else "p_value"
    sig = pd.to_numeric(models.loc[hit, p_col], errors="coerce").lt(0.05).any()
    return "moderate" if sig else "exploratory"


def _read_optional(path: Path) -> pd.DataFrame:
    try:
        return read_table(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _eye_scope_note(sample_flow: pd.DataFrame) -> str:
    rows = [
        "# Eye-tracking analysis scope",
        "",
        "The canonical eye-tracking analysis uses every eye trial for which the metric-specific inputs are available. It does not inherit EEG participant or trial exclusions.",
        "",
        "Validity-coordinate thresholds of 50%, 60%, 70%, and 80% are sensitivity analyses rather than primary-analysis gates. Pupil results use a scene-early reference, are luminance-confounded, and remain exploratory.",
    ]
    if not sample_flow.empty:
        rows.extend(["", "## Sample flow", "", dataframe_to_markdown(sample_flow)])
    return "\n".join(rows) + "\n"


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _join_list(value: object) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _join_panel_map(value: object) -> str:
    if not isinstance(value, list):
        return "" if value is None else str(value)
    rows = []
    for panel in value:
        if isinstance(panel, dict):
            rows.append(f"{panel.get('panel_id', '')}:{panel.get('claim_role', '')}:{panel.get('source_data', '')}")
        else:
            rows.append(str(panel))
    return ";".join(rows)
