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
    figure_contracts = figure_contract_index(figure_contracts_config)
    source_data = source_data_index(figure_contracts)
    data_availability = data_availability_index(data_availability_config)
    data_statement = data_availability_statement(data_availability)
    sample_flow = _read_optional(model_dir / "modality_sample_flow.csv")
    metric_samples = _read_optional(model_dir / "metric_sample_summary.csv")
    reviewer_matrix = reviewer_issue_matrix(
        reviewer_index, models, diagnostics_dir, sample_flow
    )
    reviewer_order_evidence = reviewer_order_fatigue_evidence(
        models, diagnostics_dir, sample_flow, reviewer_matrix
    )
    reviewer_onset_evidence = reviewer_onset_transition_evidence(
        diagnostics_dir, reviewer_matrix
    )
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
        "reviewer_order_fatigue_evidence": write_text(
            reviewer_order_evidence,
            response_dir / "reviewer_order_fatigue_evidence.md",
        ),
        "reviewer_onset_transition_evidence": write_text(
            reviewer_onset_evidence,
            response_dir / "reviewer_onset_transition_evidence.md",
        ),
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
        "n_trials", "formula", "status", "analysis_resolution",
        "analysis_status", "hypothesis_family", "clock_qc_policy",
    ] if c in models.columns]
    return models[keep].copy() if keep else pd.DataFrame()


def claim_strength_table(models: pd.DataFrame, diagnostics_dir: Path) -> pd.DataFrame:
    rows = []
    nonlinear_path = diagnostics_dir / "nonlinear_wwr_sensitivity.csv"
    nonlinear = read_table(nonlinear_path) if nonlinear_path.exists() else pd.DataFrame()
    rows.append({"claim_id": "C1_WWR_NONLINEAR", "support_level": "exploratory", "reason": "Only three WWR levels; use trend/planned contrasts, not optimality language."})
    rows.append({"claim_id": "C2_COMPLEXITY_PROCESSING", "support_level": _term_support(models, "Complexity"), "reason": "Requires convergence between questionnaire, EEG, and eye metrics."})
    rows.append({"claim_id": "C3_EXPERIENCE_MODERATION", "support_level": _term_support(models, "ExperienceGroup"), "reason": "Interpret as moderation only if interaction terms are stable after balance and covariates."})
    rows.append({
        "claim_id": "C4_SCENE_LEVEL_EFFECTS",
        "support_level": _term_support_resolution(models, ["WWR", "Complexity"], "scene_level"),
        "reason": "Co-primary whole-scene average-effect layer.",
    })
    rows.append({
        "claim_id": "C5_SYNCHRONIZED_TIME_DYNAMICS",
        "support_level": _term_support_resolution(models, ["time_norm"], "synchronized_timebin"),
        "reason": "Co-primary within-scene dynamic layer; includes WWR × time and Complexity × time.",
    })
    rows.append({
        "claim_id": "C6_MULTISCALE_CONVERGENCE",
        "support_level": _multiscale_support(models),
        "reason": "Convergence strengthens evidence; disagreement is reported as scale-dependent rather than overridden.",
    })
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
            {"issue_id": "R1.11", "action": "model_block_position_trial_order_and_carryover", "evidence_file": "outputs/08_reviewer_response/reviewer_order_fatigue_evidence.md"},
            {"issue_id": "R2.5", "action": "report_multimodal_fatigue_attention_proxies", "evidence_file": "outputs/08_reviewer_response/reviewer_order_fatigue_evidence.md"},
            {"issue_id": "R1_NONLINEARITY", "action": "soften_claim_and_report_planned_contrasts", "evidence_file": "outputs/06_robustness/nonlinear_wwr_sensitivity.csv"},
        ]
    out = pd.DataFrame(rows)
    if "claim_id" in out.columns:
        out = out.merge(claim_strength[["claim_id", "support_level"]], on="claim_id", how="left")
    return out


def reviewer_issue_matrix(
    reviewer_index: pd.DataFrame,
    models: pd.DataFrame | None = None,
    diagnostics_dir: Path | None = None,
    sample_flow: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    strict_ids = out["issue_id"].astype(str).isin(["R1.11", "R2.5"])
    if strict_ids.any():
        readiness, reason = _order_readiness(
            models if models is not None else pd.DataFrame(),
            diagnostics_dir,
            sample_flow if sample_flow is not None else pd.DataFrame(),
        )
        out.loc[strict_ids, "response_readiness"] = readiness
        out.loc[strict_ids, "readiness_reason"] = reason
    onset_ids = out["issue_id"].astype(str).eq("R1.11_ONSET_TRANSITION")
    if onset_ids.any():
        readiness, reason = _onset_readiness(diagnostics_dir)
        out.loc[onset_ids, "response_readiness"] = readiness
        out.loc[onset_ids, "readiness_reason"] = reason
    return out


def reviewer_onset_transition_evidence(
    diagnostics_dir: Path,
    reviewer_matrix: pd.DataFrame,
) -> str:
    onset_dir = diagnostics_dir / "eeg_onset"
    flow = _read_optional(diagnostics_dir.parent / "04_eeg" / "eeg_onset_qc_sample_flow.csv")
    comparisons = _read_optional(onset_dir / "onset_window_comparisons.csv")
    equivalence = _read_optional(onset_dir / "onset_equivalence_10_vs_15.csv")
    failures = _read_optional(onset_dir / "onset_variant_model_diagnostics.csv")
    bootstrap_failures = _read_optional(onset_dir / "onset_bootstrap_failures.csv")
    readiness = reviewer_matrix.loc[
        reviewer_matrix.get("issue_id", pd.Series(dtype=str)).astype(str).eq(
            "R1.11_ONSET_TRANSITION"
        )
    ]
    sections = [
        "# Reviewer R1.11: EEG scene-onset transition evidence",
        "",
        "The target estimand is sustained-state EEG after scene entry. The first "
        "10 s are removed before PSD and QC; 15 s is the main robustness window, "
        "5 s is a mild sensitivity window, and 0 s is historical audit only.",
        "",
        "A fixed trim can mitigate scene-transition and questionnaire-related "
        "contamination. It cannot prove that residual carryover was eliminated.",
        "",
        "## Readiness gate", "",
        dataframe_to_markdown(readiness) if not readiness.empty else "Onset readiness row missing.",
        "", "## QC sample flow", "",
        dataframe_to_markdown(flow) if not flow.empty else "Onset QC flow missing.",
        "", "## Window comparisons", "",
        dataframe_to_markdown(comparisons) if not comparisons.empty else "Onset comparison table missing.",
        "", "## Formal 10-vs-15 s equivalence", "",
        dataframe_to_markdown(equivalence) if not equivalence.empty else "Equivalence evidence missing.",
        "", "## Model and bootstrap diagnostics", "",
        dataframe_to_markdown(failures) if not failures.empty else "Diagnostics missing.",
        "", "## Bootstrap failures", "",
        dataframe_to_markdown(bootstrap_failures) if not bootstrap_failures.empty else "No bootstrap failure rows.",
    ]
    return "\n".join(sections)


def _onset_readiness(diagnostics_dir: Path | None) -> tuple[str, str]:
    if diagnostics_dir is None:
        return "needs_revision", "diagnostics directory missing"
    onset_dir = diagnostics_dir / "eeg_onset"
    required = {
        "window comparisons": onset_dir / "onset_window_comparisons.csv",
        "formal equivalence": onset_dir / "onset_equivalence_10_vs_15.csv",
        "model diagnostics": onset_dir / "onset_variant_model_diagnostics.csv",
        "readiness audit": onset_dir / "onset_reviewer_readiness.csv",
        "bootstrap failure audit": onset_dir / "onset_bootstrap_failures.csv",
    }
    failures = [
        label for label, path in required.items()
        if not path.is_file()
        or (label != "bootstrap failure audit" and _read_optional(path).empty)
    ]
    audit = _read_optional(required["readiness audit"])
    if not audit.empty:
        complete = audit.loc[
            audit.get("check", pd.Series(dtype=str)).astype(str).eq(
                "reviewer_evidence_complete"
            ), "pass"
        ]
        if complete.empty or not complete.astype(str).str.lower().isin({"true", "1", "yes"}).all():
            failures.append("reviewer evidence gate did not pass")
    if failures:
        return "needs_revision", "; ".join(dict.fromkeys(failures))
    return (
        "ready_to_draft_response",
        "0/5/10/15-s extraction, common QC, 10-vs-15-s equivalence and diagnostics present; bounded wording required",
    )


def reviewer_order_fatigue_evidence(
    models: pd.DataFrame,
    diagnostics_dir: Path,
    sample_flow: pd.DataFrame,
    reviewer_matrix: pd.DataFrame,
) -> str:
    order = _read_optional(diagnostics_dir / "order_fatigue_effects.csv")
    stability = _read_optional(diagnostics_dir / "order_condition_stability.csv")
    carryover = _read_optional(diagnostics_dir / "carryover_sensitivity.csv")
    readiness_rows = reviewer_matrix.loc[
        reviewer_matrix.get("issue_id", pd.Series(dtype=str)).astype(str).isin(
            ["R1.11", "R2.5"]
        ),
        [c for c in ("issue_id", "response_readiness", "readiness_reason") if c in reviewer_matrix],
    ]
    order_cols = [
        c for c in (
            "modality", "outcome", "order_term", "estimate", "ci_low",
            "ci_high", "effect_scale", "p_value", "p_fdr_bh", "n_subjects",
            "n_trials", "model_type", "fit_status",
        ) if c in order
    ]
    time_main = stability.loc[
        stability.get("row_type", pd.Series(index=stability.index, dtype=str)).eq(
            "time_stability_term"
        )
    ].copy() if not stability.empty else pd.DataFrame()
    carry_terms = carryover.loc[
        carryover.get("term", pd.Series(index=carryover.index, dtype=str)).astype(str).str.contains(
            "previous_WWR|previous_Complexity", regex=True
        )
    ].copy() if not carryover.empty else pd.DataFrame()
    stability_cols = [
        c for c in (
            "row_type", "modality", "outcome", "term", "estimate_controlled",
            "ci_low_controlled", "ci_high_controlled", "p_value", "p_fdr_bh",
            "direction_changed", "n_subjects", "n_trials", "model_type",
            "fit_status",
        ) if c in stability
    ]
    carry_cols = [
        c for c in (
            "modality", "outcome", "term", "estimate", "ci_low", "ci_high",
            "p_value", "p_fdr_bh", "n_subjects", "n_trials", "model_type",
            "fit_status",
        ) if c in carry_terms
    ]
    significant_order = int(
        pd.to_numeric(order.get("p_fdr_bh", pd.Series(dtype=float)), errors="coerce")
        .lt(0.05).sum()
    )
    total_order = int(len(order))
    lines = [
        "# Reviewer R1.11 and R2.5: order, fatigue-proxy and carryover evidence",
        "",
        "## Scope and decision boundary",
        "",
        "The analyses below quantify order-related and fatigue/attention-proxy effects. "
        "Block, within-block position, trial index, blink measures and pupil change "
        "are not direct measurements of subjective fatigue. The sensitivity analyses "
        "can assess whether the registered condition effects are stable to plausible "
        "order and carryover controls, but cannot prove that carryover was eliminated.",
        "",
        "All formal models are participant-clustered GEE models. Questionnaire, "
        "eye-tracking and EEG analyses use their modality-specific available samples; "
        "the synchronized three-modality intersection is reserved for aligned analyses.",
        "",
        "## Readiness gate",
        "",
        dataframe_to_markdown(readiness_rows) if not readiness_rows.empty else "No R1.11/R2.5 readiness rows were generated.",
        "",
        "## Sample flow",
        "",
        dataframe_to_markdown(sample_flow) if not sample_flow.empty else "No sample-flow table was generated.",
        "",
        "## Formal block and position estimates",
        "",
        f"{significant_order} of {total_order} prespecified order-term estimates have BH-FDR q < 0.05. "
        "Interpret direction on the stated effect scale and retain the confidence interval.",
        "",
        dataframe_to_markdown(order[order_cols]) if not order.empty else "No formal order estimates were generated.",
        "",
        "## Time stability of WWR and complexity effects",
        "",
        "These models replace the block/position parameterization with trial index and "
        "test WWR × trial-index and Complexity × trial-index terms. A non-significant "
        "interaction is evidence of no detected change, not proof of perfect stability.",
        "",
        dataframe_to_markdown(time_main[stability_cols]) if not time_main.empty else "No time-stability estimates were generated.",
        "",
        "## Controlled versus unadjusted condition coefficients",
        "",
        "The table reports the observed change in condition estimates after adding "
        "block and position. No automatic percentage threshold is used to declare "
        "absence of confounding.",
        "",
        dataframe_to_markdown(
            stability.loc[
                stability.get("row_type", pd.Series(index=stability.index, dtype=str)).eq(
                    "condition_coefficient_comparison"
                ),
                stability_cols,
            ]
        ) if not stability.empty else "No coefficient comparison was generated.",
        "",
        "## Previous-condition carryover sensitivity",
        "",
        "Position 1 of each block is excluded because its within-block lag is "
        "undefined. Previous-condition variables never cross the 120-second break "
        "between blocks.",
        "",
        dataframe_to_markdown(carry_terms[carry_cols]) if not carry_terms.empty else "No carryover estimates were generated.",
        "",
        "## Evidence summary for the response letter",
        "",
        "### R1.11",
        "",
        "We added participant-clustered GEE analyses of EEG theta/alpha order effects, "
        "formal block and position covariates, trial-index interactions, and lagged "
        "previous-condition sensitivity models. The numerical estimates, confidence "
        "intervals, raw p values, BH-FDR q values and actual sample sizes are reported "
        "above. The response should describe detected and undetected effects exactly as "
        "shown, without claiming that time-related confounding was completely excluded.",
        "",
        "### R2.5",
        "",
        "We evaluated questionnaire scores, blink count/rate, angular scan-path rate, "
        "early-reference pupil change, and EEG theta/alpha using modality-specific "
        "samples. These are order/fatigue/attention proxies rather than a direct fatigue "
        "measure; the manuscript should retain that limitation.",
        "",
    ]
    return "\n".join(lines)


def _order_readiness(
    models: pd.DataFrame,
    diagnostics_dir: Path | None,
    sample_flow: pd.DataFrame,
) -> tuple[str, str]:
    failures: list[str] = []
    main_outcomes = {
        *(f"q_S{i}" for i in range(1, 6)),
        *(f"eeg_{roi}_{band}" for roi in ("F", "P", "O") for band in ("theta", "alpha")),
        "blink_count",
        "blink_rate_per_min",
        "angular_scanpath_deg_per_s",
        "pupil_post_early_delta_mm",
    }
    if models.empty:
        failures.append("canonical model table missing")
    else:
        eeg = models.loc[
            models.get("family", pd.Series(index=models.index, dtype=str)).eq("eeg")
            & models.get("scope", pd.Series(index=models.index, dtype=str)).eq("eeg_qc_passed")
        ]
        if eeg["outcome"].nunique() < 9 if "outcome" in eeg else True:
            failures.append("fewer than nine EEG canonical outcomes fitted")
    if diagnostics_dir is None:
        failures.append("diagnostics directory missing")
    else:
        required = {
            "formal order models": diagnostics_dir / "order_fatigue_effects.csv",
            "time interaction models": diagnostics_dir / "order_condition_stability.csv",
            "carryover models": diagnostics_dir / "carryover_sensitivity.csv",
        }
        for label, path in required.items():
            table = _read_optional(path)
            if table.empty:
                failures.append(f"{label} missing or empty")
        order = _read_optional(required["formal order models"])
        if not order.empty and pd.to_numeric(
            order.get("p_fdr_bh", pd.Series(dtype=float)), errors="coerce"
        ).notna().sum() == 0:
            failures.append("formal order FDR values missing")
        for label in ("time interaction models", "carryover models"):
            table = _read_optional(required[label])
            if table.empty:
                continue
            main = table.loc[
                table.get("outcome", pd.Series(index=table.index, dtype=str)).isin(
                    main_outcomes
                )
            ]
            missing = main_outcomes - set(main.get("outcome", pd.Series(dtype=str)))
            if missing:
                failures.append(
                    f"{label} missing main outcomes: {','.join(sorted(missing))}"
                )
            if main.get(
                "fit_status", pd.Series(index=main.index, dtype=str)
            ).astype(str).str.contains("failed|rank_deficient|nonfinite", case=False, regex=True).any():
                failures.append(f"{label} contains failed or unstable main-outcome fits")
        carry = _read_optional(required["carryover models"])
        if not carry.empty:
            carry_terms = carry.get("term", pd.Series(index=carry.index, dtype=str)).astype(str)
            for fragment in (
                "previous_WWR", "previous_Complexity", "order_scheme",
            ):
                if not carry_terms.str.contains(fragment, regex=False).any():
                    failures.append(f"carryover term missing: {fragment}")
    expected = {
        ("questionnaire", "questionnaire_available"): (56, 672),
        ("eeg", "eeg_qc_passed"): (42, 471),
        ("trimodal_intersection", "descriptive_alignment_only"): (42, 468),
    }
    if sample_flow.empty:
        failures.append("sample-flow QA missing")
    else:
        for (modality, policy), counts in expected.items():
            hit = sample_flow.loc[
                sample_flow["modality"].astype(str).eq(modality)
                & sample_flow["eligibility_policy"].astype(str).eq(policy)
            ]
            if hit.empty or (int(hit.iloc[0]["n_subjects"]), int(hit.iloc[0]["n_trials"])) != counts:
                failures.append(
                    f"sample QA mismatch for {modality}: expected {counts[0]}/{counts[1]}"
                )
    if failures:
        return "needs_revision", "; ".join(dict.fromkeys(failures))
    return (
        "ready_to_draft_response",
        "EEG, formal order/FDR, time interaction, carryover, and sample-flow gates passed",
    )


def figure_contract_index(figure_contracts_config: str | Path) -> pd.DataFrame:
    data = _load_json(figure_contracts_config)
    rows = []
    for row in data.get("figures", []):
        out = row.copy()
        out["source_data"] = _join_list(out.get("source_data"))
        out["optional_source_data"] = _join_list(out.get("optional_source_data"))
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
        for column, requirement in (
            ("source_data", "required"),
            ("optional_source_data", "conditional_synchronized_analysis"),
        ):
            for source_file in str(figure.get(column, "")).split(";"):
                source_file = source_file.strip()
                if not source_file:
                    continue
                rows.append({
                    "figure_id": figure.get("figure_id"),
                    "source_file": source_file,
                    "source_role": figure.get("manuscript_role"),
                    "source_requirement": requirement,
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


def _term_support_resolution(
    models: pd.DataFrame,
    term_fragments: list[str],
    resolution: str,
) -> str:
    if models.empty or not {"term", "analysis_resolution"}.issubset(models.columns):
        return "unsupported_no_model"
    hit = models["analysis_resolution"].astype(str).eq(resolution)
    hit &= models["term"].astype(str).apply(
        lambda term: any(fragment in term for fragment in term_fragments)
    )
    if not hit.any():
        return "unsupported_no_term"
    q = pd.to_numeric(models.loc[hit, "p_fdr_bh"], errors="coerce")
    return "moderate" if q.lt(0.05).any() else "bounded_no_fdr_signal"


def _multiscale_support(models: pd.DataFrame) -> str:
    if "analysis_resolution" not in models.columns:
        return "unsupported_no_multiscale_models"
    levels = set(models["analysis_resolution"].dropna().astype(str))
    if not {"scene_level", "synchronized_timebin"}.issubset(levels):
        return "incomplete_one_resolution"
    q = pd.to_numeric(models.get("p_fdr_bh"), errors="coerce")
    significant = models.assign(_q=q).loc[lambda frame: frame["_q"].lt(0.05)]
    supported = set(significant["analysis_resolution"].astype(str))
    if {"scene_level", "synchronized_timebin"}.issubset(supported):
        return "convergent_multiscale_support"
    if supported:
        return "scale_dependent_support"
    return "bounded_no_fdr_signal"


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
