#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


EXPECTED_FLOW = {
    ("questionnaire", "questionnaire_available"): (56, 672),
    ("eye", "eye_metric_specific_available_no_eeg_filter"): (56, 672),
    ("eeg", "eeg_qc_passed"): (42, 471),
    ("trimodal_intersection", "descriptive_alignment_only"): (42, 468),
}
EEG_OUTCOMES = {
    f"eeg_{roi}_{band}"
    for roi in ("F", "P", "O")
    for band in ("theta", "alpha", "beta")
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate promotion of the R1.11/R2.5 order-fatigue evidence package."
    )
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args()
    root = Path(args.outputs_root)
    checks: list[dict[str, object]] = []

    flow = _read(root / "06_models" / "modality_sample_flow.csv", checks)
    if not flow.empty:
        for (modality, policy), expected in EXPECTED_FLOW.items():
            hit = flow.loc[
                flow["modality"].astype(str).eq(modality)
                & flow["eligibility_policy"].astype(str).eq(policy)
            ]
            actual = (
                (int(hit.iloc[0]["n_subjects"]), int(hit.iloc[0]["n_trials"]))
                if not hit.empty else None
            )
            _check(
                checks,
                f"sample_flow:{modality}",
                actual == expected,
                f"expected={expected}, actual={actual}",
            )

    models = _read(root / "06_models" / "model_results.csv", checks)
    if not models.empty:
        eeg = models.loc[
            models["family"].astype(str).eq("eeg")
            & models["scope"].astype(str).eq("eeg_qc_passed")
        ]
        _check(
            checks,
            "nine_eeg_canonical_outcomes",
            set(eeg["outcome"].astype(str)) == EEG_OUTCOMES,
            f"actual={sorted(set(eeg['outcome'].astype(str)))}",
        )
        _check(
            checks,
            "canonical_models_are_clustered_gee",
            models["model_type"].astype(str).str.startswith("gee_").all()
            and ~models["model_type"].astype(str).str.contains("ols", case=False).any(),
            "model_results.csv must contain no OLS model",
        )

    order = _read(root / "06_robustness" / "order_fatigue_effects.csv", checks)
    required_order = {
        "modality", "grain", "scope", "outcome", "order_term", "estimate",
        "std_error", "ci_low", "ci_high", "effect_scale", "p_value",
        "p_fdr_bh", "n_subjects", "n_trials", "model_type", "fit_status",
    }
    if not order.empty:
        _check(
            checks, "formal_order_schema",
            required_order.issubset(order.columns),
            f"missing={sorted(required_order - set(order.columns))}",
        )
        _check(
            checks, "formal_order_modalities",
            {"questionnaire", "eye", "eeg"}.issubset(set(order["modality"].astype(str))),
            f"actual={sorted(set(order['modality'].astype(str)))}",
        )
        _check(
            checks, "formal_order_fdr",
            pd.to_numeric(order["p_fdr_bh"], errors="coerce").notna().all(),
            "every prespecified order estimate requires a BH-FDR q value",
        )
        _check(
            checks, "no_aoi_expanded_scene_n",
            ~pd.to_numeric(order["n_trials"], errors="coerce").eq(1168).any(),
            "scene-level outcomes cannot use the AOI-expanded 1,168-row count",
        )

    stability = _read(root / "06_robustness" / "order_condition_stability.csv", checks)
    if not stability.empty:
        row_types = set(stability.get("row_type", pd.Series(dtype=str)).astype(str))
        terms = stability.get("term", pd.Series(dtype=str)).astype(str)
        _check(
            checks, "controlled_unadjusted_comparison",
            "condition_coefficient_comparison" in row_types,
            f"row_types={sorted(row_types)}",
        )
        _check(
            checks, "wwr_time_interaction",
            terms.str.contains("WWR", regex=False).any()
            and terms.str.contains("trial_index", regex=False).any(),
            "WWR × trial_index term missing",
        )
        _check(
            checks, "complexity_time_interaction",
            terms.str.contains("Complexity", regex=False).any()
            and terms.str.contains("trial_index", regex=False).any(),
            "Complexity × trial_index term missing",
        )

    carry = _read(root / "06_robustness" / "carryover_sensitivity.csv", checks)
    if not carry.empty:
        terms = carry.get("term", pd.Series(dtype=str)).astype(str)
        for fragment in (
            "previous_WWR", "previous_Complexity", "order_scheme",
        ):
            _check(
                checks, f"carryover_term:{fragment}",
                terms.str.contains(fragment, regex=False).any(),
                f"{fragment} missing",
            )
        _check(
            checks, "carryover_models_are_clustered_gee",
            carry.get("model_type", pd.Series(dtype=str)).dropna().astype(str).str.startswith("gee_").all(),
            "carryover model rows must use participant-clustered GEE",
        )

    reviewer = _read(root / "08_reviewer_response" / "reviewer_issue_matrix.csv", checks)
    if not reviewer.empty:
        for issue_id in ("R1.11", "R2.5"):
            hit = reviewer.loc[reviewer["issue_id"].astype(str).eq(issue_id)]
            readiness = str(hit.iloc[0]["response_readiness"]) if not hit.empty else ""
            _check(
                checks, f"reviewer_readiness:{issue_id}",
                readiness == "ready_to_draft_response",
                f"actual={readiness}",
            )

    evidence_path = root / "08_reviewer_response" / "reviewer_order_fatigue_evidence.md"
    evidence = evidence_path.read_text(encoding="utf-8") if evidence_path.exists() else ""
    _check(checks, "reviewer_evidence_present", bool(evidence.strip()), str(evidence_path))
    lowered = evidence.lower()
    _check(
        checks, "bounded_language",
        "prove fatigue" not in lowered and "completely exclude carryover" not in lowered,
        "overstated fatigue/carryover language detected",
    )

    figure_qa = _read(root / "10_figures" / "figure_qa.csv", checks)
    if not figure_qa.empty:
        fig5 = figure_qa.loc[
            figure_qa["figure_id"].astype(str).eq("Fig5_robustness_and_claims")
        ]
        _check(
            checks, "fig5_qa",
            not fig5.empty and fig5["qa_status"].astype(str).eq("pass").all(),
            "Fig. 5 QA did not pass",
        )

    passed = all(bool(row["passed"]) for row in checks)
    payload = {"status": "pass" if passed else "fail", "checks": checks}
    audit = root / "11_audit" / "reviewer_order_evidence_qa.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


def _read(path: Path, checks: list[dict[str, object]]) -> pd.DataFrame:
    exists = path.exists()
    _check(checks, f"file:{path.name}", exists, str(path))
    if not exists:
        return pd.DataFrame()
    try:
        table = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        table = pd.DataFrame()
    _check(checks, f"nonempty:{path.name}", not table.empty, f"rows={len(table)}")
    return table


def _check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


if __name__ == "__main__":
    main()
