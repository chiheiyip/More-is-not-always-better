from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper_analysis.diagnostics.pipeline import formal_order_fatigue_effects
from paper_analysis.stats.canonical import (
    EEG_ORDER_PRIMARY,
    EEG_OUTCOMES,
    _add_order_features,
    _formula,
    _hypothesis_block,
    _normalize_eeg_columns,
    modality_sample_flow,
)


def test_eeg_columns_accept_unprefixed_and_prefixed_inputs() -> None:
    raw = pd.DataFrame({
        outcome.removeprefix("eeg_"): [1.0, 2.0]
        for outcome in EEG_OUTCOMES
    })
    normalized = _normalize_eeg_columns(raw)
    assert set(EEG_OUTCOMES).issubset(normalized.columns)
    assert not any(outcome.removeprefix("eeg_") in normalized for outcome in EEG_OUTCOMES)

    already = pd.DataFrame({outcome: [1.0, 2.0] for outcome in EEG_OUTCOMES})
    pd.testing.assert_frame_equal(_normalize_eeg_columns(already), already)


def test_eeg_duplicate_columns_require_numeric_identity() -> None:
    identical = pd.DataFrame({
        "F_theta": [1.0, np.nan, 3.0],
        "eeg_F_theta": [1.0, np.nan, 3.0],
    })
    result = _normalize_eeg_columns(identical)
    assert "F_theta" not in result
    assert "eeg_F_theta" in result

    conflict = pd.DataFrame({
        "F_theta": [1.0, 2.0],
        "eeg_F_theta": [1.0, 9.0],
    })
    with pytest.raises(ValueError, match="Conflicting EEG columns"):
        _normalize_eeg_columns(conflict)


def test_trial_index_lag_break_and_order_scheme_are_derived_at_scene_grain() -> None:
    rows = []
    for participant, scheme in [("P01", "order1"), ("P02", "order2")]:
        for scene_id in range(1, 13):
            rows.append({
                "participant_id": participant,
                "scene_id": scene_id,
                "block": 1 if scene_id <= 6 else 2,
                "position": (scene_id - 1) % 6 + 1,
                "WWR": [15, 45, 75][(scene_id - 1) % 3],
                "Complexity": scene_id % 2,
                "order_scheme": scheme,
            })
    out = _add_order_features(pd.DataFrame(rows))
    assert out["trial_index"].tolist() == out["scene_id"].tolist()
    first = out.loc[out["scene_id"].eq(1)]
    assert first["previous_WWR"].isna().all()
    assert first["previous_Complexity"].isna().all()
    block2_first = out.loc[out["scene_id"].eq(7)]
    assert block2_first["break_before_trial"].eq(1).all()
    assert block2_first["previous_WWR"].isna().all()
    assert block2_first["previous_Complexity"].isna().all()
    assert set(out["order_scheme"]) == {"order1", "order2"}

    complete = pd.DataFrame(rows)
    filtered = complete.loc[
        ~(
            complete["participant_id"].eq("P01")
            & complete["scene_id"].eq(6)
        )
    ]
    with_reference = _add_order_features(filtered, complete)
    scene7 = with_reference.loc[
        with_reference["participant_id"].eq("P01")
        & with_reference["scene_id"].eq(7)
    ].iloc[0]
    assert pd.isna(scene7["previous_WWR"])


def test_eeg_theta_alpha_and_beta_use_distinct_interpretation_families() -> None:
    for outcome in EEG_ORDER_PRIMARY:
        assert _hypothesis_block(
            outcome, "block", model_family="eeg", scope="eeg_qc_passed"
        ) == "H4_eeg_theta_alpha_order"
    beta = {outcome for outcome in EEG_OUTCOMES if outcome.endswith("_beta")}
    for outcome in beta:
        assert _hypothesis_block(
            outcome, "block", model_family="eeg", scope="eeg_qc_passed"
        ) == "exploratory_not_in_primary_fdr"


def test_primary_time_and_carryover_formulas_have_registered_terms() -> None:
    data = pd.DataFrame({
        "participant_id": ["P01", "P01", "P02", "P02"],
        "y": [1.0, 2.0, 1.5, 2.5],
        "WWR": [15, 45, 75, 15],
        "Complexity": [0, 1, 0, 1],
        "block": [1, 2, 1, 2],
        "position": [1, 1, 2, 2],
        "trial_index": [1, 7, 2, 8],
        "previous_WWR": [np.nan, 15, np.nan, 75],
        "previous_Complexity": [np.nan, 0, np.nan, 1],
        "order_scheme": ["order1", "order1", "order2", "order2"],
        "break_before_trial": [0, 1, 0, 1],
    })
    primary = _formula(data, "y", aoi=False, sensitivity=False, order_policy="primary")
    time = _formula(data, "y", aoi=False, sensitivity=False, order_policy="time_stability")
    carry = _formula(data, "y", aoi=False, sensitivity=False, order_policy="carryover")
    assert "block" in primary and "position" in primary
    assert "trial_index" in time
    assert "C(WWR):trial_index" in time
    assert "C(Complexity):trial_index" in time
    assert "C(previous_WWR)" in carry
    assert "C(previous_Complexity)" in carry
    assert "C(order_scheme)" in carry
    assert "break_before_trial" in carry


def test_sample_flow_uses_modality_specific_and_exact_trimodal_counts() -> None:
    q_keys = pd.DataFrame([
        {"participant_id": f"P{participant:02d}", "scene_id": scene}
        for participant in range(1, 57)
        for scene in range(1, 13)
    ])
    eye_dynamic = q_keys.copy()
    eye_dynamic["blink_rate_per_min"] = 1.0
    eye_aoi = q_keys.loc[q_keys.index.repeat(2)].copy()
    eye_aoi["class_name"] = ["table", "window"] * len(q_keys)
    eye_qc = q_keys.copy()
    eye_qc["analysis_valid_ratio"] = 0.9

    eeg_full = pd.DataFrame([
        {"participant_id": f"P{participant:02d}", "scene_id": scene}
        for participant in range(1, 43)
        for scene in range(1, 13)
    ])
    participant_number = eeg_full["participant_id"].str.removeprefix("P").astype(int)
    eeg_keys = eeg_full.loc[
        ~(participant_number.le(33) & eeg_full["scene_id"].eq(12))
    ].copy()
    assert len(eeg_keys) == 471
    trimodal = eeg_keys.drop(index=eeg_keys.index[:3]).copy()
    trimodal["excluded_from_analysis"] = False

    flow = modality_sample_flow(
        q_keys, eye_aoi, eye_dynamic, eye_qc, eeg_keys, trimodal
    )
    lookup = {
        (row.modality, row.eligibility_policy): (row.n_subjects, row.n_trials)
        for row in flow.itertuples()
    }
    assert lookup[("questionnaire", "questionnaire_available")] == (56, 672)
    assert lookup[("eye", "eye_metric_specific_available_no_eeg_filter")] == (56, 672)
    assert lookup[("eeg", "eeg_qc_passed")] == (42, 471)
    assert lookup[("trimodal_intersection", "descriptive_alignment_only")] == (42, 468)


def test_formal_order_table_never_uses_aoi_expanded_scene_counts() -> None:
    models = pd.DataFrame([{
        "family": "eeg",
        "grain": "eeg_scene",
        "scope": "eeg_qc_passed",
        "hypothesis_block": "H4_eeg_theta_alpha_order",
        "interpretation_tier": "primary",
        "outcome": "eeg_F_theta",
        "term": "block",
        "estimate": -0.1,
        "std_error": 0.03,
        "ci_low": -0.16,
        "ci_high": -0.04,
        "effect_scale": "log_mean",
        "p_value": 0.002,
        "p_fdr_bh": 0.01,
        "n_subjects": 42,
        "n_trials": 471,
        "model_type": "gee_gamma_participant_clustered",
        "status": "fit",
        "formula": "eeg_F_theta ~ block + position",
    }])
    out = formal_order_fatigue_effects(models)
    assert len(out) == 1
    assert int(out.iloc[0]["n_trials"]) == 471
    assert int(out.iloc[0]["n_trials"]) != 1168
