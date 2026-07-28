from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from paper_analysis.utils.coding import experience_group
from paper_analysis.utils.io import is_truthy, read_table


TRIAL_KEYS = ["Participant", "GlobalTrialOrder"]
REQUIRED_PARTICIPANT_COLUMNS = [
    "Participant",
    "Gender",
    "ExperienceRaw",
    "ExperienceGroup",
    "OrderGroup",
    "IncludeEyeCandidate",
    "IncludeEEGValid",
    "IncludeQuestionnaireValid",
]
REQUIRED_TRIAL_COLUMNS = [
    "Participant",
    "OrderGroup",
    "Block",
    "PositionWithinBlock",
    "GlobalTrialOrder",
    "SceneID",
    "WWR",
    "Complexity",
    "CSVFile",
]

PARTICIPANT_ALIASES = {
    "participant_id": "Participant",
    "subject_id": "Participant",
    "GenderRaw": "Gender",
    "Experience": "ExperienceRaw",
    "order_scheme": "OrderGroup",
    "Order": "OrderGroup",
    "has_eye_raw": "IncludeEyeCandidate",
    "has_questionnaire": "IncludeQuestionnaireValid",
}

TRIAL_ALIASES = {
    "participant_id": "Participant",
    "order_scheme": "OrderGroup",
    "block": "Block",
    "block_id": "Block",
    "round": "Block",
    "position": "PositionWithinBlock",
    "cycle_in_block": "PositionWithinBlock",
    "scene_id": "GlobalTrialOrder",
    "global_trial_order": "GlobalTrialOrder",
    "condition_id": "ConditionID",
    "eye_csv_path": "CSVFile",
}


def normalize_order_group(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    compact = " ".join(text.split())
    if compact in {"1", "order 1", "order1"}:
        return "order1"
    if compact in {"2", "order 2", "order2"}:
        return "order2"
    if compact in {"new order 2", "neworder2", "new order2"}:
        return "new order2"
    return str(value or "").strip()


def normalize_complexity(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"0", "0.0", "C0", "LOW"}:
        return "C0"
    if text in {"1", "1.0", "C1", "HIGH"}:
        return "C1"
    return text


def normalize_wwr(value: object) -> str:
    text = str(value or "").strip().upper().replace("WWR", "").replace("W", "")
    try:
        numeric = int(float(text))
    except (TypeError, ValueError):
        return str(value or "").strip()
    return f"WWR{numeric}"


def _rename_known_columns(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    # Legacy tables may contain several synonymous columns at the same time
    # (subject_id + participant_id, block_id + block + round, etc.). A bulk
    # rename would create duplicate column labels, so coalesce and validate.
    for target in dict.fromkeys(aliases.values()):
        sources = [
            source for source, mapped in aliases.items()
            if mapped == target and source in out.columns
        ]
        if target in out.columns:
            sources.insert(0, target)
        if not sources:
            continue
        combined = out[sources[0]].copy()
        for source in sources[1:]:
            candidate = out[source]
            overlap = combined.notna() & candidate.notna()
            if overlap.any():
                left_text = combined.loc[overlap].astype(str).str.strip()
                right_text = candidate.loc[overlap].astype(str).str.strip()
                equal = left_text.eq(right_text)
                left_num = pd.to_numeric(left_text, errors="coerce")
                right_num = pd.to_numeric(right_text, errors="coerce")
                equal |= left_num.notna() & right_num.notna() & left_num.eq(right_num)
                if not bool(equal.all()):
                    examples = pd.DataFrame({
                        "left": left_text.loc[~equal],
                        "right": right_text.loc[~equal],
                    }).head(3).to_dict("records")
                    raise ValueError(
                        f"conflicting aliases for {target}: "
                        f"{sources[0]} vs {source}; examples={examples}"
                    )
            combined = combined.combine_first(candidate)
        out[target] = combined
        drop = [source for source in sources if source != target]
        if drop:
            out = out.drop(columns=drop)
    return out


def normalize_participant_information(frame: pd.DataFrame) -> pd.DataFrame:
    out = _rename_known_columns(frame, PARTICIPANT_ALIASES)
    if "Participant" not in out:
        raise ValueError("participant_information must contain Participant")
    out["Participant"] = out["Participant"].astype(str).str.strip()
    if "OrderGroup" in out:
        out["OrderGroup"] = out["OrderGroup"].map(normalize_order_group)
    if "ExperienceGroup" not in out and "ExperienceRaw" in out:
        out["ExperienceGroup"] = out["ExperienceRaw"].map(experience_group)
    elif "ExperienceGroup" in out:
        out["ExperienceGroup"] = out["ExperienceGroup"].map(experience_group)
    for column in (
        "IncludeEyeCandidate",
        "IncludeEEGValid",
        "IncludeQuestionnaireValid",
    ):
        if column not in out:
            out[column] = False
        out[column] = out[column].map(is_truthy)
    for column in REQUIRED_PARTICIPANT_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out


def build_modality_registry(
    participant_information: pd.DataFrame | str | Path | None = None,
    *,
    eye_participants: Iterable[object] = (),
    eeg_participants: Iterable[object] = (),
    questionnaire_participants: Iterable[object] = (),
) -> pd.DataFrame:
    """Build a union registry; modality absence never removes another modality."""
    if participant_information is not None:
        base_raw = (
            read_table(participant_information)
            if isinstance(participant_information, (str, Path))
            else participant_information.copy()
        )
        explicit_flags = {
            column for column in (
                "IncludeEyeCandidate",
                "IncludeEEGValid",
                "IncludeQuestionnaireValid",
            )
            if column in base_raw.columns
        }
        base = normalize_participant_information(base_raw)
    else:
        explicit_flags = set()
        base = pd.DataFrame(columns=REQUIRED_PARTICIPANT_COLUMNS)
    base_participants = set(base.get("Participant", pd.Series(dtype=str)).astype(str))
    eye = {str(value).strip() for value in eye_participants if str(value).strip()}
    eeg = {str(value).strip() for value in eeg_participants if str(value).strip()}
    questionnaire = {
        str(value).strip() for value in questionnaire_participants if str(value).strip()
    }
    union = sorted(
        set(base.get("Participant", pd.Series(dtype=str)).astype(str)) | eye | eeg | questionnaire
    )
    if not union:
        return pd.DataFrame(columns=REQUIRED_PARTICIPANT_COLUMNS)
    registry = pd.DataFrame({"Participant": union})
    if not base.empty:
        base = base.drop_duplicates("Participant", keep="last")
        registry = registry.merge(base, on="Participant", how="left")
    for column, members in (
        ("IncludeEyeCandidate", eye),
        ("IncludeEEGValid", eeg),
        ("IncludeQuestionnaireValid", questionnaire),
    ):
        existing = (
            registry[column].map(is_truthy)
            if column in registry
            else pd.Series(False, index=registry.index)
        )
        inferred = registry["Participant"].isin(members)
        if column in explicit_flags:
            inferred &= ~registry["Participant"].isin(base_participants)
        registry[column] = existing | inferred
    for column in REQUIRED_PARTICIPANT_COLUMNS:
        if column not in registry:
            registry[column] = pd.NA
    return registry[REQUIRED_PARTICIPANT_COLUMNS + [
        c for c in registry.columns if c not in REQUIRED_PARTICIPANT_COLUMNS
    ]]


def canonicalize_trials(frame: pd.DataFrame, *, require_complete: bool = False) -> pd.DataFrame:
    out = _rename_known_columns(frame, TRIAL_ALIASES)
    if "GlobalTrialOrder" not in out and {"Block", "PositionWithinBlock"}.issubset(out):
        out["GlobalTrialOrder"] = (
            (pd.to_numeric(out["Block"], errors="coerce") - 1) * 6
            + pd.to_numeric(out["PositionWithinBlock"], errors="coerce")
        )
    if "SceneID" not in out and "GlobalTrialOrder" in out:
        out["SceneID"] = out["GlobalTrialOrder"]
    required = (
        REQUIRED_TRIAL_COLUMNS
        if require_complete
        else ["Participant", "Block", "PositionWithinBlock", "GlobalTrialOrder"]
    )
    missing = sorted(set(required) - set(out.columns))
    if missing:
        raise ValueError(f"trial mapping missing columns: {missing}")
    out["Participant"] = out["Participant"].astype(str).str.strip()
    for column in ("Block", "PositionWithinBlock", "GlobalTrialOrder", "SceneID"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    if "OrderGroup" in out:
        out["OrderGroup"] = out["OrderGroup"].map(normalize_order_group)
    if "WWR" in out:
        out["WWR"] = out["WWR"].map(normalize_wwr)
    if "Complexity" in out:
        out["Complexity"] = out["Complexity"].map(normalize_complexity)
    out["PositionWithinBlockCentered"] = (
        pd.to_numeric(out["PositionWithinBlock"], errors="coerce") - 3.5
    )
    out = out.sort_values(
        ["Participant", "Block", "PositionWithinBlock"], kind="stable"
    ).reset_index(drop=True)
    group = out.groupby(["Participant", "Block"], sort=False, dropna=False)
    out["PreviousWWR"] = group["WWR"].shift(1) if "WWR" in out else pd.NA
    out["PreviousComplexity"] = (
        group["Complexity"].shift(1) if "Complexity" in out else pd.NA
    )
    first = pd.to_numeric(out["PositionWithinBlock"], errors="coerce").eq(1)
    out.loc[first, ["PreviousWWR", "PreviousComplexity"]] = pd.NA
    return out


def trial_contract_audit(frame: pd.DataFrame) -> pd.DataFrame:
    trials = canonicalize_trials(frame)
    rows: list[dict] = []
    for participant, sub in trials.groupby("Participant", sort=True):
        duplicate_orders = int(sub["GlobalTrialOrder"].duplicated(keep=False).sum())
        blocks = sorted(pd.to_numeric(sub["Block"], errors="coerce").dropna().unique())
        rows.append({
            "Participant": participant,
            "TrialCount": int(len(sub)),
            "Block1TrialCount": int(pd.to_numeric(sub["Block"], errors="coerce").eq(1).sum()),
            "Block2TrialCount": int(pd.to_numeric(sub["Block"], errors="coerce").eq(2).sum()),
            "BlockValues": ",".join(str(int(v)) for v in blocks),
            "DuplicateGlobalTrialRows": duplicate_orders,
            "MissingGlobalTrialOrders": ",".join(
                str(v) for v in sorted(set(range(1, 13)) - set(
                    pd.to_numeric(sub["GlobalTrialOrder"], errors="coerce").dropna().astype(int)
                ))
            ),
            "ContractPass": (
                len(sub) == 12
                and blocks == [1, 2]
                and duplicate_orders == 0
                and set(pd.to_numeric(sub["GlobalTrialOrder"], errors="coerce").dropna().astype(int))
                == set(range(1, 13))
            ),
        })
    return pd.DataFrame(rows)
