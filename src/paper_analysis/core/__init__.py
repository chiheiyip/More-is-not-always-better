"""Canonical data model for the paper analysis repository."""

from .schema import DESIGN_COLUMNS, KEYS, PARTICIPANT_COLUMNS
from .tables import (
    add_participant_fields,
    build_trial_index,
    coalesce_design_columns,
    prefix_non_core_columns,
    standardize_eeg_scene,
)

__all__ = [
    "DESIGN_COLUMNS",
    "KEYS",
    "PARTICIPANT_COLUMNS",
    "add_participant_fields",
    "build_trial_index",
    "coalesce_design_columns",
    "prefix_non_core_columns",
    "standardize_eeg_scene",
]
