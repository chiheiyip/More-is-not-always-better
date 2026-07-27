"""Teacher-specified staged analysis workflows.

These modules are the authoritative implementation of the 2026-07-26
eye-tracking and EEG analysis instructions.  The older canonical GEE pipeline
remains available as a supplementary compatibility analysis.
"""

from paper_analysis.teacher.contracts import (
    REQUIRED_PARTICIPANT_COLUMNS,
    TRIAL_KEYS,
    build_modality_registry,
    canonicalize_trials,
)
from paper_analysis.teacher.state import StageBlockedError

__all__ = [
    "REQUIRED_PARTICIPANT_COLUMNS",
    "TRIAL_KEYS",
    "StageBlockedError",
    "build_modality_registry",
    "canonicalize_trials",
]
