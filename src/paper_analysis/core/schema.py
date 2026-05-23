from __future__ import annotations

KEYS = ["participant_id", "scene_id"]

DESIGN_COLUMNS = [
    "condition_id",
    "WWR",
    "WWR_numeric",
    "Cond",
    "Complexity",
    "block",
    "position",
    "round",
    "participant_order",
    "order_code",
]

PARTICIPANT_COLUMNS = [
    "participant_id",
    "eeg_subject_id",
    "eye_subject_id",
    "ExperienceRaw",
    "Experience",
    "ExperienceGroup",
    "SportFreq",
    "Gender",
    "Age",
    "RecruitmentBatch",
    "SupplementFlag",
    "exclude",
    "ExcludeReason",
]

SCENE_PATH_COLUMNS = ["eye_csv_path", "aoi_json_path"]

EEG_TIMING_COLUMNS = ["view_start_s", "view_end_s", "view_dur_s", "dur_s", "duration_s"]

CORE_COLUMNS = KEYS + DESIGN_COLUMNS + PARTICIPANT_COLUMNS + SCENE_PATH_COLUMNS
