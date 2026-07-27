from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from paper_analysis.eeg.clock_cache import build_eeg_clock_cache, summarize_easy


def test_clock_cache_builds_regular_recording_and_trigger_tables(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    easy = acquisition / "20260101120000_P01.easy"
    event_codes, event_samples = _write_easy(easy, participant="P01")
    metadata = tmp_path / "set_metadata.csv"
    pd.DataFrame([{
        "participant_id": "P01",
        "set_path": str(tmp_path / "P01.set"),
        "history_easy_path": str(easy),
        "srate_hz": 500,
        "n_samples": 200,
        "event_types": "|".join(map(str, event_codes)),
        "event_latencies": "|".join(map(str, event_samples)),
    }]).to_csv(metadata, index=False, encoding="utf-8-sig")

    paths = build_eeg_clock_cache(metadata, acquisition, tmp_path / "cache")
    clock = pd.read_csv(paths["recording_clock"])
    triggers = pd.read_csv(paths["trigger_events"])
    resolution = pd.read_csv(paths["source_resolution"])

    assert clock.loc[0, "first_eeg_epoch_ms"] == 1_700_000_000_000
    assert clock.loc[0, "sample_interval_ms"] == 2
    assert bool(clock.loc[0, "timestamp_regular"])
    assert len(triggers) == len(event_codes)
    assert triggers["trigger_code"].tolist() == event_codes
    assert resolution.loc[resolution["selected"], "selection_reason"].iloc[0] == "validated_set_history_source"


def test_structural_match_beats_larger_incomplete_duplicate(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    valid = acquisition / "20260101120000_P01.easy"
    event_codes, event_samples = _write_easy(valid, participant="P01")
    invalid = acquisition / "20260101130000_P01.easy"
    _write_easy(invalid, participant="P01", sample_count=250, include_events=False)
    metadata = tmp_path / "set_metadata.csv"
    pd.DataFrame([{
        "participant_id": "P01",
        "set_path": str(tmp_path / "P01.set"),
        "history_easy_path": str(tmp_path / "missing" / valid.name),
        "srate_hz": 500,
        "n_samples": 200,
        "event_types": "|".join(map(str, event_codes)),
        "event_latencies": "|".join(map(str, event_samples)),
    }]).to_csv(metadata, index=False, encoding="utf-8-sig")

    paths = build_eeg_clock_cache(metadata, acquisition, tmp_path / "cache")
    clock = pd.read_csv(paths["recording_clock"])
    assert Path(clock.loc[0, "source_easy_path"]).name == valid.name


def test_irregular_timestamp_vector_is_materialized(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    easy = acquisition / "20260101120000_P01.easy"
    event_codes, event_samples = _write_easy(easy, participant="P01", irregular_at=150)
    metadata = tmp_path / "set_metadata.csv"
    pd.DataFrame([{
        "participant_id": "P01",
        "set_path": str(tmp_path / "P01.set"),
        "history_easy_path": str(easy),
        "srate_hz": 500,
        "n_samples": 200,
        "event_types": "|".join(map(str, event_codes)),
        "event_latencies": "|".join(map(str, event_samples)),
    }]).to_csv(metadata, index=False, encoding="utf-8-sig")

    paths = build_eeg_clock_cache(metadata, acquisition, tmp_path / "cache")
    clock = pd.read_csv(paths["recording_clock"])
    irregular_path = Path(clock.loc[0, "irregular_timestamp_path"])
    assert irregular_path.exists()
    assert len(pd.read_csv(irregular_path)) == 200
    summary = summarize_easy(easy)
    assert not summary.timestamp_regular


def test_non_scene_event_differences_do_not_reject_named_clock_source(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    easy = acquisition / "20260101120000_P01.easy"
    event_codes, event_samples = _write_easy(easy, participant="P01")
    set_codes = [1, 2, *event_codes, 9]
    set_samples = [1, 2, *event_samples, 199]
    metadata = tmp_path / "set_metadata.csv"
    pd.DataFrame([{
        "participant_id": "P01",
        "set_path": str(tmp_path / "P01.set"),
        "history_easy_path": str(easy),
        "srate_hz": 500,
        "n_samples": 200,
        "event_types": "|".join(map(str, set_codes)),
        "event_latencies": "|".join(map(str, set_samples)),
    }]).to_csv(metadata, index=False, encoding="utf-8-sig")

    paths = build_eeg_clock_cache(metadata, acquisition, tmp_path / "cache")
    resolution = pd.read_csv(paths["source_resolution"])
    triggers = pd.read_csv(paths["trigger_events"])
    assert bool(resolution.loc[0, "structural_match"])
    assert not bool(resolution.loc[0, "event_sequence_match"])
    assert triggers["trigger_code"].tolist() == set_codes
    assert triggers["sample_index"].tolist() == set_samples


def _write_easy(
    path: Path,
    participant: str,
    sample_count: int = 200,
    include_events: bool = True,
    irregular_at: int | None = None,
) -> tuple[list[int], list[int]]:
    event_codes: list[int] = []
    event_samples: list[int] = []
    if include_events:
        cursor = 3
        for _ in range(12):
            event_codes.extend([7, 8])
            event_samples.extend([cursor, cursor + 5])
            cursor += 12
    event_lookup = dict(zip(event_samples, event_codes, strict=True))
    first_epoch = 1_700_000_000_000
    lines: list[str] = []
    extra = 0
    for sample_index in range(1, sample_count + 1):
        if irregular_at is not None and sample_index == irregular_at:
            extra += 2
        epoch = first_epoch + (sample_index - 1) * 2 + extra
        trigger = event_lookup.get(sample_index, 0)
        fields = ["0"] * 8 + ["0", "0", "0", str(trigger), str(epoch)]
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = (
        "Step Details\n"
        f"Step name: {participant}\n"
        f"StartDate (firstEEGtimestamp): {first_epoch}\n"
        "EEG Settings\n"
        f"Number of records of EEG: {sample_count}\n"
        "EEG sampling rate: 500 Samples/second\n"
        "Number of packets lost: 0\n"
    )
    path.with_suffix(".info").write_text(info, encoding="utf-8")
    return event_codes, event_samples
