from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from paper_analysis.utils.io import write_table


EVENT_SEPARATOR = "|"
EXPECTED_VIEW_COUNT = 12


@dataclass(frozen=True)
class EasySummary:
    path: Path
    n_samples: int
    first_epoch_ms: int
    last_epoch_ms: int
    sample_interval_ms: float
    timestamp_regular: bool
    trigger_codes: tuple[int, ...]
    trigger_samples: tuple[int, ...]
    trigger_epochs_ms: tuple[int, ...]
    view_count: int
    sha256: str
    info_first_epoch_ms: int | None
    info_srate_hz: float | None
    info_n_samples: int | None
    info_packets_lost: int | None


def build_eeg_clock_cache(
    set_metadata_csv: str | Path,
    acquisition_root: str | Path,
    cache_root: str | Path,
    timezone_name: str = "Asia/Shanghai",
    expected_view_count: int = EXPECTED_VIEW_COUNT,
    participants: Iterable[str] | None = None,
    allow_partial: bool = False,
) -> dict[str, Path]:
    metadata = pd.read_csv(set_metadata_csv, encoding="utf-8-sig")
    required = {"participant_id", "set_path", "history_easy_path", "srate_hz", "n_samples", "event_types", "event_latencies"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"EEG set metadata missing columns: {sorted(missing)}")
    if participants:
        selected = {str(v).strip() for v in participants}
        metadata = metadata.loc[metadata["participant_id"].astype(str).isin(selected)].copy()
    if metadata.empty:
        raise ValueError("No EEG participants selected for clock-cache construction")

    acquisition_root = Path(acquisition_root)
    cache_root = Path(cache_root)
    irregular_root = cache_root / "irregular_timestamps"
    cache_root.mkdir(parents=True, exist_ok=True)
    irregular_root.mkdir(parents=True, exist_ok=True)

    all_easy = sorted(acquisition_root.rglob("*.easy"))
    recording_rows: list[dict] = []
    trigger_rows: list[dict] = []
    resolution_rows: list[dict] = []
    failures: list[str] = []

    for _, row in metadata.sort_values("participant_id").iterrows():
        participant_id = str(row["participant_id"]).strip()
        candidates = _candidate_paths(row, all_easy)
        if not candidates:
            resolution_rows.append({
                "participant_id": participant_id,
                "candidate_easy_path": "",
                "candidate_size_bytes": np.nan,
                "history_exact": False,
                "history_basename_match": False,
                "sample_count_match": False,
                "sample_interval_match": False,
                "timestamp_regular": False,
                "event_sequence_match": False,
                "scene_latency_match": False,
                "view_count": 0,
                "view_count_match": False,
                "info_metadata_match": False,
                "structural_match": False,
                "selected": False,
                "selection_reason": "no_named_or_history_basename_candidate_in_acquisition_root",
            })
        candidate_summaries: list[tuple[EasySummary, dict]] = []
        for candidate in candidates:
            summary = summarize_easy(candidate)
            validation = validate_easy_against_set(
                summary,
                row,
                expected_view_count=expected_view_count,
            )
            candidate_summaries.append((summary, validation))
            resolution_rows.append({
                "participant_id": participant_id,
                "candidate_easy_path": str(candidate),
                "candidate_size_bytes": candidate.stat().st_size,
                "history_exact": _same_path(candidate, row.get("history_easy_path")),
                "history_basename_match": _same_basename(candidate, row.get("history_easy_path")),
                **validation,
                "selected": False,
                "selection_reason": "",
            })

        valid = [(summary, check) for summary, check in candidate_summaries if check["structural_match"]]
        selected: EasySummary | None = None
        selection_reason = ""
        history_valid = [(s, c) for s, c in valid if _same_path(s.path, row.get("history_easy_path"))]
        if len(history_valid) == 1:
            selected = history_valid[0][0]
            selection_reason = "validated_set_history_source"
        elif len(history_valid) == 0:
            basename_valid = [(s, c) for s, c in valid if _same_basename(s.path, row.get("history_easy_path"))]
            if len(basename_valid) == 1:
                selected = basename_valid[0][0]
                selection_reason = "validated_set_history_basename_fallback"
        if selected is None and len(valid) == 1:
            selected = valid[0][0]
            selection_reason = "unique_structural_match"
        elif selected is None and len(valid) > 1:
            sizes = sorted({s.path.stat().st_size for s, _ in valid}, reverse=True)
            largest = [(s, c) for s, c in valid if s.path.stat().st_size == sizes[0]]
            if len(largest) == 1:
                selected = largest[0][0]
                selection_reason = "largest_among_structural_matches"

        if selected is None:
            failures.append(participant_id)
            continue

        for resolution in resolution_rows:
            if resolution["participant_id"] == participant_id and _same_path(Path(resolution["candidate_easy_path"]), selected.path):
                resolution["selected"] = True
                resolution["selection_reason"] = selection_reason

        interval = 1000.0 / float(row["srate_hz"])
        irregular_path = ""
        if not selected.timestamp_regular:
            irregular_path = str(irregular_root / f"{_safe_name(participant_id)}_timestamps.csv")
            write_irregular_timestamps(selected.path, irregular_path)
        tz = ZoneInfo(timezone_name)
        recording_rows.append({
            "participant_id": participant_id,
            "first_eeg_epoch_ms": selected.first_epoch_ms,
            "last_eeg_epoch_ms": selected.last_epoch_ms,
            "srate_hz": float(row["srate_hz"]),
            "n_samples": selected.n_samples,
            "sample_interval_ms": interval,
            "timezone": timezone_name,
            "source_easy_path": str(selected.path),
            "source_size": selected.path.stat().st_size,
            "source_sha256": selected.sha256,
            "source_created_time_local": datetime.fromtimestamp(selected.path.stat().st_ctime, tz=tz).isoformat(timespec="milliseconds"),
            "source_modified_time_local": datetime.fromtimestamp(selected.path.stat().st_mtime, tz=tz).isoformat(timespec="milliseconds"),
            "timestamp_regular": selected.timestamp_regular,
            "irregular_timestamp_path": irregular_path,
            "cache_status": "pass" if selected.timestamp_regular else "pass_irregular_timestamp_vector",
        })
        set_types = _split_ints(row.get("event_types"))
        set_latencies = _split_ints(row.get("event_latencies"))
        set_epochs = _event_epochs_from_clock(selected, set_latencies, interval)
        for event_index, (code, sample, epoch) in enumerate(
            zip(set_types, set_latencies, set_epochs, strict=True), start=1,
        ):
            trigger_rows.append({
                "participant_id": participant_id,
                "event_index": event_index,
                "trigger_code": code,
                "sample_index": sample,
                "eeg_epoch_ms": epoch,
                "eeg_datetime_local": datetime.fromtimestamp(epoch / 1000.0, tz=timezone.utc).astimezone(tz).isoformat(timespec="milliseconds"),
            })

    if failures and not allow_partial:
        resolution_path = write_table(pd.DataFrame(resolution_rows), cache_root / "eeg_source_resolution.csv")
        raise ValueError(
            "Could not uniquely resolve validated EASY sources for "
            f"{len(failures)} participant(s). Audit: {resolution_path}"
        )
    if not recording_rows:
        raise ValueError("No EEG clock-cache rows were generated")

    return {
        "recording_clock": write_table(pd.DataFrame(recording_rows), cache_root / "eeg_recording_clock.csv"),
        "trigger_events": write_table(pd.DataFrame(trigger_rows), cache_root / "eeg_trigger_events.csv"),
        "source_resolution": write_table(pd.DataFrame(resolution_rows), cache_root / "eeg_source_resolution.csv"),
    }


def summarize_easy(path: str | Path) -> EasySummary:
    path = Path(path)
    sha = hashlib.sha256()
    n_samples = 0
    first_epoch: int | None = None
    last_epoch: int | None = None
    previous_epoch: int | None = None
    intervals: set[int] = set()
    trigger_codes: list[int] = []
    trigger_samples: list[int] = []
    trigger_epochs: list[int] = []
    previous_trigger = 0

    with path.open("rb") as raw:
        for raw_line in raw:
            sha.update(raw_line)
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            fields = line.split("\t")
            if len(fields) < 13:
                continue
            n_samples += 1
            epoch = int(float(fields[12]))
            trigger = int(float(fields[11] or 0))
            if first_epoch is None:
                first_epoch = epoch
            if previous_epoch is not None:
                intervals.add(epoch - previous_epoch)
            previous_epoch = epoch
            last_epoch = epoch
            if trigger != 0 and trigger != previous_trigger:
                trigger_codes.append(trigger)
                trigger_samples.append(n_samples)
                trigger_epochs.append(epoch)
            previous_trigger = trigger

    if n_samples == 0 or first_epoch is None or last_epoch is None:
        raise ValueError(f"No valid EASY samples found: {path}")
    positive_intervals = sorted(v for v in intervals if v > 0)
    sample_interval = float(positive_intervals[0]) if positive_intervals else np.nan
    timestamp_regular = len(intervals) == 1 and len(positive_intervals) == 1
    view_count = sum(
        1 for left, right in zip(trigger_codes[:-1], trigger_codes[1:], strict=False)
        if left == 7 and right == 8
    )
    info = parse_info(path.with_suffix(".info"))
    return EasySummary(
        path=path,
        n_samples=n_samples,
        first_epoch_ms=first_epoch,
        last_epoch_ms=last_epoch,
        sample_interval_ms=sample_interval,
        timestamp_regular=timestamp_regular,
        trigger_codes=tuple(trigger_codes),
        trigger_samples=tuple(trigger_samples),
        trigger_epochs_ms=tuple(trigger_epochs),
        view_count=view_count,
        sha256=sha.hexdigest(),
        info_first_epoch_ms=info.get("first_epoch_ms"),
        info_srate_hz=info.get("srate_hz"),
        info_n_samples=info.get("n_samples"),
        info_packets_lost=info.get("packets_lost"),
    )


def validate_easy_against_set(summary: EasySummary, metadata: pd.Series, expected_view_count: int = EXPECTED_VIEW_COUNT) -> dict:
    set_types = _split_ints(metadata.get("event_types"))
    set_latencies = _split_ints(metadata.get("event_latencies"))
    event_match = set_types == list(summary.trigger_codes) and set_latencies == list(summary.trigger_samples)
    set_view_pairs = _view_pairs(set_types, set_latencies)
    easy_view_pairs = _view_pairs(list(summary.trigger_codes), list(summary.trigger_samples))
    scene_latency_match = set_view_pairs == easy_view_pairs
    srate = float(metadata["srate_hz"])
    expected_interval = 1000.0 / srate
    info_match = (
        summary.info_first_epoch_ms in {None, summary.first_epoch_ms}
        and summary.info_srate_hz in {None, srate}
        and summary.info_n_samples in {None, summary.n_samples}
    )
    checks = {
        "sample_count_match": summary.n_samples == int(metadata["n_samples"]),
        "sample_interval_match": bool(np.isclose(summary.sample_interval_ms, expected_interval)),
        "timestamp_regular": summary.timestamp_regular,
        "event_sequence_match": event_match,
        "scene_latency_match": scene_latency_match,
        "view_count": summary.view_count,
        "view_count_match": summary.view_count == expected_view_count,
        "info_metadata_match": info_match,
    }
    checks["structural_match"] = all([
        checks["sample_count_match"],
        checks["sample_interval_match"],
        checks["info_metadata_match"],
    ])
    return checks


def parse_info(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "first_epoch_ms": r"StartDate \(firstEEGtimestamp\):\s*(\d+)",
        "srate_hz": r"EEG sampling rate:\s*([0-9.]+)",
        "n_samples": r"Number of records of EEG:\s*(\d+)",
        "packets_lost": r"Number of packets lost:\s*(\d+)",
    }
    out: dict[str, int | float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        out[key] = float(match.group(1)) if key == "srate_hz" else int(match.group(1))
    return out


def write_irregular_timestamps(easy_path: str | Path, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(easy_path).open("r", encoding="utf-8", errors="replace") as source, output_csv.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["sample_index", "eeg_epoch_ms"])
        sample_index = 0
        for line in source:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 13:
                continue
            sample_index += 1
            writer.writerow([sample_index, int(float(fields[12]))])
    return output_csv


def _event_epochs_from_clock(
    summary: EasySummary,
    sample_indices: list[int],
    sample_interval_ms: float,
) -> list[int]:
    if summary.timestamp_regular:
        return [
            int(round(summary.first_epoch_ms + (sample_index - 1) * sample_interval_ms))
            for sample_index in sample_indices
        ]
    wanted = set(sample_indices)
    epochs: dict[int, int] = {}
    sample_index = 0
    with summary.path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 13:
                continue
            sample_index += 1
            if sample_index in wanted:
                epochs[sample_index] = int(float(fields[12]))
            if len(epochs) == len(wanted):
                break
    missing = [sample for sample in sample_indices if sample not in epochs]
    if missing:
        raise ValueError(f"Event sample indices outside EASY clock vector for {summary.path}: {missing[:5]}")
    return [epochs[sample] for sample in sample_indices]


def _candidate_paths(metadata: pd.Series, all_easy: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    history = str(metadata.get("history_easy_path") or "").strip()
    history_path = Path(history) if history else None
    if history_path and history_path.exists():
        candidates.append(history_path)
    if history_path:
        candidates.extend(path for path in all_easy if path.name.casefold() == history_path.name.casefold())
    participant = str(metadata["participant_id"]).strip().casefold()
    candidates.extend(path for path in all_easy if participant and participant in path.stem.casefold())
    unique: dict[str, Path] = {}
    for path in candidates:
        try:
            key = str(path.resolve()).casefold()
        except OSError:
            key = str(path.absolute()).casefold()
        unique[key] = path
    return sorted(unique.values(), key=lambda p: str(p).casefold())


def _same_path(left: Path, right: object) -> bool:
    text = str(right or "").strip()
    if not text:
        return False
    try:
        return left.resolve() == Path(text).resolve()
    except OSError:
        return str(left).casefold() == text.casefold()


def _same_basename(left: Path, right: object) -> bool:
    text = str(right or "").strip()
    return bool(text) and left.name.casefold() == Path(text).name.casefold()


def _split_ints(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(round(float(part))) for part in str(value).split(EVENT_SEPARATOR) if str(part).strip()]


def _view_pairs(types: list[int], latencies: list[int]) -> list[tuple[int, int]]:
    return [
        (int(latencies[index]), int(latencies[index + 1]))
        for index in range(min(len(types), len(latencies)) - 1)
        if types[index] == 7 and types[index + 1] == 8
    ]


def _safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", value).strip() or "participant"
