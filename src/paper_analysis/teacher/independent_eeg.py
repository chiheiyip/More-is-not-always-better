"""Independent EEG QC/predecessor reconstruction and guarded prior-result reuse."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .independent_eye import KEY, dump, save, sha, unique, read_questionnaire

METRICS = ["hf_ratio_20_40Hz", "rms_mean_uV", "peak_to_peak_uV"]


def reconstruct_qc(features, config):
    """Compute thresholds separately for each trim across all available people."""
    unique(features, KEY + ["onset_trim_s"])
    frames, thresholds = [], []
    for trim, rows in features.groupby("onset_trim_s", sort=True):
        d = rows.copy().reset_index(drop=True)
        reasons = [[] for _ in range(len(d))]
        for field, cutoff, label in [("analysis_dur_s", config["min_segment_duration_s"], "segment_duration"),
                                     ("nan_fraction", config["nan_fraction_threshold"], "nan_fraction"),
                                     ("flat_fraction", config["flat_fraction_threshold"], "flat_fraction")]:
            v = pd.to_numeric(d[field], errors="coerce")
            bad = (v.isna() | v.lt(cutoff)) if field == "analysis_dur_s" else v.gt(cutoff)
            for i in np.flatnonzero(bad):
                reasons[i].append(label)
        for i in np.flatnonzero(~d.segment_valid_duration.astype(str).str.lower().isin(["true", "1"])):
            reasons[i].append("segment_valid_duration")
        for metric in config.get("robust_metrics", METRICS):
            values = pd.to_numeric(d[metric], errors="coerce")
            finite = values.dropna()
            median = float(finite.median()) if len(finite) else np.nan
            mad = float((finite - median).abs().median()) if len(finite) else np.nan
            cutoff = median + config["robust_k"] * mad if len(finite) >= config["robust_min_n"] else np.nan
            thresholds.append({"onset_trim_s": trim, "metric": metric, "n": len(finite), "median": median, "mad": mad, "threshold": cutoff})
            for i in np.flatnonzero(values.gt(cutoff)):
                reasons[i].append("robust_" + metric)
        d["bad_eeg_quality"] = [bool(r) for r in reasons]
        d["eeg_subject_quality_exclusion"] = False
        for _, indexes in d.groupby("Participant").groups.items():
            if d.loc[indexes, "bad_eeg_quality"].mean() > config["bad_scene_fraction_threshold"]:
                d.loc[indexes, "bad_eeg_quality"] = True
                d.loc[indexes, "eeg_subject_quality_exclusion"] = True
                for i in indexes:
                    reasons[i].append("eeg_subject_quality_exclusion")
        d["QCReasons"] = [";".join(r) for r in reasons]
        d["QCIncluded"] = ~d.bad_eeg_quality
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    counts = out.groupby(KEY).agg(Variants=("onset_trim_s", "nunique"), AllPass=("QCIncluded", "all")).reset_index()
    counts["CommonQCIncluded"] = counts.Variants.eq(4) & counts.AllPass
    out = out.merge(counts[KEY + ["CommonQCIncluded"]], on=KEY, validate="many_to_one")
    return out, pd.DataFrame(thresholds)


def add_design(features, config):
    q = read_questionnaire(config["questionnaire_file"])
    design = pd.read_excel(config["trial_order_mapping"], sheet_name=0)[KEY + ["WWR", "Complexity", "Block", "PositionWithinBlock", "OrderGroup", "SceneID"]]
    unique(design, KEY)
    design = design.sort_values(KEY).copy()
    design["PreviousWWR"] = design.groupby(["Participant", "Block"]).WWR.shift()
    design["PreviousComplexity"] = design.groupby(["Participant", "Block"]).Complexity.shift()
    design["PositionWithinBlockCentered"] = design.PositionWithinBlock - 3.5
    participants = pd.read_excel(config["participant_information"], sheet_name=0)[["Participant", "Gender"]]
    result = features.merge(design, on=KEY, how="left", validate="many_to_one")
    if result.WWR.isna().any():
        raise ValueError("EEG source event has no original trial mapping")
    result = result.merge(participants, on="Participant", how="left", validate="many_to_one").merge(q[["Participant", "ExerciseFrequency", "Q1.4Original"]], on="Participant", how="left", validate="many_to_one")
    if result[["Gender", "ExerciseFrequency"]].isna().any().any():
        raise ValueError("EEG participant linkage failed")
    return result


def compare_qc(reconstructed, historical_path):
    old = pd.read_csv(historical_path, encoding="utf-8-sig")
    old = old.rename(columns={"participant_id": "Participant", "scene_id": "GlobalTrialOrder"})
    unique(old, KEY + ["onset_trim_s"])
    names = KEY + ["onset_trim_s"]
    merged = reconstructed.merge(old[names + ["bad_eeg_quality", "eeg_subject_quality_exclusion", "analysis_start_s", "analysis_end_s"]], on=names, how="outer", suffixes=("_audit", "_historical"), indicator=True)
    merged["HistoricalIncluded"] = ~merged.bad_eeg_quality_historical.astype(str).str.lower().isin(["true", "1"])
    merged["InclusionMatch"] = merged._merge.eq("both") & merged.QCIncluded.eq(merged.HistoricalIncluded)
    merged["HistoricalStartSample"] = np.rint(merged.analysis_start_s * merged.srate)
    merged["HistoricalEndSample"] = np.rint(merged.analysis_end_s * merged.srate)
    merged["EndpointMatch"] = merged.AnalysisStartSample.eq(merged.HistoricalStartSample) & merged.EndSample.eq(merged.HistoricalEndSample)
    return merged


def verify_prior(config, repo):
    """Exact prior verification outputs + original input manifests; no relabelling."""
    root = Path(config["prior_verification"])
    summary = json.loads((root / "verification_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "verified" or not summary["source_files_unchanged"]:
        raise ValueError("Prior independent EEG verification is not valid")
    checks = []
    for list_name in ["input_hashes_after.json", "verification_output_reuse_hashes.json"]:
        records = json.loads((root / list_name).read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("outputs", records.get("files", records.get("records", [])))
        if not isinstance(records, list):
            raise ValueError("Unsupported prior hash manifest")
        for record in records:
            p = Path(record.get("path", record.get("file", "")))
            if not p.is_absolute():
                p = root / p
            expected = record.get("sha256")
            if not expected or not p.is_file() or sha(p) != expected:
                raise ValueError(f"Prior verification source/output changed: {p}")
            checks.append({"path": str(p), "sha256": expected, "status": "unchanged"})
    # Statistical implementations are unchanged; entrypoint/presentation changes
    # are checked by their relevant module hashes rather than whole CLI bytes.
    for record in summary["methods"]:
        relative = record["path"]
        if relative.startswith("scripts/"):
            continue
        if sha(Path(repo) / relative) != record["sha256"]:
            raise ValueError(f"Verified statistical implementation changed: {relative}")
    return summary, checks


def prepare(config, output):
    output = Path(output)
    features = pd.read_csv(output / "independent_eeg_source_features.csv", encoding="utf-8-sig")
    qc = json.loads(Path(config["eeg_qc_config"]).read_text(encoding="utf-8"))
    reconstructed, thresholds = reconstruct_qc(features, qc)
    reconstructed = add_design(reconstructed, config)
    save(reconstructed, output / "independent_eeg_trial_QC.csv")
    save(thresholds, output / "independent_eeg_QC_thresholds.csv")
    # Independent reconstruction is frozen before loading historical input.
    dump(output / "source_reconstruction_seal.json", {"independent_eeg_trial_QC_sha256": sha(output / "independent_eeg_trial_QC.csv"),
         "independent_eeg_QC_thresholds_sha256": sha(output / "independent_eeg_QC_thresholds.csv"),
         "source_state": "existing preprocessed SET/FDT; no repeat preprocessing or ICA removal"})
    return reconstructed


def read_raw_acquisition(candidate):
    """NIC EASY: 8 EEG + 3 accelerometer + trigger + epoch-ms, blank lines ignored."""
    p = Path(candidate["candidate_easy_path"])
    before = sha(p)
    data = pd.read_csv(p, sep="\t", header=None, usecols=[11, 12], dtype=np.int64, skip_blank_lines=True)
    trigger, timestamps = data[11].to_numpy(), data[12].to_numpy()
    indexes = np.flatnonzero((trigger != 0) & np.r_[True, trigger[1:] != trigger[:-1]])
    events = pd.DataFrame({"Participant": candidate["participant_id"], "Marker": trigger[indexes].astype(str),
                           "LatencySample": indexes + 1, "EpochMS": timestamps[indexes]})
    after = sha(p)
    if before != after:
        raise ValueError("Raw acquisition source changed during read")
    delta = np.diff(timestamps)
    info = p.with_suffix(".info")
    # Info bytes have ASCII numeric metadata and may contain GB18030 labels.
    text = ""
    if info.is_file():
        for encoding in ["utf-8-sig", "gb18030"]:
            try: text = info.read_text(encoding=encoding); break
            except UnicodeError: pass
    import re
    def number(label):
        match = re.search(re.escape(label) + r":\s*(\d+)", text)
        return int(match.group(1)) if match else None
    record = {"Participant": candidate["participant_id"], "SourceEASY": str(p), "SHA256": before,
              "Samples": len(data), "Triggers": len(indexes), "TimestampMedianIntervalMS": float(np.median(delta)),
              "TimestampRegular": bool(np.all(delta == 2)), "NonpositiveTimestampSteps": int((delta <= 0).sum()),
              "InfoSamples": number("Number of records of EEG"), "InfoEEGChannels": number("Number of EEG channels"),
              "InfoSamplingRate": number("EEG sampling rate"), "InfoSHA256": sha(info) if info.is_file() else None}
    return record, events


def acquisition_audit(config, output):
    """Cached paths locate sources; every numerical check reads original EASY."""
    import concurrent.futures
    output = Path(output)
    location = pd.read_csv(config["acquisition_source_map"], encoding="utf-8-sig")
    chosen = location.loc[location.selected.astype(str).str.lower().isin(["true", "1"])].copy()
    unique(chosen, ["participant_id"])
    records, events = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(read_raw_acquisition, r) for r in chosen.to_dict("records") if Path(r["candidate_easy_path"]).is_file()]
        for i, task in enumerate(concurrent.futures.as_completed(tasks), 1):
            record, ev = task.result(); records.append(record); events.append(ev)
            if i % 10 == 0: print(f"Independent original EASY reads: {i}/{len(tasks)}", flush=True)
    raw = pd.DataFrame(records)
    all_events = pd.concat(events, ignore_index=True)
    save(raw, output / "independent_raw_acquisition_audit.csv")
    save(all_events, output / "independent_raw_trigger_events.csv")
    metadata = pd.read_csv(output / "independent_eeg_source_inventory.csv", encoding="utf-8-sig")
    combined = metadata.merge(raw, on="Participant", how="left", suffixes=("_SET", "_EASY"), validate="one_to_one")
    combined["RawSourceAvailable"] = combined.SourceEASY.notna()
    combined["SampleCountMatch"] = combined.Samples_SET.eq(combined.Samples_EASY)
    combined["InfoSampleCountMatch"] = combined.Samples_EASY.eq(combined.InfoSamples)
    save(combined, output / "raw_to_SET_source_identity.csv")
    archived_events = pd.read_csv(output / "independent_eeg_set_events.csv", encoding="utf-8-sig", dtype={"Marker": str})
    comparisons = []
    for participant in raw.Participant:
        a = all_events.loc[all_events.Participant.eq(participant) & all_events.Marker.isin(["7", "8"])].reset_index(drop=True)
        b = archived_events.loc[archived_events.Participant.eq(participant) & archived_events.Marker.isin(["7", "8"])].reset_index(drop=True)
        same = len(a) == len(b)
        for i in range(max(len(a), len(b))):
            x = a.iloc[i] if i < len(a) else None; y = b.iloc[i] if i < len(b) else None
            comparisons.append({"Participant": participant, "SceneEventOrdinal": i+1,
                "RawMarker": x.Marker if x is not None else None, "SETMarker": y.Marker if y is not None else None,
                "RawSample": int(x.LatencySample) if x is not None else None,
                "SETSample": float(y.LatencySample) if y is not None else None,
                "Match": bool(same and x is not None and y is not None and x.Marker == y.Marker and x.LatencySample == y.LatencySample)})
    save(pd.DataFrame(comparisons), output / "raw_to_SET_scene_event_audit.csv")
    return combined


def match_scene_epochs(raw_events, set_events):
    """Match adjacent 7→8 epochs by endpoints, never by marker ordinal."""
    def epochs(events, source):
        rows = []
        for person, group in events.groupby("Participant"):
            group = group.sort_values("LatencySample").reset_index(drop=True)
            marker = group.Marker.astype(str).str.replace(r"\.0$", "", regex=True)
            for i in range(len(group)-1):
                if marker.iloc[i] == "7" and marker.iloc[i+1] == "8":
                    rows.append({"Participant": person, "StartSample": float(group.LatencySample.iloc[i]),
                                 "EndSample": float(group.LatencySample.iloc[i+1]), source+"Ordinal": len([r for r in rows if r["Participant"]==person])+1})
        return pd.DataFrame(rows, columns=["Participant","StartSample","EndSample",source+"Ordinal"])
    a, b = epochs(raw_events, "Raw"), epochs(set_events, "SET")
    keys = ["Participant","StartSample","EndSample"]
    unique(a, keys); unique(b, keys)
    result = a.merge(b, on=keys, how="outer", indicator="EndpointMatchStatus", validate="one_to_one")
    result["Interpretation"] = result.EndpointMatchStatus.map({"both":"exact original acquisition endpoints", "left_only":"original pair not present in SET; investigate event editing", "right_only":"SET pair not present among adjacent original triggers; investigate event editing"}).astype(str)
    return result


def current_model_inputs(config, output):
    """Keep historical A/B/C intact; freeze additional current-source QC D/E."""
    output = Path(output)
    d = pd.read_csv(output / "independent_eeg_trial_QC.csv", encoding="utf-8-sig")
    d = d.loc[d.CommonQCIncluded].copy()
    columns = KEY + ["WWR","Complexity","ExerciseFrequency","Gender","Block","PositionWithinBlock",
                     "PositionWithinBlockCentered","OrderGroup","PreviousWWR","PreviousComplexity"]
    powers = [f"{roi}_{band}" for roi in ["F","P","O"] for band in ["theta","alpha","beta"]]
    for trim in [0,5,10,15]:
        selected = d.loc[d.onset_trim_s.eq(trim)].copy()
        unique(selected, KEY)
        if trim == 0: frozen = set(map(tuple, selected[KEY].to_numpy()))
        elif frozen != set(map(tuple, selected[KEY].to_numpy())): raise ValueError("Current common-QC identities differ between trims")
        base = selected[columns].rename(columns={"ExerciseFrequency":"ExperienceGroup"}).copy()
        for name, prefix in [("WWR","WWR"),("Complexity","C"),("PreviousWWR","WWR"),("PreviousComplexity","C")]:
            base[name] = base[name].map(lambda v: prefix+str(int(v)) if pd.notna(v) else np.nan)
        for stem in powers:
            absolute = selected[stem+"_absolute"].to_numpy()
            if not np.isfinite(absolute).all() or (absolute<=0).any(): raise ValueError("Invalid current-QC absolute numerator")
            base["log10_"+stem+"_absolute"] = np.log10(absolute)
        for version, suffix in [("D_current_QC_1_45",""),("E_current_QC_1_40","_1_40")]:
            table = base.copy()
            for stem in powers: table[stem+"_relative"] = selected[stem+"_relative"+suffix].to_numpy()
            if not np.isfinite(table[[stem+"_relative" for stem in powers]].to_numpy()).all(): raise ValueError("Invalid current-QC relative power")
            target = output / "current_QC_inputs" / version / f"trim_{trim}s"
            target.mkdir(parents=True,exist_ok=True); save(table,target/"input.csv")
    return output / "current_QC_inputs"
