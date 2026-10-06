"""Independent source-to-trial eye audit. No production eye helpers are imported.

The comparison phase is deliberately separate and requires a sealed independent
run. This module may read approved measurement inputs, never formal result tables.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

KEY = ["Participant", "GlobalTrialOrder"]
AOIS = ["Table", "Window", "Equipment", "Background"]
CORE = ["TableShare", "WindowShare", "RawCompetition", "LogTableEnrichment",
        "LogWindowEnrichment", "AdjustedCompetition"]
RAW_COLUMNS = ["User", "Recording Time Stamp[ms]", "Tracking Ratio[%]", "Validity Left",
               "Validity Right", "Fixation Index", "Fixation Duration[ms]",
               "Fixation Point X[px]", "Fixation Point Y[px]"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def canonical_name(value):
    return re.split(r"[·•]", str(value).strip())[0].strip()


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def unique(frame, columns):
    if frame[columns].isna().any().any() or frame[columns].astype(str).eq("").any().any() or frame.duplicated(columns).any():
        raise ValueError(f"Missing or duplicate identity: {columns}")


def read_questionnaire(path):
    q = pd.read_excel(path, sheet_name=0, header=0, dtype=object)
    fields = {}
    for code in ("Q1.0", "Q1.4", "Q1.5"):
        cols = [c for c in q if re.match(re.escape(code) + r"(?:_|\s|$)", str(c)) and not str(c).endswith("_word")]
        if len(cols) != 1:
            raise ValueError(f"Ambiguous questionnaire field {code}: {cols}")
        fields[code] = cols[0]
    out = pd.DataFrame({"Participant": q[fields["Q1.0"]].map(canonical_name),
                        "QuestionnaireName": q[fields["Q1.0"]],
                        "Q1.4Original": q[fields["Q1.4"]], "Q1.5Original": q[fields["Q1.5"]]})
    def group(v):
        text = str(v).strip()
        if text.startswith(("从不", "偶尔")):
            return "Low"
        if text.startswith(("有时", "经常")):
            return "High"
        raise ValueError(f"Unrecognized Q1.4 answer: {text}")
    out["ExerciseFrequency"] = out["Q1.4Original"].map(group)
    unique(out, ["Participant"])
    return out


def load_measurement_inputs(config):
    participant = pd.read_excel(config["participant_information"], sheet_name=0)
    mapping = pd.read_excel(config["trial_order_mapping"], sheet_name=0)
    scene = pd.read_excel(config["scene_aoi_mapping"], sheet_name=0)
    q = read_questionnaire(config["questionnaire_file"])
    unique(participant, ["Participant"])
    unique(mapping, KEY)
    unique(scene, ["SceneID", "OrderGroup", "Block"])
    required = ["Participant", "OrderGroup", "Block", "PositionWithinBlock", "GlobalTrialOrder", "SceneID", "WWR", "Complexity", "CSVFile"]
    if not set(required).issubset(mapping):
        raise ValueError("Incomplete original trial mapping")
    # Exclude precomputed predecessors and group classifications as input values.
    mapping = mapping[required].copy()
    participant = participant[["Participant", "Gender", "OrderGroup", "IncludeEyeCandidate", "IncludeEEGValid"]].copy()
    data = mapping.merge(participant.drop(columns="OrderGroup"), on="Participant", validate="many_to_one")
    data = data.merge(q, on="Participant", validate="many_to_one")
    data = data.loc[data.IncludeEyeCandidate.map(truth)].copy()
    if len(data) != len(mapping.loc[mapping.Participant.isin(participant.loc[participant.IncludeEyeCandidate.map(truth), "Participant"])]):
        raise ValueError("Questionnaire or participant linkage lost a trial")
    data = data.sort_values(KEY).reset_index(drop=True)
    data["PositionWithinBlockCentered"] = data.PositionWithinBlock - 3.5
    data["PreviousWWR"] = data.groupby(["Participant", "Block"]).WWR.shift()
    data["PreviousComplexity"] = data.groupby(["Participant", "Block"]).Complexity.shift()
    audit = []
    for name, d in data.groupby("Participant"):
        valid = (len(d) == 12 and set(d.GlobalTrialOrder) == set(range(1, 13))
                 and (d.Block == np.ceil(d.GlobalTrialOrder / 6)).all()
                 and (d.PositionWithinBlock == (d.GlobalTrialOrder - 1) % 6 + 1).all()
                 and d.OrderGroup.nunique() == 1)
        audit.append({"Participant": name, "Trials": len(d), "Blocks": d.Block.nunique(),
                      "OrderGroup": d.OrderGroup.iloc[0], "StructureCorrect": bool(valid),
                      "CSVFilesExist": bool(d.CSVFile.map(lambda p: Path(p).is_file()).all())})
    audit = pd.DataFrame(audit)
    if not audit.StructureCorrect.all() or not audit.CSVFilesExist.all():
        raise ValueError("Original participant/trial structure or source file failed")
    sequences = data.groupby(["OrderGroup", "Block", "PositionWithinBlock"])[["SceneID", "WWR", "Complexity"]].nunique()
    if (sequences > 1).any().any():
        raise ValueError("Inconsistent original order-group sequence")
    return data, scene, q, audit


def make_masks(record, delta=0, spherical=False):
    labels = json.loads(Path(record["AOIFile"]).read_text(encoding="utf-8-sig"))
    width, height = int(labels["image"]["width"]), int(labels["image"]["height"])
    with Image.open(record["BaseImageFile"]) as base:
        if base.size != (width, height):
            raise ValueError("AOI and original base image dimensions differ")
    with Image.open(record["ValidSceneFile"]) as picture:
        valid = np.asarray(picture.convert("L")) > 0
    if valid.shape != (height, width) or not valid.any():
        raise ValueError("Invalid approved ValidScene mask")
    raw = {}
    for aoi in AOIS[:3]:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in labels["aoi_classes"].get(aoi.lower(), []):
            pts = np.rint(np.asarray(polygon["points"], dtype=float)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        if delta:
            kernel = np.ones((2 * abs(delta) + 1, 2 * abs(delta) + 1), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel) if delta > 0 else cv2.erode(mask, kernel)
        raw[aoi] = mask.astype(bool) & valid
    overlaps = sum(int((raw[a] & raw[b]).sum()) for a, b in [("Table", "Window"), ("Table", "Equipment"), ("Window", "Equipment")])
    resolved = {}
    used = np.zeros_like(valid)
    # The approved measurement mapping explicitly specifies this priority.
    if str(record.get("OverlapResolution", "")) not in {"table_window_equipment_priority", "none", "nan", ""}:
        raise ValueError("Unknown approved overlap resolution")
    for aoi in AOIS[:3]:
        resolved[aoi] = raw[aoi] & ~used
        used |= resolved[aoi]
    if int(record["Complexity"]) == 0:
        if raw["Equipment"].any():
            raise ValueError("C0 contains Equipment polygons")
        resolved["Equipment"][:] = False
    resolved["Background"] = valid & ~used
    code = np.zeros(valid.shape, dtype=np.uint8)
    for number, aoi in enumerate(AOIS, 1):
        code[resolved[aoi]] = number
    if (code[valid] == 0).any() or (code[~valid] != 0).any():
        raise ValueError("AOI partition is incomplete")
    if spherical:
        if str(record["ProjectionType"]).lower() != "equirectangular":
            raise ValueError("Spherical weighting requires verified equirectangular projection")
        weight = np.cos(np.pi * ((np.arange(height) + .5) / height - .5))[:, None]
    else:
        weight = np.ones((height, 1))
    total = float((valid * weight).sum())
    shares = {a + "AreaShare": float((m * weight).sum()) / total for a, m in resolved.items()}
    if int(record["Complexity"]) == 0:
        shares["EquipmentAreaShare"] = np.nan
    return code, shares, {"ImageWidth": width, "ImageHeight": height, "ValidScenePixels": int(valid.sum()),
                          "OverlapPixelsBeforeResolution": overlaps,
                          "CoordinateOrigin": "upper_left", "YAxis": "down",
                          "ProjectionType": str(record["ProjectionType"]),
                          "OffStimulusPixels": int((~valid).sum()), "AreaMethod": "spherical" if spherical else "pixel"}


def read_raw(path):
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            d = pd.read_csv(path, encoding=encoding, usecols=RAW_COLUMNS)
            return d, encoding
        except UnicodeError:
            continue
    raise ValueError(f"No valid raw CSV encoding: {path}")


def deduplicate_fixations(raw, tolerance=1):
    numeric = [c for c in RAW_COLUMNS if c != "User"]
    d = raw.copy()
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    first = float(d["Recording Time Stamp[ms]"].min())
    events, issues = [], []
    valid = d["Fixation Index"].notna() & (d["Fixation Index"] >= 0)
    for index, rows in d.loc[valid].groupby("Fixation Index", sort=True):
        x, y, durations = (rows[c].dropna() for c in ["Fixation Point X[px]", "Fixation Point Y[px]", "Fixation Duration[ms]"])
        finite = len(x) and len(y) and len(durations)
        if not finite or float(durations.median()) <= 0:
            issues.append({"FixationIndex": index, "Issue": "missing_or_nonpositive_duration_or_coordinate"})
            continue
        spread = float(max(x.max() - x.min(), y.max() - y.min()))
        duration_conflict = durations.max() != durations.min()
        if spread > tolerance or duration_conflict:
            issues.append({"FixationIndex": index, "Issue": "coordinate_conflict" if spread > tolerance else "duration_conflict",
                           "CoordinateSpread": spread, "DurationValues": ";".join(map(str, durations.unique()))})
        events.append({"FixationIndex": index, "RawRows": len(rows),
                       "RawXMin": float(x.min()), "RawXMax": float(x.max()), "RawYMin": float(y.min()), "RawYMax": float(y.max()),
                       "FixationX": float(x.median()), "FixationY": float(y.median()),
                       "FixationDuration": float(durations.iloc[0]),
                       "FixationStartMS": float(rows["Recording Time Stamp[ms]"].min()) - first,
                       "CoordinateConflict": spread > tolerance, "DurationConflict": bool(duration_conflict)})
    columns = ["FixationIndex", "RawRows", "RawXMin", "RawXMax", "RawYMin", "RawYMax", "FixationX", "FixationY",
               "FixationDuration", "FixationStartMS", "CoordinateConflict", "DurationConflict"]
    return pd.DataFrame(events, columns=columns), pd.DataFrame(issues)


def assign_events(events, code):
    out = events.copy()
    x = np.rint(out.FixationX).astype(int).to_numpy()
    y = np.rint(out.FixationY).astype(int).to_numpy()
    inside = (x >= 0) & (x < code.shape[1]) & (y >= 0) & (y < code.shape[0])
    category = np.zeros(len(out), dtype=int)
    category[inside] = code[y[inside], x[inside]]
    out["AOICategory"] = np.asarray(["OffStimulus", *AOIS])[category]
    out["ValidSceneHit"] = category > 0
    return out


def trial_metrics(events, area, duration, tracking, complexity):
    row = dict(area)
    total = float(events.loc[events.ValidSceneHit, "FixationDuration"].sum())
    row["ValidSceneTFD"] = total
    valid_seconds = duration * tracking
    row["ValidTrackingSeconds"] = valid_seconds
    for aoi in AOIS:
        d = events.loc[events.AOICategory.eq(aoi)]
        structural = aoi == "Equipment" and int(complexity) == 0
        row.update({aoi + "Visited": np.nan if structural else int(len(d) > 0),
                    aoi + "FixationCount": np.nan if structural else len(d),
                    aoi + "TFD": np.nan if structural else float(d.FixationDuration.sum()),
                    aoi + "Share": np.nan if structural or total <= 0 else float(d.FixationDuration.sum()) / total,
                    aoi + "MeanFixationDuration": np.nan if structural or d.empty else float(d.FixationDuration.mean()),
                    aoi + "TTFF": np.nan if structural or d.empty else float(d.FixationStartMS.min()) / 1000,
                    aoi + "FCR": np.nan if structural or valid_seconds <= 0 else len(d) / valid_seconds})
    for aoi in ("Table", "Window"):
        share, size = row[aoi + "Share"], row[aoi + "AreaShare"]
        enrichment = share / size if size > 0 else np.nan
        row[aoi + "Enrichment"] = enrichment
        row["Log" + aoi + "Enrichment"] = np.log(enrichment) if enrichment > 0 else np.nan
    row["RawCompetition"] = row["TableShare"] - row["WindowShare"]
    row["AdjustedCompetition"] = row["LogTableEnrichment"] - row["LogWindowEnrichment"]
    rest = 1 - row["TableShare"] - row["WindowShare"]
    for label, num, den in [("CompositionTableRest", row["TableShare"], rest), ("CompositionWindowRest", row["WindowShare"], rest),
                            ("CompositionTableWindow", row["TableShare"], row["WindowShare"])]:
        row[label] = np.log(num / den) if num > 0 and den > 0 else np.nan
    return row


def preflight(config):
    data, scene, q, audit = load_measurement_inputs(config)
    approval = json.loads(Path(config["aoi_approval"]).read_text(encoding="utf-8-sig"))
    if not approval.get("approved"):
        raise ValueError("No approved measurement input authorization")
    for col in ["AOIFile", "BaseImageFile", "ValidSceneFile"]:
        if not scene[col].map(lambda p: Path(p).is_file()).all():
            raise ValueError(f"Missing original measurement input: {col}")
    return {"raw_trials": len(data), "candidate_participants": int(data.Participant.nunique()),
            "scene_images": int(scene.AOIFile.nunique()), "group_counts": q.ExerciseFrequency.value_counts().to_dict(),
            "projection_types": sorted(scene.ProjectionType.astype(str).unique()),
            "approval_notes": approval.get("notes", [])}


def source_paths(config, data, scene):
    initial = [Path(config[k]) for k in ["participant_information", "trial_order_mapping", "scene_aoi_mapping", "questionnaire_file", "aoi_approval"]]
    initial += [Path(p) for p in data.CSVFile]
    initial += [Path(p) for c in ["AOIFile", "BaseImageFile", "ValidSceneFile"] for p in scene[c]]
    return sorted(set(initial))


def process(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "independent_eye_trials.csv").exists():
        raise ValueError("Independent outputs already exist; use a new directory")
    data, scene, questionnaire, structure = load_measurement_inputs(config)
    sources = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in source_paths(config, data, scene)]
    dump(output / "source_hashes_before.json", sources)
    save(structure, output / "01_participant_trial_structure_audit.csv")
    save(questionnaire.drop(columns="Q1.5Original"), output / "independent_Q1_4_groups.csv")
    masks, areas, area_rows = {}, {}, []
    for _, s in scene.drop_duplicates("AOIFile").iterrows():
        base = s.to_dict()
        code, shares, detail = make_masks(base)
        masks[base["AOIFile"]], areas[base["AOIFile"]] = code, shares
        area_rows.append({"AOIFile": base["AOIFile"], "SceneID": base["SceneID"], "WWR": base["WWR"],
                          "Complexity": base["Complexity"], **detail, **shares})
    save(pd.DataFrame(area_rows), output / "02_AOI_coordinate_area_audit.csv")
    variants = [(0, False), (5, False), (-5, False)]
    if config.get("boundary_10px", False):
        variants += [(10, False), (-10, False)]
    if scene.ProjectionType.astype(str).str.lower().eq("equirectangular").all():
        variants += [(0, True)]
    variant_masks = {}
    for delta, sphere in variants[1:]:
        for _, s in scene.drop_duplicates("AOIFile").iterrows():
            code, shares, _ = make_masks(s.to_dict(), delta, sphere)
            variant_masks[(s.AOIFile, delta, sphere)] = (code, shares)
    trials, fixation_frames, issue_frames, variant_rows = [], [], [], []
    join = data.merge(scene, on=["SceneID", "OrderGroup", "Block", "WWR", "Complexity"], validate="many_to_one")
    if join.AOIFile.isna().any() or len(join) != len(data):
        raise ValueError("Scene-to-AOI join lost original trials")
    for number, (_, record) in enumerate(join.iterrows(), 1):
        raw, encoding = read_raw(record.CSVFile)
        identities = raw.User.dropna().map(canonical_name).unique()
        if len(identities) != 1 or identities[0] != record.Participant:
            raise ValueError(f"Raw User disagrees with mapping: {record.CSVFile}")
        timestamps = pd.to_numeric(raw["Recording Time Stamp[ms]"], errors="coerce")
        if timestamps.isna().any() or (timestamps.diff().dropna() < 0).any():
            raise ValueError("Invalid/nonmonotone raw timestamps")
        duration = float(timestamps.max() - timestamps.min()) / 1000
        left, right = (pd.to_numeric(raw[c], errors="coerce").eq(1) for c in ["Validity Left", "Validity Right"])
        valid = left & right
        tracking = float(valid.mean())
        events, issues = deduplicate_fixations(raw, config.get("coordinate_tolerance_px", 1.0))
        events = assign_events(events, masks[record.AOIFile])
        for c in KEY:
            events[c] = record[c]
            if not issues.empty:
                issues[c] = record[c]
        events["RawSource"] = record.CSVFile
        fixation_frames.append(events)
        if not issues.empty:
            issue_frames.append(issues)
        base = {c: record[c] for c in data.columns}
        quality = {"RecordingDuration": duration, "TotalSamples": len(raw), "ValidSamples": int(valid.sum()),
                   "ValidTrackingRatio": tracking, "SoftwareTrackingRatio": float(pd.to_numeric(raw["Tracking Ratio[%]"], errors="coerce").median()) / 100,
                   "LeftEyeValidRatio": float(left.mean()), "RightEyeValidRatio": float(right.mean()),
                   "FixationCount": len(events), "ValidSceneFixationCount": int(events.ValidSceneHit.sum()),
                   "OffStimulusFixationCount": int((~events.ValidSceneHit).sum()),
                   "OffStimulusTFDShare": float(events.loc[~events.ValidSceneHit, "FixationDuration"].sum()) / events.FixationDuration.sum() if events.FixationDuration.sum() > 0 else np.nan,
                   "RawEncoding": encoding, "FixationConflictCount": len(issues)}
        metrics = trial_metrics(events, areas[record.AOIFile], duration, tracking, record.Complexity)
        quality["QC50"] = tracking >= .5 and metrics["ValidSceneTFD"] > 0
        quality["QC60"] = tracking >= .6 and metrics["ValidSceneTFD"] > 0
        quality["QC70"] = tracking >= .7 and metrics["ValidSceneTFD"] > 0
        trials.append({**base, **quality, **metrics})
        for delta, sphere in variants[1:]:
            code, size = variant_masks[(record.AOIFile, delta, sphere)]
            changed = assign_events(events, code)
            variant_rows.append({**base, **quality, **trial_metrics(changed, size, duration, tracking, record.Complexity),
                                 "AuditVariant": "spherical" if sphere else f"boundary_{delta:+d}"})
        if number % 60 == 0:
            print(f"Independent raw eye trials: {number}/{len(join)}", flush=True)
    frame = pd.DataFrame(trials)
    unique(frame, KEY)
    sums = frame[[a + "Share" for a in AOIS]].sum(axis=1, min_count=1)
    if not np.allclose(sums.loc[frame.ValidSceneTFD > 0], 1, atol=1e-12, rtol=1e-10):
        raise ValueError("Independent shares do not partition ValidScene")
    save(frame, output / "independent_eye_trials.csv")
    save(pd.DataFrame(variant_rows), output / "independent_eye_sensitivity_trials.csv")
    events = pd.concat(fixation_frames, ignore_index=True)
    save(events, output / "independent_fixations.csv")
    save(events.sample(n=min(20, len(events)), random_state=20261006), output / "03_fixation_spot_check.csv")
    qc_cols = KEY + list(quality)
    save(frame[qc_cols], output / "04_trial_QC_audit.csv")
    formula_cols = KEY + [a + suffix for a in AOIS for suffix in ["TFD", "AreaShare", "Share"]] + ["ValidSceneTFD", *CORE]
    save(frame[formula_cols].sample(n=min(10, len(frame)), random_state=20261006), output / "05_outcome_formula_spot_check.csv")
    structural = frame.loc[frame.Complexity.eq(0), KEY + [c for c in frame if c.startswith("Equipment")]].copy()
    structural["StructuralNAValid"] = structural.filter(regex="^Equipment").isna().all(axis=1)
    save(structural, output / "06_zero_structural_NA_audit.csv")
    save(pd.DataFrame({"Field": frame.columns, "MissingRows": frame.isna().sum().values, "DataType": frame.dtypes.astype(str).values}), output / "07_trial_level_dataset_audit.csv")
    prev = frame[KEY + ["Block", "PositionWithinBlock", "WWR", "Complexity", "PreviousWWR", "PreviousComplexity"]].copy()
    prev["FirstPositionPreviousMissing"] = (~prev.PositionWithinBlock.eq(1)) | prev[["PreviousWWR", "PreviousComplexity"]].isna().all(axis=1)
    save(prev, output / "11_temporal_order_carryover_audit.csv")
    save(pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame(columns=KEY + ["FixationIndex", "Issue"]), output / "fixation_anomalies.csv")
    for source in sources:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Source changed during independent eye processing")
    dump(output / "source_hashes_after.json", sources)
    count = [{"Threshold": cut / 100, "Participants": int(frame.loc[frame[f"QC{cut}"], "Participant"].nunique()),
              "Trials": int(frame[f"QC{cut}"].sum())} for cut in [50, 60, 70]]
    dump(output / "processing_summary.json", {"candidate_participants": int(frame.Participant.nunique()), "raw_trials": len(frame),
         "qc": count, "fixations": len(events), "fixation_anomalies": sum(len(i) for i in issue_frames),
         "projection": sorted(scene.ProjectionType.astype(str).unique()), "variants": ["main", *sorted(pd.DataFrame(variant_rows).AuditVariant.unique())],
         "independence": "raw sources and approved measurement inputs only; no original processing/models/results read",
         "prior_context": "Project summaries were seen before this audit; computational isolation, not analyst blinding."})
    return frame
