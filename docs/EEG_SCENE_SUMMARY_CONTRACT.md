# EEG Scene-Level Summary Contract

This repository treats raw EEGLAB `.set/.fdt` files as read-only inputs. Formal Python analysis starts from a scene-level EEG CSV, usually exported by `matlab/run_eeg_bandpower_from_set.m` to `summary/all_subjects_scene_level.csv`.

Minimum required columns:

- `participant_id` or `subject_id`
- `scene_id`
- At least one core EEG metric: `O_theta`, `F_theta`, or `O_alpha`

Strongly recommended timing columns:

- `view_start_s`
- `view_end_s`
- `view_dur_s` or an accepted duration alias: `duration_s`, `dur_s`, `view_duration_s`, `eeg_view_dur_s`

Strongly recommended QC columns:

- `hf_ratio_20_40Hz`
- `rms_mean_uV`
- `peak_to_peak_uV`
- `nan_fraction`
- `flat_fraction`
- `segment_valid_duration`

Validation command:

```bash
python scripts/validate_eeg_scene_summary.py path/to/all_subjects_scene_level.csv
```

Run the raw `.set/.fdt` exporter from Python when MATLAB and EEGLAB are available:

```bash
python scripts/run_eeg_from_raw.py --eeg_root E:\26\补\脑电数据 --eeglab_root "D:\Program Files\MATLAB\eeglab" --outdir outputs/eeg_realdata
```

For a small smoke test, pass one `.set` file as `--eeg_root` and write to a scratch outdir.

JSON output for automated checks:

```bash
python scripts/validate_eeg_scene_summary.py path/to/all_subjects_scene_level.csv --json
```

The validator is read-only. It does not edit EEG raw files or generated summary files.
