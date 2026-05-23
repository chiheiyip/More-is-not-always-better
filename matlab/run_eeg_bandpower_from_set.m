function run_eeg_bandpower_from_set(raw_root, outdir, varargin)
%RUN_EEG_BANDPOWER_FROM_SET Export scene-level EEG bandpower from EEGLAB .set/.fdt files.
%
% This is the end-to-end EEG entry for the fusion project. It reads every
% .set file in raw_root, pairs it with the sibling .fdt file, segments view
% periods from marker 7 to the next marker 8, computes ROI bandpower, and
% writes outputs/eeg/summary/all_subjects_scene_level.csv.
%
% Example:
%   run_eeg_bandpower_from_set('E:/eeg原始文件', 'outputs/eeg')

p = inputParser;
addRequired(p, 'raw_root', @(x) ischar(x) || isstring(x));
addRequired(p, 'outdir', @(x) ischar(x) || isstring(x));
addParameter(p, 'ViewStartMarker', '7');
addParameter(p, 'ViewEndMarker', '8');
addParameter(p, 'Bands', struct('theta', [4 7], 'alpha', [8 13], 'beta', [13 30]));
addParameter(p, 'Rois', default_rois());
parse(p, raw_root, outdir, varargin{:});

raw_root = char(p.Results.raw_root);
outdir = char(p.Results.outdir);
summary_dir = fullfile(outdir, 'summary');
subject_dir = fullfile(outdir, 'subjects');
if ~exist(summary_dir, 'dir'), mkdir(summary_dir); end
if ~exist(subject_dir, 'dir'), mkdir(subject_dir); end

if exist('pop_loadset', 'file') ~= 2
    error('EEGLAB is required. Add EEGLAB to the MATLAB path before running this function.');
end

set_files = dir(fullfile(raw_root, '*.set'));
all_rows = table();
for i = 1:numel(set_files)
    set_path = fullfile(set_files(i).folder, set_files(i).name);
    [~, subject_id] = fileparts(set_path);
    fprintf('Processing EEG subject %s (%d/%d)\n', subject_id, i, numel(set_files));

    EEG = pop_loadset('filename', set_files(i).name, 'filepath', set_files(i).folder);
    subject_rows = export_subject_scene_bandpower(EEG, subject_id, p.Results);
    subject_out = fullfile(subject_dir, [subject_id '_scene_level.csv']);
    writetable(subject_rows, subject_out);
    all_rows = [all_rows; subject_rows]; %#ok<AGROW>
end

summary_out = fullfile(summary_dir, 'all_subjects_scene_level.csv');
writetable(all_rows, summary_out);
fprintf('Wrote %s\n', summary_out);
end

function rows = export_subject_scene_bandpower(EEG, subject_id, opts)
events = EEG.event;
types = arrayfun(@(e) marker_to_string(e.type), events, 'UniformOutput', false);
latencies = arrayfun(@(e) double(e.latency), events);
start_marker = marker_to_string(opts.ViewStartMarker);
end_marker = marker_to_string(opts.ViewEndMarker);
start_idx = find(strcmp(types, start_marker));

rows = table();
scene_id = 0;
for s = reshape(start_idx, 1, [])
    later_end = find(strcmp(types, end_marker) & latencies > latencies(s), 1, 'first');
    if isempty(later_end)
        warning('Subject %s marker %s at %.0f has no following marker %s. Skipping.', ...
            subject_id, start_marker, latencies(s), end_marker);
        continue;
    end
    start_sample = max(1, round(latencies(s)));
    end_sample = min(size(EEG.data, 2), round(latencies(later_end)));
    if end_sample <= start_sample
        continue;
    end
    scene_id = scene_id + 1;
    segment = double(EEG.data(:, start_sample:end_sample));
    presentation_pos = scene_id;
    row = table(string(subject_id), scene_id, presentation_pos, ceil(scene_id / 6), mod(scene_id - 1, 6) + 1, ...
        start_sample / EEG.srate, end_sample / EEG.srate, (end_sample - start_sample) / EEG.srate, ...
        'VariableNames', {'subject_id','scene_id','presentation_pos','block_id','cycle_in_block','view_start_s','view_end_s','view_dur_s'});
    row = append_bandpower_columns(row, segment, EEG, opts.Rois, opts.Bands);
    rows = [rows; row]; %#ok<AGROW>
end
end

function row = append_bandpower_columns(row, segment, EEG, rois, bands)
labels = arrayfun(@(c) string(c.labels), EEG.chanlocs);
roi_names = fieldnames(rois);
band_names = fieldnames(bands);
for r = 1:numel(roi_names)
    roi_name = roi_names{r};
    roi_channels = string(rois.(roi_name));
    [present, idx] = ismember(roi_channels, labels);
    idx = idx(present);
    if isempty(idx)
        warning('ROI %s has no matching channels in this EEG file.', roi_name);
        roi_signal = nan(1, size(segment, 2));
    else
        roi_signal = mean(segment(idx, :), 1, 'omitnan');
    end
    for b = 1:numel(band_names)
        band_name = band_names{b};
        value = bandpower_welch(roi_signal, EEG.srate, bands.(band_name));
        row.([roi_name '_' band_name]) = value;
    end
end
end

function value = bandpower_welch(signal, fs, band)
signal = signal(:);
signal = signal(isfinite(signal));
if numel(signal) < max(8, round(fs))
    value = NaN;
    return;
end
win = min(numel(signal), max(round(2 * fs), 8));
noverlap = floor(win / 2);
nfft = max(2 ^ nextpow2(win), win);
[pxx, f] = pwelch(signal, win, noverlap, nfft, fs);
mask = f >= band(1) & f <= band(2);
if ~any(mask)
    value = NaN;
else
    value = trapz(f(mask), pxx(mask));
end
end

function text = marker_to_string(value)
if isnumeric(value)
    text = string(value);
elseif ischar(value) || isstring(value)
    text = string(strtrim(char(value)));
else
    text = string(value);
end
end

function rois = default_rois()
rois = struct();
rois.F = {'Fz','F3','F4','FCz','FC1','FC2'};
rois.P = {'Pz','P3','P4','CPz','CP1','CP2'};
rois.O = {'Oz','O1','O2','POz','PO3','PO4'};
end
