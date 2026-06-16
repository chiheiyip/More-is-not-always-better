function outputs = run_eeg_bandpower_pipeline(input_path, outdir, varargin)
%RUN_EEG_BANDPOWER_PIPELINE Export scene-level EEG bandpower and QC tables.
%
% Input data are assumed to be preprocessed EEGLAB .set files. This function
% performs experiment marker segmentation, ROI bandpower extraction, view-gray
% pairing, and per-segment QC metric export. Formal inclusion/exclusion is
% performed by the Python pipeline from these auditable metrics.

if nargin < 1 || isempty(input_path)
    input_path = uigetdir(pwd, 'Select a folder containing .set files');
end
if nargin < 2 || isempty(outdir)
    if isfolder(input_path)
        outdir = fullfile(input_path, 'bandpower_outputs');
    else
        outdir = fullfile(fileparts(input_path), 'bandpower_outputs');
    end
end

p = inputParser;
addRequired(p, 'input_path', @(x) ischar(x) || isstring(x));
addRequired(p, 'outdir', @(x) ischar(x) || isstring(x));
addParameter(p, 'ConfigPath', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'ViewStartMarker', '7');
addParameter(p, 'ViewEndMarker', '8');
addParameter(p, 'StrictStructure', true);
addParameter(p, 'MinSegmentDurationS', 1.0);
addParameter(p, 'LegacyHfThreshold', 0.4);
addParameter(p, 'Bands', default_bands());
addParameter(p, 'Rois', default_rois());
parse(p, input_path, outdir, varargin{:});
opts = p.Results;
opts = apply_config(opts);

if exist('pop_loadset', 'file') ~= 2
    error('EEGLAB is required. Add EEGLAB to the MATLAB path before running this function.');
end

files = resolve_set_files(char(input_path));
outdir = char(outdir);
summary_dir = fullfile(outdir, 'summary');
subject_dir = fullfile(outdir, 'subjects');
if ~exist(summary_dir, 'dir'), mkdir(summary_dir); end
if ~exist(subject_dir, 'dir'), mkdir(subject_dir); end

all_scene = table();
all_qc = table();
all_pairs = table();
for i = 1:numel(files)
    [fp, base, ext] = fileparts(files{i});
    fprintf('Processing EEG subject %s (%d/%d)\n', base, i, numel(files));
    EEG = pop_loadset('filename', [base ext], 'filepath', fp);
    EEG = eeg_checkset(EEG);
    sub_out = fullfile(subject_dir, base);
    if ~exist(sub_out, 'dir'), mkdir(sub_out); end
    [segments, scene_rows, qc_rows, pairs] = export_subject(EEG, base, opts);
    writetable(segments, fullfile(sub_out, [base '_bandpower_roi.csv']));
    writetable(scene_rows, fullfile(sub_out, [base '_scene_level.csv']));
    writetable(qc_rows, fullfile(sub_out, [base '_qc.csv']));
    writetable(pairs, fullfile(sub_out, [base '_pairs_check.csv']));
    write_methods_snapshot(fullfile(sub_out, [base '_methods_snapshot.md']), opts);
    all_scene = [all_scene; scene_rows]; %#ok<AGROW>
    all_qc = [all_qc; qc_rows]; %#ok<AGROW>
    all_pairs = [all_pairs; pairs]; %#ok<AGROW>
end

outputs = struct();
outputs.all_subjects_scene_level = fullfile(summary_dir, 'all_subjects_scene_level.csv');
outputs.all_subjects_qc = fullfile(summary_dir, 'all_subjects_qc.csv');
outputs.all_subjects_pairs_check = fullfile(summary_dir, 'all_subjects_pairs_check.csv');
writetable(all_scene, outputs.all_subjects_scene_level);
writetable(all_qc, outputs.all_subjects_qc);
writetable(all_pairs, outputs.all_subjects_pairs_check);
fprintf('Wrote %s\n', outputs.all_subjects_scene_level);
end

function [segments, scene_rows, qc_rows, pairs] = export_subject(EEG, subject_id, opts)
events = EEG.event;
types = arrayfun(@(e) marker_to_string(e.type), events, 'UniformOutput', false);
latencies = arrayfun(@(e) double(e.latency), events);
segments = table();
view_count = 0;
for i = 1:(numel(events) - 1)
    m0 = marker_to_string(types{i});
    m1 = marker_to_string(types{i + 1});
    cond = classify_transition(m0, m1);
    if strcmp(cond, 'skip')
        continue;
    end
    start_sample = max(1, round(latencies(i)));
    end_sample = min(size(EEG.data, 2), round(latencies(i + 1)));
    if end_sample <= start_sample
        continue;
    end
    if strcmp(cond, 'view')
        view_count = view_count + 1;
        scene_id = view_count;
    elseif strcmp(cond, 'gray') || startsWith(cond, 'questionnaire')
        scene_id = view_count;
    else
        scene_id = NaN;
    end
    segment = double(EEG.data(:, start_sample:end_sample));
    dur_s = (end_sample - start_sample) / EEG.srate;
    row = table(string(subject_id), string(subject_id), string(cond), string(m0), string(m1), ...
        scene_id, ceil(max(scene_id, 1) / 6), mod(max(scene_id, 1) - 1, 6) + 1, ...
        start_sample / EEG.srate, end_sample / EEG.srate, dur_s, ...
        'VariableNames', {'subject_id','participant_id','cond','m0','m1','scene_id','block_id','cycle_in_block','start_s','end_s','dur_s'});
    row = append_bandpower_columns(row, segment, EEG, opts);
    row = append_qc_columns(row, segment, EEG.srate, dur_s, opts);
    segments = [segments; row]; %#ok<AGROW>
end

if opts.StrictStructure && view_count ~= 12
    warning('Subject %s has %d view segments; expected 12.', subject_id, view_count);
end

scene_rows = segments(strcmp(string(segments.cond), 'view'), :);
if ~isempty(scene_rows)
    scene_rows.view_start_s = scene_rows.start_s;
    scene_rows.view_end_s = scene_rows.end_s;
    scene_rows.view_dur_s = scene_rows.dur_s;
    scene_rows = scene_rows(:, scene_level_columns(scene_rows));
end
qc_rows = segments(:, qc_columns(segments));
pairs = build_pairs_table(segments);
end

function row = append_bandpower_columns(row, segment, EEG, opts)
labels = upper(string({EEG.chanlocs.labels}));
roi_names = fieldnames(opts.Rois);
for r = 1:numel(roi_names)
    roi_name = roi_names{r};
    roi_channels = upper(string(opts.Rois.(roi_name)));
    [present, idx] = ismember(roi_channels, labels);
    idx = idx(present);
    if isempty(idx)
        roi_signal = nan(1, size(segment, 2));
    else
        roi_signal = mean(segment(idx, :), 1, 'omitnan');
    end
    theta = bandpower_welch(roi_signal, EEG.srate, opts.Bands.theta);
    alpha = bandpower_welch(roi_signal, EEG.srate, opts.Bands.alpha);
    beta = bandpower_welch(roi_signal, EEG.srate, opts.Bands.beta);
    low_beta = bandpower_welch(roi_signal, EEG.srate, opts.Bands.low_beta);
    high_beta = bandpower_welch(roi_signal, EEG.srate, opts.Bands.high_beta);
    low_gamma = bandpower_welch(roi_signal, EEG.srate, opts.Bands.low_gamma);
    prefix = roi_prefix(roi_name);
    row.([prefix '_theta']) = theta;
    row.([prefix '_alpha']) = alpha;
    row.([prefix '_beta']) = beta;
    row.([prefix '_low_beta']) = low_beta;
    row.([prefix '_high_beta']) = high_beta;
    row.([prefix '_low_gamma']) = low_gamma;
    row.([prefix '_TAR']) = safe_ratio(theta, alpha);
    row.([prefix '_TBR']) = safe_ratio(theta, beta);
    row.([prefix '_BA']) = safe_ratio(beta, alpha);
end
end

function row = append_qc_columns(row, segment, fs, dur_s, opts)
all_signal = mean(segment, 1, 'omitnan');
hf = bandpower_welch(all_signal, fs, [20 40]);
total = bandpower_welch(all_signal, fs, opts.Bands.totalBand40);
row.hf_ratio_20_40Hz = safe_ratio(hf, total);
row.rms_mean_uV = channel_mean_rms(segment);
row.peak_to_peak_uV = channel_mean_peak_to_peak(segment);
row.nan_fraction = mean(~isfinite(segment(:)));
row.flat_fraction = flat_channel_fraction(segment);
row.near_boundary = contains(lower(string(row.m0)), 'boundary') || contains(lower(string(row.m1)), 'boundary');
row.segment_valid_duration = dur_s >= opts.MinSegmentDurationS;
row.eeg_legacy_hf_flag = row.hf_ratio_20_40Hz > opts.LegacyHfThreshold;
end

function pairs = build_pairs_table(segments)
pairs = table();
views = find(strcmp(string(segments.cond), 'view'));
for i = reshape(views, 1, [])
    scene_id = segments.scene_id(i);
    later = find(strcmp(string(segments.cond), 'gray') & segments.scene_id == scene_id & (1:height(segments))' > i, 1, 'first');
    if isempty(later)
        gray_start = NaN; gray_end = NaN; gray_dur = NaN; delta_alpha = NaN; status = "missing_gray";
    else
        gray_start = segments.start_s(later);
        gray_end = segments.end_s(later);
        gray_dur = segments.dur_s(later);
        if ismember('O_alpha', segments.Properties.VariableNames)
            delta_alpha = segments.O_alpha(later) - segments.O_alpha(i);
        else
            delta_alpha = NaN;
        end
        status = "paired";
    end
    row = table(segments.subject_id(i), scene_id, segments.block_id(i), segments.cycle_in_block(i), status, ...
        segments.start_s(i), segments.end_s(i), segments.dur_s(i), gray_start, gray_end, gray_dur, delta_alpha, ...
        'VariableNames', {'subject_id','scene_id','block_id','cycle_in_block','pair_status','view_start_s','view_end_s','view_dur_s','gray_start_s','gray_end_s','gray_dur_s','delta_O_alpha'});
    pairs = [pairs; row]; %#ok<AGROW>
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

function files = resolve_set_files(input_path)
if isfolder(input_path)
    d = dir(fullfile(input_path, '*.set'));
    files = arrayfun(@(x) fullfile(x.folder, x.name), d, 'UniformOutput', false);
else
    files = {input_path};
end
if isempty(files)
    error('No .set files found at %s', input_path);
end
end

function cond = classify_transition(m0, m1)
if strcmp(m0, '1') && strcmp(m1, '2')
    cond = 'adapt';
elseif strcmp(m0, '2') && strcmp(m1, '3')
    cond = 'intro';
elseif strcmp(m0, '3') && strcmp(m1, '4')
    cond = 'eyes_closed';
elseif strcmp(m0, '4') && strcmp(m1, '9')
    cond = 'eyes_open';
elseif strcmp(m0, '7') && strcmp(m1, '8')
    cond = 'view';
elseif strcmp(m0, '8') && strcmp(m1, '9')
    cond = 'questionnaire_small';
elseif strcmp(m0, '8') && (strcmp(m1, '5') || strcmp(m1, '6'))
    cond = 'questionnaire_big';
elseif strcmp(m0, '9') && (strcmp(m1, '7') || strcmp(m1, '8') || strcmp(m1, '5') || strcmp(m1, '6'))
    cond = 'gray';
elseif strcmp(m0, '5') && strcmp(m1, '7')
    cond = 'rest';
else
    cond = 'skip';
end
end

function cols = scene_level_columns(T)
base = {'subject_id','participant_id','scene_id','block_id','cycle_in_block','view_start_s','view_end_s','view_dur_s'};
metric_cols = T.Properties.VariableNames(contains(T.Properties.VariableNames, {'_theta','_alpha','_beta','_low_beta','_high_beta','_low_gamma','_TAR','_TBR','_BA'}));
qc = {'hf_ratio_20_40Hz','rms_mean_uV','peak_to_peak_uV','nan_fraction','flat_fraction','near_boundary','segment_valid_duration','eeg_legacy_hf_flag'};
cols = [base, metric_cols, qc];
cols = cols(ismember(cols, T.Properties.VariableNames));
end

function cols = qc_columns(T)
cols = {'subject_id','participant_id','cond','scene_id','block_id','cycle_in_block','start_s','end_s','dur_s','hf_ratio_20_40Hz','rms_mean_uV','peak_to_peak_uV','nan_fraction','flat_fraction','near_boundary','segment_valid_duration','eeg_legacy_hf_flag'};
cols = cols(ismember(cols, T.Properties.VariableNames));
end

function prefix = roi_prefix(name)
name = lower(string(name));
if name == "front" || name == "f"
    prefix = 'F';
elseif name == "par" || name == "p"
    prefix = 'P';
elseif name == "occ" || name == "o"
    prefix = 'O';
else
    prefix = char(upper(extractBefore(name + "_", "_")));
end
end

function value = safe_ratio(num, den)
if ~isfinite(num) || ~isfinite(den) || den == 0
    value = NaN;
else
    value = num / den;
end
end

function value = channel_mean_rms(segment)
vals = nan(size(segment, 1), 1);
for i = 1:size(segment, 1)
    x = segment(i, :);
    x = x(isfinite(x));
    if ~isempty(x), vals(i) = sqrt(mean(x .^ 2)); end
end
value = mean(vals, 'omitnan');
end

function value = channel_mean_peak_to_peak(segment)
vals = nan(size(segment, 1), 1);
for i = 1:size(segment, 1)
    x = segment(i, :);
    x = x(isfinite(x));
    if ~isempty(x), vals(i) = max(x) - min(x); end
end
value = mean(vals, 'omitnan');
end

function value = flat_channel_fraction(segment)
flat = false(size(segment, 1), 1);
for i = 1:size(segment, 1)
    x = segment(i, :);
    x = x(isfinite(x));
    flat(i) = isempty(x) || std(x) < 1e-6;
end
value = mean(flat);
end

function text = marker_to_string(value)
if iscell(value), value = value{1}; end
if isnumeric(value)
    text = char(string(value));
elseif ischar(value) || isstring(value)
    text = char(strtrim(string(value)));
else
    text = char(string(value));
end
end

function rois = default_rois()
rois = struct();
rois.F = {'F3','F4'};
rois.P = {'P3','PZ','P4'};
rois.O = {'O1','OZ','O2'};
end

function bands = default_bands()
bands = struct();
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.low_beta = [13 20];
bands.high_beta = [20 30];
bands.low_gamma = [30 45];
bands.totalBand40 = [1 45];
bands.totalBand30 = [1 30];
end

function opts = apply_config(opts)
if isempty(opts.ConfigPath) || exist(char(opts.ConfigPath), 'file') ~= 2
    return;
end
try
    raw = fileread(char(opts.ConfigPath));
    cfg = jsondecode(raw);
    if isfield(cfg, 'bands'), opts.Bands = merge_struct(opts.Bands, cfg.bands); end
    if isfield(cfg, 'roi')
        r = cfg.roi;
        if isfield(r, 'front'), opts.Rois.F = cellstr(string(r.front)); end
        if isfield(r, 'par'), opts.Rois.P = cellstr(string(r.par)); end
        if isfield(r, 'occ'), opts.Rois.O = cellstr(string(r.occ)); end
    end
    if isfield(cfg, 'strict_structure'), opts.StrictStructure = logical(cfg.strict_structure); end
    if isfield(cfg, 'qc_hf_threshold'), opts.LegacyHfThreshold = double(cfg.qc_hf_threshold); end
catch ME
    warning('Could not read EEG config %s: %s', char(opts.ConfigPath), ME.message);
end
end

function out = merge_struct(out, incoming)
names = fieldnames(incoming);
for i = 1:numel(names)
    out.(names{i}) = incoming.(names{i});
end
end

function write_methods_snapshot(path, opts)
fid = fopen(path, 'w');
if fid == -1, return; end
fprintf(fid, '# EEG Bandpower Methods Snapshot\n\n');
fprintf(fid, '- Input: preprocessed EEGLAB .set files.\n');
fprintf(fid, '- Segmentation: marker state machine with 7->8 as scene viewing.\n');
fprintf(fid, '- Bandpower: Welch PSD integrated within configured bands.\n');
fprintf(fid, '- QC metrics: HF ratio, RMS, peak-to-peak, NaN fraction, flat fraction, boundary flag, valid duration.\n');
fprintf(fid, '- Legacy HF threshold: %.3f, exported for audit/sensitivity only.\n', opts.LegacyHfThreshold);
fprintf(fid, '- Formal robust exclusion is applied downstream in Python, not by this exporter.\n');
fprintf(fid, '- Low-gamma and high-beta are exploratory because high-frequency EEG is sensitive to muscle artifacts.\n');
fclose(fid);
end
