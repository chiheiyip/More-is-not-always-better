function tests = test_eeg_onset_trimming
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'eeg_bandpower_pipeline'));
test_root = tempname;
mkdir(test_root);
stub_dir = fullfile(test_root, 'eeg_stubs');
mkdir(stub_dir);
write_stub(fullfile(stub_dir, 'pop_loadset.m'), [
    "function EEG = pop_loadset(varargin)" newline ...
    "global SYNTHETIC_EEG" newline ...
    "EEG = SYNTHETIC_EEG;" newline ...
    "end" newline]);
write_stub(fullfile(stub_dir, 'eeg_checkset.m'), [
    "function EEG = eeg_checkset(EEG)" newline ...
    "end" newline]);
addpath(stub_dir, '-begin');
testCase.TestData.stub_dir = stub_dir;
testCase.TestData.test_root = test_root;
end

function teardownOnce(testCase)
rmpath(testCase.TestData.stub_dir);
rmdir(testCase.TestData.test_root, 's');
end

function testVariantsRecomputeWaveformMetricsAndBoundaries(testCase)
global SYNTHETIC_EEG
fs = 100;
t = (0:(20 * fs)) / fs;
signal = sin(2 * pi * 6 * t);
signal(1:(10 * fs)) = 5 * sin(2 * pi * 20 * t(1:(10 * fs)));
SYNTHETIC_EEG = synthetic_eeg(signal, fs);
input_dir = fullfile(testCase.TestData.test_root, 'synthetic_input');
outdir = fullfile(testCase.TestData.test_root, 'synthetic_output');
mkdir(input_dir);
fclose(fopen(fullfile(input_dir, 'P01.set'), 'w'));
outputs = run_eeg_bandpower_pipeline( ...
    input_dir, outdir, 'StrictStructure', false, ...
    'PrimaryOnsetTrimS', 10, 'OnsetTrimVariantsS', [0 5 10 15]);
primary = readtable(outputs.all_subjects_scene_level);
variants = readtable(outputs.all_subjects_scene_level_onset_sensitivity);
verifyEqual(testCase, height(primary), 1);
verifyEqual(testCase, primary.onset_trim_s, 10);
verifyEqual(testCase, sort(variants.onset_trim_s)', [0 5 10 15]);
verifyEqual(testCase, sort(variants.onset_samples_removed)', [0 500 1000 1500]);
verifyEqual(testCase, sort(variants.analysis_dur_s, 'descend')', [20 15 10 5], 'AbsTol', 1e-10);
verifyGreaterThan(testCase, variants.F_theta(variants.onset_trim_s == 10), variants.F_theta(variants.onset_trim_s == 0));
verifyLessThan(testCase, variants.rms_mean_uV(variants.onset_trim_s == 10), variants.rms_mean_uV(variants.onset_trim_s == 0));
end

function testInsufficientPostTrimDurationKeepsInvalidRow(testCase)
global SYNTHETIC_EEG
fs = 100;
t = (0:(10.5 * fs)) / fs;
SYNTHETIC_EEG = synthetic_eeg(sin(2 * pi * 6 * t), fs);
input_dir = fullfile(testCase.TestData.test_root, 'short_input');
outdir = fullfile(testCase.TestData.test_root, 'short_output');
mkdir(input_dir);
fclose(fopen(fullfile(input_dir, 'P02.set'), 'w'));
outputs = run_eeg_bandpower_pipeline( ...
    input_dir, outdir, 'StrictStructure', false, ...
    'PrimaryOnsetTrimS', 10, 'OnsetTrimVariantsS', [0 10]);
variants = readtable(outputs.all_subjects_scene_level_onset_sensitivity, 'TextType', 'string');
short = variants(variants.onset_trim_s == 10, :);
verifyEqual(testCase, short.trim_status, "insufficient_post_trim_duration");
verifyFalse(testCase, logical(short.segment_valid_duration));
verifyTrue(testCase, isnan(short.F_theta));
verifyTrue(testCase, isnan(short.hf_ratio_20_40Hz));
end

function EEG = synthetic_eeg(signal, fs)
labels = {'F3', 'F4', 'P3', 'PZ', 'P4', 'O1', 'OZ', 'O2'};
EEG.data = repmat(signal, numel(labels), 1);
EEG.srate = fs;
EEG.pnts = size(EEG.data, 2);
EEG.chanlocs = struct('labels', labels);
EEG.event = struct('type', {'7', '8'}, 'latency', {1, EEG.pnts});
end

function write_stub(path, content)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', content);
end
