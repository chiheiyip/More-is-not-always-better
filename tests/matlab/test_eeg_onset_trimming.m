function tests = test_eeg_onset_trimming
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'eeg_bandpower_pipeline'));
test_root = tempname;
mkdir(test_root);
addpath(fullfile(root,'matlab'));
assert_analysis_runtime();
testCase.TestData.test_root = test_root;
end

function teardownOnce(testCase)
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
EEG=SYNTHETIC_EEG;save(fullfile(input_dir,'P01.set'),'EEG','-v7');
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
EEG=SYNTHETIC_EEG;save(fullfile(input_dir,'P02.set'),'EEG','-v7');
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

function testCommittedParallelConfigSelectsReferenceExport(testCase)
global SYNTHETIC_EEG
fs = 100;
t = (0:(20 * fs)) / fs;
SYNTHETIC_EEG = synthetic_eeg(sin(2 * pi * 6 * t), fs);
input_dir = fullfile(testCase.TestData.test_root, 'config_input');
outdir = fullfile(testCase.TestData.test_root, 'config_output');
mkdir(input_dir);
EEG=SYNTHETIC_EEG;save(fullfile(input_dir,'P03.set'),'EEG','-v7');
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
outputs = run_eeg_bandpower_pipeline(input_dir, outdir, ...
    'StrictStructure', false, 'ConfigPath', fullfile(root, 'configs', 'eeg_analysis.json'));
primary = readtable(outputs.all_subjects_scene_level);
variants = readtable(outputs.all_subjects_scene_level_onset_sensitivity);
verifyEqual(testCase, primary.onset_trim_s, 10);
verifyEqual(testCase, sort(variants.onset_trim_s)', [0 5 10 15]);
reference = variants(variants.onset_trim_s == 10, :);
verifyEqual(testCase, primary.F_theta, reference.F_theta, 'AbsTol', 1e-12);
verifyEqual(testCase, primary.hf_ratio_20_40Hz, reference.hf_ratio_20_40Hz, 'AbsTol', 1e-12);
end

function EEG = synthetic_eeg(signal, fs)
labels = {'F3', 'F4', 'P3', 'PZ', 'P4', 'O1', 'OZ', 'O2'};
EEG=eeg_emptyset;
EEG.data = repmat(signal, numel(labels), 1);
EEG.nbchan=numel(labels);EEG.trials=1;
EEG.srate = fs;
EEG.pnts = size(EEG.data, 2);
EEG.xmin=0;EEG.xmax=(EEG.pnts-1)/fs;
EEG.chanlocs = struct('labels', labels);
EEG.event = struct('type', {'7', '8'}, 'latency', {1, EEG.pnts});
end

function write_stub(path, content)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', content);
end
