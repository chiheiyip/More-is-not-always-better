function outputs = run_eeg_bandpower_from_set(raw_root, outdir, varargin)
%RUN_EEG_BANDPOWER_FROM_SET Compatibility wrapper for the EEG export pipeline.
%
% Existing calls keep working:
%   run_eeg_bandpower_from_set('E:/eeg_raw', 'outputs/eeg')
%
% The implementation lives in matlab/eeg_bandpower_pipeline and exports
% scene-level EEG metrics plus auditable QC columns for the Python pipeline.

if nargin < 2
    outdir = fullfile(raw_root, 'bandpower_outputs');
end

this_dir = fileparts(mfilename('fullpath'));
pipeline_dir = fullfile(this_dir, 'eeg_bandpower_pipeline');
if exist(pipeline_dir, 'dir') == 7
    addpath(pipeline_dir, '-begin');
else
    error('Missing EEG pipeline directory: %s', pipeline_dir);
end

outputs = run_eeg_bandpower_pipeline(raw_root, outdir, varargin{:});
end
