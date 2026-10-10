function snapshot = assert_analysis_runtime()
% Verify in the executing MATLAB process before any numerical export.
folder=fileparts(mfilename('fullpath'));
lock=jsondecode(fileread(fullfile(folder,'..','configs','analysis_matlab.lock.json')));
if exist('pop_loadset','file')~=2
    root=fileparts(fileparts(fileparts(lock.snapshot.functions.pop_loadset.path)));
    addpath(root); addpath(genpath(fullfile(root,'functions')));
end
snapshot=analysis_runtime_snapshot();
assert(isequaln(snapshot,lock.snapshot),'Formal MATLAB/EEGLAB environment mismatch; restore registered runtime.');
end
