function snapshot = analysis_runtime_snapshot(eeglab_root)
% Read-only fingerprint of the functions actually resolved by MATLAB.
if nargin > 0 && ~isempty(eeglab_root)
    addpath(eeglab_root);
    addpath(genpath(fullfile(eeglab_root,'functions')));
end
snapshot = struct('matlab',version,'release',version('-release'), ...
    'platform',computer,'blas',version('-blas'),'lapack',version('-lapack'), ...
    'threads',maxNumCompThreads);
state=rng; snapshot.rng_type=state.Type;
v = ver('signal'); assert(numel(v)==1,'Signal Processing Toolbox unavailable');
snapshot.signal_version = v.Version;
snapshot.eeglab_version = eeg_getversion;
options=fullfile(getenv('USERPROFILE'),'eeg_options.m');
snapshot.user_options_sha256='';
if isfile(options), snapshot.user_options_sha256=digest_file(options); end
names = {'pwelch','trapz','pop_loadset','eeg_checkset','eeg_getdatact','floatread','eeg_getversion'};
snapshot.functions = struct();
for i=1:numel(names)
    p=which(names{i}); assert(isfile(p),'Runtime function missing: %s',names{i});
    snapshot.functions.(names{i})=struct('path',p,'sha256',digest_file(p));
end
end

function value=digest_file(path)
md=java.security.MessageDigest.getInstance('SHA-256');
fid=fopen(path,'rb'); assert(fid>=0); c=onCleanup(@()fclose(fid));
while true
    bytes=fread(fid,1048576,'*uint8'); if isempty(bytes),break;end
    md.update(bytes);
end
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
