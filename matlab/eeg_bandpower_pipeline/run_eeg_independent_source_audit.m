function run_eeg_independent_source_audit(job_path)
% Independent SET/FDT -> event, epoch, Welch and QC features. No exporter helper.
% Existing preprocessing is read only; ICA audit/removal is outside this task.
job = jsondecode(fileread(job_path));
files = dir(fullfile(job.preprocessed_root, '*.set'));
rows = struct([]); records = struct([]); spectra = struct([]); event_rows = struct([]);
for person = 1:numel(files)
    source = fullfile(files(person).folder, files(person).name);
    S = load(source, '-mat');
    if isfield(S,'EEG'), S = S.EEG; end
    [~, name] = fileparts(source);
    datfile = char(string(S.data));
    fdt = fullfile(files(person).folder, datfile);
    remapped = false;
    if ~isfile(fdt)
        fdt = fullfile(files(person).folder, [name '.fdt']);
        remapped = true;
    end
    info = dir(fdt);
    if isempty(info) || info.bytes ~= double(S.nbchan)*double(S.pnts)*4
        error('Invalid FDT dimensions: %s',source);
    end
    fid = fopen(fdt,'r','ieee-le'); cleanup = onCleanup(@()fclose(fid));
    signal = fread(fid,[double(S.nbchan),double(S.pnts)],'single=>double');
    clear cleanup;
    fs = double(S.srate); labels = upper(string({S.chanlocs.labels}));
    channels = {{'F3','F4'},{'P3','PZ','P4'},{'O1','OZ','O2'}}; rois = {'F','P','O'};
    type = strings(numel(S.event),1); latency = zeros(numel(S.event),1);
    for k=1:numel(S.event)
        value=S.event(k).type; if iscell(value),value=value{1};end
        type(k)=strtrim(string(value));latency(k)=double(S.event(k).latency);
        ev=struct('Participant',string(name),'EventIndex',k,'Marker',type(k),'LatencySample',latency(k));
        if isempty(event_rows),event_rows=ev;else,event_rows(end+1)=ev;end %#ok<AGROW>
    end
    counter=0;
    for k=1:numel(type)-1
        if ~(type(k)=="7" && type(k+1)=="8"),continue;end
        counter=counter+1;
        start=max(1,round(latency(k)));stop=min(double(S.pnts),round(latency(k+1)));
        for trim=[0 5 10 15]
            first=start+round(trim*fs); duration=max((stop-start)/fs-trim,0);
            valid=stop>start && duration>=1 && first<=stop;
            row=struct('Participant',string(name),'GlobalTrialOrder',counter,'onset_trim_s',trim, ...
                'EventStartIndex',k,'EventEndIndex',k+1,'StartSample',start,'EndSample',stop, ...
                'AnalysisStartSample',min(first,stop),'srate',fs,'view_start_s',start/fs,'view_end_s',stop/fs, ...
                'analysis_dur_s',duration,'segment_valid_duration',valid, ...
                'nan_fraction',NaN,'flat_fraction',NaN,'hf_ratio_20_40Hz',NaN,'rms_mean_uV',NaN,'peak_to_peak_uV',NaN);
            if valid
                x=signal(:,first:stop);
                row.nan_fraction=mean(~isfinite(x(:)));
                rms_values=nan(size(x,1),1);ptp=nan(size(x,1),1);flat=false(size(x,1),1);
                for c=1:size(x,1)
                    a=x(c,isfinite(x(c,:)));
                    flat(c)=isempty(a)||std(a)<1e-6;
                    if ~isempty(a),rms_values(c)=sqrt(mean(a.^2));ptp(c)=max(a)-min(a);end
                end
                row.flat_fraction=mean(flat);row.rms_mean_uV=mean(rms_values,'omitnan');row.peak_to_peak_uV=mean(ptp,'omitnan');
                [f,p]=spectrum(mean(x,1,'omitnan'),fs);
                row.hf_ratio_20_40Hz=integral(f,p,[20 40])/integral(f,p,[1 45]);
            else
                x=nan(size(signal,1),0);
            end
            for r=1:3
                [present,idx]=ismember(upper(string(channels{r})),labels);
                if ~all(present),error('Missing required ROI channel');end
                [f,p]=spectrum(mean(x(idx,:),1,'omitnan'),fs);
                a=integral(f,p,[4 7]);b=integral(f,p,[8 12]);c=integral(f,p,[13 30]);
                den45=integral(f,p,[1 45]);den40=integral(f,p,[1 40]);tail=integral(f,p,[40 45]);
                for band={'theta','alpha','beta'}
                    v=struct('theta',a,'alpha',b,'beta',c);
                    row.([rois{r} '_' band{1} '_absolute'])=v.(band{1});
                    row.([rois{r} '_' band{1} '_relative'])=v.(band{1})/den45;
                    row.([rois{r} '_' band{1} '_relative_1_40'])=v.(band{1})/den40;
                end
                row.([rois{r} '_total_1_45'])=den45;row.([rois{r} '_total_1_40'])=den40;
                row.([rois{r} '_power_40_45'])=tail;
                if numel(spectra)<24 && valid
                    sp=struct('Participant',string(name),'GlobalTrialOrder',counter,'onset_trim_s',trim,'ROI',rois{r},'f',f,'pxx',p);
                    if isempty(spectra),spectra=sp;else,spectra(end+1)=sp;end %#ok<AGROW>
                end
            end
            if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
        end
    end
    record=struct('Participant',string(name),'SET',string(source),'FDT',string(fdt),'FDTRemapped',remapped, ...
        'Channels',double(S.nbchan),'Samples',double(S.pnts),'srate',fs,'ViewEvents',counter);
    if isempty(records),records=record;else,records(end+1)=record;end %#ok<AGROW>
    fprintf('Independent EEG source: %d/%d, %s, %d trials\n',person,numel(files),name,counter);
end
if ~isfolder(job.outdir),mkdir(job.outdir);end
writetable(struct2table(rows),fullfile(job.outdir,'independent_eeg_source_features.csv'),'Encoding','UTF-8');
writetable(struct2table(records),fullfile(job.outdir,'independent_eeg_source_inventory.csv'),'Encoding','UTF-8');
writetable(struct2table(event_rows),fullfile(job.outdir,'independent_eeg_set_events.csv'),'Encoding','UTF-8');
save(fullfile(job.outdir,'independent_eeg_waveform_spot_spectra.mat'),'spectra','-v7');
end

function [f,p]=spectrum(x,fs)
x=x(:);x=x(isfinite(x));
if numel(x)<max(8,round(fs)),f=[];p=[];return;end
window=min(numel(x),max(round(2*fs),8));overlap=floor(window/2);nfft=max(2^nextpow2(window),window);
[p,f]=pwelch(x,window,overlap,nfft,fs);
end

function v=integral(f,p,bounds)
mask=f>=bounds(1)&f<=bounds(2);
if ~any(mask),v=NaN;else,v=trapz(f(mask),p(mask));end
end
