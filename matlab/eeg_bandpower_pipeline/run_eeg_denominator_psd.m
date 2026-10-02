function run_eeg_denominator_psd(config_path)
% Cache complete spectra only for the frozen historical sample. Source hashes
% and cache identity are supplied and verified before/after by the Python runner.
cfg=jsondecode(fileread(config_path));
trials=readtable(cfg.trials,'TextType','string','VariableNamingRule','preserve');
roi_names={'F','P','O'}; roi_channels={{'F3','F4'},{'P3','PZ','P4'},{'O1','OZ','O2'}};
rows=cell(height(trials)*3,16); count=0;
participants=unique(trials.Participant,'stable');
for i=1:numel(participants)
    participant=participants(i);
    source=cfg.sources(i);
    assert(string(source.participant)==participant,'Source ordering mismatch');
    E=load(source.set_path,'-mat','srate','pnts','nbchan','chanlocs','history','ref','icaweights','reject');
    fid=fopen(source.fdt_path,'r','ieee-le'); assert(fid>=0,'Cannot open FDT');
    cleanup=onCleanup(@() fclose(fid));
    waveform=fread(fid,[E.nbchan,E.pnts],'single=>double'); clear cleanup;
    assert(size(waveform,2)==E.pnts,'Waveform dimensions differ');
    labels={E.chanlocs.labels}; sub=trials(trials.Participant==participant,:);
    spectra=cell(height(sub),3);
    for j=1:height(sub)
        for r=1:3
            s=eeg_denominator_spectrum(waveform,labels,roi_channels{r},E.srate,...
                sub.view_start_s(j),sub.view_end_s(j),sub.onset_trim_s(j));
            spectra{j,r}=s; count=count+1;
            rows(count,:)={participant,sub.GlobalTrialOrder(j),sub.onset_trim_s(j),roi_names{r},...
                s.powers(1),s.powers(2),s.powers(3),s.powers(4),s.powers(5),s.powers(6),...
                s.first_sample,s.last_sample,s.window,s.overlap,s.nfft,E.srate};
        end
    end
    metadata=struct('cache_key',cfg.cache_key,'source',source,'trials',sub,...
        'roi_channels',{roi_channels},'preprocessing_history',E.history,'reference',E.ref);
    save(fullfile(cfg.cache_dir,sprintf('spectra_%03d.mat',i)),'spectra','metadata','-v7');
    fprintf('PSD participant %d/%d; spectra %d\n',i,numel(participants),count);
end
T=cell2table(rows(1:count,:),'VariableNames',{'Participant','GlobalTrialOrder','onset_trim_s','roi',...
 'total_1_45','total_1_40','power_40_45','theta','alpha','beta','first_sample',...
 'last_sample','window','overlap','nfft','srate'});
writetable(T,fullfile(cfg.cache_dir,'paired_powers.csv'),'Encoding','UTF-8');
end
