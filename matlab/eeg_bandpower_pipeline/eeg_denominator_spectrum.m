function result = eeg_denominator_spectrum(waveform, labels, roi, fs, first, last, trim)
% Same time-domain averaging, inclusive endpoints and numeric-window pwelch
% as run_eeg_bandpower_pipeline; no filtering, rereferencing or ICA here.
idx = find(ismember(upper(string(labels)), upper(string(roi))));
assert(~isempty(idx), 'No channels present for requested ROI');
first = max(1,round(first*fs)) + round(trim*fs);
last = min(size(waveform,2),round(last*fs));
assert(first<=last,'Invalid trimmed epoch');
signal = mean(waveform(idx,first:last),1,'omitnan');
signal = signal(isfinite(signal));
assert(numel(signal)>=max(8,round(fs)),'Insufficient finite samples');
win = min(numel(signal),max(round(2*fs),8));
nfft = max(2^nextpow2(win),win);
[pxx,f] = pwelch(signal(:),win,floor(win/2),nfft,fs);
bands = [1 45;1 40;40 45;4 7;8 12;13 30];
values = zeros(1,6);
for k=1:6
    mask=f>=bands(k,1) & f<=bands(k,2);
    assert(nnz(mask)>=2,'Frequency range lacks two bins');
    values(k)=trapz(f(mask),pxx(mask));
end
assert(all(isfinite(values)) && all(values([1 2 4 5 6])>0), 'Invalid power');
result = struct('f',f,'pxx',pxx,'powers',values,'first_sample',first,...
    'last_sample',last,'finite_samples',numel(signal),'window',win,...
    'overlap',floor(win/2),'nfft',nfft,'channels',{cellstr(string(labels(idx)))});
end
