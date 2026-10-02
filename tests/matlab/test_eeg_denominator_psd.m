function tests=test_eeg_denominator_psd
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'matlab','eeg_bandpower_pipeline'));
end
function testOriginalWelchAndBothDenominators(testCase)
fs=500;t=(0:fs*25)/fs;
x=sin(2*pi*6*t)+2*sin(2*pi*10*t)+.5*sin(2*pi*42*t);
s=eeg_denominator_spectrum([x;x],{'F3','F4'},{'F3','F4'},fs,.002,25,5);
first=1+5*fs;last=25*fs;
[p,f]=pwelch(x(first:last)',1000,500,1024,fs);
verifyEqual(testCase,s.first_sample,first);verifyEqual(testCase,s.last_sample,last);
verifyEqual(testCase,s.f,f);verifyEqual(testCase,s.pxx,p,'AbsTol',1e-14);
verifyEqual(testCase,s.powers(1),trapz(f(f>=1 & f<=45),p(f>=1 & f<=45)),'AbsTol',1e-14);
verifyEqual(testCase,s.powers(2),trapz(f(f>=1 & f<=40),p(f>=1 & f<=40)),'AbsTol',1e-14);
verifyGreaterThan(testCase,s.powers(1)-s.powers(2),s.powers(3));
verifyEqual(testCase,max(f(f<=40)),39.55078125);
verifyEqual(testCase,min(f(f>=40)),40.0390625);
end
function testRoiAverageBeforePsdAndNonfinite(testCase)
fs=100;t=(0:fs*5)/fs;x=sin(2*pi*6*t); y=2*sin(2*pi*10*t);
wave=[x;y];wave(1,5)=NaN;wave(:,8)=Inf;
s=eeg_denominator_spectrum(wave,{'O1','O2'},{'O1','OZ','O2'},fs,.01,5,0);
signal=mean(wave(:,1:500),1,'omitnan');signal=signal(isfinite(signal));
[p,f]=pwelch(signal',200,100,256,fs);
verifyEqual(testCase,s.pxx,p,'AbsTol',1e-14);verifyEqual(testCase,s.f,f);
verifyEqual(testCase,s.finite_samples,499);
end
