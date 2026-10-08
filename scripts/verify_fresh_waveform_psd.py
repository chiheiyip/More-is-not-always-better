"""Independent SciPy Welch checks against a completed current-source PSD cache."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from paper_analysis.teacher.fresh import set_record, git_sha, dump
from paper_analysis.teacher.state import file_sha256


def verify(run, sample_count=12):
    config=json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
    provenance=run/'10_eeg_denominator_sensitivity/PSD_provenance.json'
    cache=json.loads(provenance.read_text(encoding='utf-8'))
    root=Path(cache['cache_dir']); powers=pd.read_csv(root/'paired_powers.csv')
    anchors={Path(k).stem for k in config['fresh'].get('locked_source_hashes',{})}
    sources=list(enumerate(cache['sources'],1))
    selected=([entry for entry in sources if entry[1]['participant'] in anchors] +
              [entry for entry in sources if entry[1]['participant'] not in anchors])[:sample_count]
    rows=[];frequency=[]
    for index,source in sources:
        mat=loadmat(root/f'spectra_{index:03d}.mat',squeeze_me=True,struct_as_record=False)
        assert mat['metadata'].source.participant==source['participant']
        spectra=np.asarray(mat['spectra'],dtype=object)
        for col,roi in enumerate(['F','P','O']):
            sub=powers.loc[powers.Participant.eq(source['participant'])&powers.roi.eq(roi)]
            assert len(sub)==spectra.shape[0]
            for j,(_,record) in enumerate(sub.iterrows()):
                s=spectra[j,col];assert s.first_sample==record.first_sample and s.last_sample==record.last_sample
                f=np.asarray(s.f).ravel()
                bands=[f[(f>=low)&(f<=high)] for low,high in [(1,45),(1,40),(40,45)]]
                frequency.append({**{k:record[k] for k in ['Participant','GlobalTrialOrder','onset_trim_s','roi']},
                    'frequency_step':float(f[1]-f[0]),'nominal_1_45_first':float(bands[0][0]),
                    'nominal_1_45_last':float(bands[0][-1]),'nominal_1_40_last':float(bands[1][-1]),
                    'nominal_40_45_first':float(bands[2][0])})
    for counter,(index,source) in enumerate(selected):
        inventory=set_record(source['set_path'],config['fresh'].get('fdt_aliases'))
        assert inventory['fdt_path']==source['fdt_path']
        for kind in ['set','fdt']:
            assert file_sha256(source[kind+'_path'])==source[kind+'_sha256']
        mat=loadmat(root/f'spectra_{index:03d}.mat',squeeze_me=True,struct_as_record=False)
        assert mat['metadata'].source.participant==source['participant']
        spectra=np.asarray(mat['spectra'],dtype=object)
        roi=['F','P','O'][counter%3]; col=['F','P','O'].index(roi)
        sub=powers.loc[powers.Participant.eq(source['participant'])&powers.roi.eq(roi)]
        assert len(sub)==spectra.shape[0]
        row=sub.loc[sub.onset_trim_s.eq([0,5,10,15][counter%4])].iloc[0]
        s=spectra[list(sub.index).index(row.name),col]
        assert s.first_sample==row.first_sample and s.last_sample==row.last_sample
        labels=[value.upper() for value in inventory['channels']]
        channels={'F':['F3','F4'],'P':['P3','PZ','P4'],'O':['O1','OZ','O2']}[roi]
        idx=[labels.index(value) for value in channels]
        raw=np.memmap(source['fdt_path'],dtype='<f4',mode='r').reshape((inventory['nbchan'],inventory['pnts']),order='F')
        signal=np.nanmean(np.asarray(raw[idx,int(row.first_sample)-1:int(row.last_sample)],dtype=np.float64),axis=0)
        signal=signal[np.isfinite(signal)]
        # MATLAB's numeric window invokes hamming(length), with symmetric
        # endpoints; pwelch does not detrend each segment.
        f,pxx=welch(signal,fs=inventory['srate'],window=np.hamming(int(row.window)),
                    nperseg=int(row.window),noverlap=int(row.overlap),nfft=int(row.nfft),
                    detrend=False,scaling='density')
        sf,sp=np.asarray(s.f).ravel(),np.asarray(s.pxx).ravel()
        passed=np.allclose(f,sf,rtol=1e-10,atol=1e-12) and np.allclose(pxx,sp,rtol=1e-10,atol=1e-12)
        record={k:row[k] for k in ['Participant','GlobalTrialOrder','onset_trim_s','roi']}
        record.update(max_psd_abs_difference=float(np.max(abs(pxx-sp))),Pass=bool(passed))
        assert passed,record
        for band,(low,high) in {'theta':(4,7),'alpha':(8,12),'beta':(13,30)}.items():
            mask=(f>=low)&(f<=high)
            actual=float(np.trapz(pxx[mask],f[mask]))
            assert np.isclose(actual,row[band],rtol=1e-10,atol=1e-12)
            record[band+'_abs_difference']=abs(actual-row[band])
        rows.append(record)
    out=run/'12_final_verification';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'waveform_Welch_spot_checks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(frequency).to_csv(out/'frequency_mask_endpoints.csv',index=False,encoding='utf-8-sig')
    dump(out/'waveform_Welch_verification.json',{'status':'passed','spectra':len(rows),'band_checks':3*len(rows),
         'analysis_sha':json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))['git_commit'],
         'verification_sha':git_sha(REPO),'script_sha256':file_sha256(__file__),'scipy':scipy.__version__,
         'rtol':1e-10,'atol':1e-12,'all_frequency_grids':len(frequency),'source_provenance_sha256':file_sha256(provenance)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=12)
    args=parser.parse_args();verify(args.run_root,args.samples)
