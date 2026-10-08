from pathlib import Path

import numpy as np
import pandas as pd

from paper_analysis.fusion.clock_sync import run_clock_synchronized_fusion


def test_participant_partition_preserves_clock_qc_and_all_parallel_bins(tmp_path):
    scene=[];samples=[];fs=500;n=18*fs
    start=pd.Timestamp('2026-10-08 12:00:00',tz='Asia/Shanghai').value//1_000_000
    t=np.arange(n)/fs
    for person in ['P01','P02','P03','P04']:
        epath=tmp_path/f'{person}_eeg.csv';eye=tmp_path/f'{person}_eye.csv'
        eeg=pd.DataFrame({'sample_index_recording':np.arange(n)+1,'sample_index_scene':np.arange(n),
             'eeg_epoch_ms':start+np.arange(n)*2,'eeg_scene_time_ms':np.arange(n)*2})
        for channel in ['F3','F4','P3','Pz','P4','O1','Oz','O2']:
            eeg['preproc_'+channel+'_uV']=np.sin(2*np.pi*6*t)+np.sin(2*np.pi*10*t)+np.sin(2*np.pi*20*t)
        eeg.to_csv(epath,index=False)
        ms=np.arange(0,18000,4)
        pd.DataFrame({'Time of Day[HH:mm:ss.ms]':[f'12:00:{value//1000:02d}.{value%1000:03d}' for value in ms],
              'Recording Time Stamp[ms]':ms,'Gaze Point X[px]':100.,'Gaze Point Y[px]':100.,
              'Fixation Index':ms//200}).to_csv(eye,index=False)
        scene.append({'participant_id':person,'scene_id':1,'eye_csv_path':str(eye),'eye_record_id':'261008_sample','aoi_json_path':'','WWR':15,'Complexity':0})
        samples.append({'participant_id':person,'scene_id':1,'eeg_sample_csv_path':str(epath)})
    manifest=tmp_path/'scene.csv';sample_manifest=tmp_path/'samples.csv'
    pd.DataFrame(scene).to_csv(manifest,index=False);pd.DataFrame(samples).to_csv(sample_manifest,index=False)
    options=dict(export_pointwise=False,onset_trim_s=10,onset_trim_variants_s=[0,5,10,15])
    all_outputs=run_clock_synchronized_fusion(manifest,sample_manifest,tmp_path/'serial',**options)
    parts=[]
    for index,row in enumerate(scene):
        subset=tmp_path/f'part_{index}.csv';pd.DataFrame([row]).to_csv(subset,index=False)
        parts.append(run_clock_synchronized_fusion(subset,sample_manifest,tmp_path/f'out_{index}',**options))
    for field in ['clock_alignment_scene_qc','clock_alignment_participant_qc','aligned_synchronized_timebin']:
        full=pd.read_csv(all_outputs[field]);combined=pd.concat([pd.read_csv(part[field]) for part in parts],ignore_index=True)
        keys=[key for key in ['participant_id','scene_id','onset_trim_s','bin_index','class_name'] if key in full]
        full=full.sort_values(keys).reset_index(drop=True);combined=combined.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(full,combined,check_exact=True)
