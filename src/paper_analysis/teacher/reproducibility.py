"""Numerical comparison and sealed evidence shared by all three scopes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .state import StageBlockedError, file_sha256

def compare_tables(left, right, keys, *, model=True):
    if left.empty or right.empty:
        raise StageBlockedError('Empty/failed result tables cannot establish reproducibility')
    if set(left.columns)!=set(right.columns):raise StageBlockedError('Comparison fields differ')
    for table in (left,right):
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise StageBlockedError('Missing/duplicate reproducibility comparison keys')
    paired=left.merge(right,on=keys,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
    if not paired._merge.eq('both').all():raise StageBlockedError('Recalculated sample/test identities differ')
    rows=[]
    for field in left.columns:
        if field in keys:continue
        a=paired[field+'_a'];b=paired[field+'_b']
        missing=a.isna()!=b.isna()
        numeric=pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        sig=direction=0
        if numeric:
            pfield=field in {'p','q','p.value','p_value','q_value','within_q','joint_q','p_holm','q_BH','p_CR2','q_CR2'} or field.startswith(('p_','q_'))
            tol=1e-6 if pfield else 1e-8
            av=a.to_numpy(dtype=float,na_value=np.nan);bv=b.to_numpy(dtype=float,na_value=np.nan)
            same=np.isclose(av,bv,atol=tol if model else 1e-12,rtol=0 if model else 1e-10,equal_nan=True)
            valid=np.isfinite(av)&np.isfinite(bv)
            if pfield:sig=int(((av[valid]<.05)!=(bv[valid]<.05)).sum())
            if field in {'estimate','beta','Estimate'}:direction=int((np.sign(av[valid])!=np.sign(bv[valid])).sum())
            difference=np.abs(av[valid]-bv[valid]);max_delta=float(difference.max()) if difference.size else 0.
        else:
            same=(a.eq(b)|(a.isna()&b.isna())).to_numpy();max_delta=None
        rows.append({'field':field,'rows':len(paired),'missing_differences':int(missing.sum()),
                     'outside_tolerance':int((~same).sum()),'p_q_crossings':sig,'direction_changes':direction,
                     'max_absolute_difference':max_delta,'passed':bool(same.all() and not sig and not direction)})
    return pd.DataFrame(rows)

def seal_files(paths):
    return [{'path':str(Path(p).resolve()),'sha256':file_sha256(p)} for p in paths]

def verify_seal(records):
    for record in records:
        if file_sha256(record['path'])!=record['sha256']:
            raise StageBlockedError('Reproducibility evidence/input changed: '+record['path'])

def reference_inputs(scene_path, eeg_path, config_path):
    paths={Path(p).resolve() for p in (scene_path,eeg_path,config_path)}
    for table_path in (scene_path,eeg_path):
        table=pd.read_csv(table_path,encoding='utf-8-sig')
        for field in ('eye_csv_path','eeg_sample_csv_path','aoi_json_path'):
            if field not in table:continue
            for value in table[field].dropna().unique():
                p=Path(str(value))
                if not p.is_absolute():p=Path(table_path).parent/p
                if p.is_file():paths.add(p.resolve())
                # Missing file is a documented synchronization QC exclusion.
    return seal_files(sorted(paths))
