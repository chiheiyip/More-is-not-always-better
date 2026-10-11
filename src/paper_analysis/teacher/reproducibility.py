"""Numerical comparison and sealed evidence shared by all three scopes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .state import StageBlockedError, file_sha256

def inference_counts(table):
    """Count usable inference without deleting failed comparison rows."""
    probabilities = [c for c in table if c in {'p', 'p.value', 'p_value'}
                     or c.startswith(('p.value.', 'p_'))]
    statistics = [c for c in ('estimate', 'beta', 'Estimate', 'Fstat') if c in table]
    if statistics:
        values = table[statistics].apply(pd.to_numeric, errors='coerce')
        p = table[probabilities].apply(pd.to_numeric, errors='coerce')
        usable = (np.isfinite(p) & p.ge(0) & p.le(1)).any(axis=1)
        for low,high in [('conf.low','conf.high'),('ci_low','ci_high'),('CI_low','CI_high')]:
            if low in table and high in table:
                bounds=table[[low,high]].apply(pd.to_numeric,errors='coerce')
                usable |= np.isfinite(bounds).all(axis=1) & bounds[low].le(bounds[high])
        valid = np.isfinite(values).any(axis=1) & usable & ~(np.isfinite(p) & (p.lt(0)|p.gt(1))).any(axis=1)
    elif all(c in table for c in ('bootstrap_median', 'ci_low', 'ci_high', 'successful')):
        values = table[['bootstrap_median', 'ci_low', 'ci_high', 'successful']].apply(pd.to_numeric, errors='coerce')
        valid = np.isfinite(values).all(axis=1) & values.successful.gt(0) & values.ci_low.le(values.ci_high)
    else:
        raise StageBlockedError('Model comparison lacks inference fields; declare diagnostics/input tables as model=false')
    if 'inference_valid' in table:
        valid &= table.inference_valid.astype(str).str.lower().isin(['true', '1'])
    if 'status' in table:
        valid &= table.status.eq('fit')
    if not valid.any():
        raise StageBlockedError('Failed/missing model inference cannot establish a reproducibility pass')
    return {'valid_inference_rows': int(valid.sum()), 'invalid_inference_rows': int((~valid).sum())}


def compare_tables(left, right, keys, *, model=True, exact_numeric=False):
    if left.empty or right.empty:
        raise StageBlockedError('Empty/failed result tables cannot establish reproducibility')
    if set(left.columns)!=set(right.columns):raise StageBlockedError('Comparison fields differ')
    for table in (left,right):
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise StageBlockedError('Missing/duplicate reproducibility comparison keys')
    paired=left.merge(right,on=keys,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
    if not paired._merge.eq('both').all():raise StageBlockedError('Recalculated sample/test identities differ')
    rows=[{'field':'__keys__','rows':len(paired),'missing_differences':0,
           'outside_tolerance':0,'p_q_crossings':0,'direction_changes':0,
           'max_absolute_difference':0.,'passed':True}]
    for field in left.columns:
        if field in keys:continue
        a=paired[field+'_a'];b=paired[field+'_b']
        missing=a.isna()!=b.isna()
        numeric=pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        sig=direction=0
        if numeric:
            pfield=field in {'p','q','p.value','p_value','q_value','within_q','joint_q','p_holm','q_BH','p_CR2','q_CR2'} or field.startswith(('p_','q_','p.value.','q.value.'))
            tol=1e-6 if pfield else 1e-8
            av=a.to_numpy(dtype=float,na_value=np.nan);bv=b.to_numpy(dtype=float,na_value=np.nan)
            if exact_numeric:
                same=(a.eq(b)|(a.isna()&b.isna())).to_numpy(dtype=bool,na_value=False)
            else:
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


def verify_complete_bootstrap_repeat(proof_path):
    """Require actual repeated fits for all four windows, not a draw replay."""
    proof_path=Path(proof_path)
    proof_hash=file_sha256(proof_path)
    proof=json.loads(proof_path.read_text(encoding='utf-8-sig'))
    jobs=proof.get('jobs',[])
    if (proof.get('status')!='passed' or len(jobs)!=4
            or {j.get('trim_s') for j in jobs}!={0,5,10,15}
            or proof.get('fits_requested')!=80000):
        raise StageBlockedError('Complete verification needs four fresh 20000-fit bootstrap repeats')
    verify_seal(proof.get('scripts',[]))
    if not proof.get('scripts') or not proof.get('code_sha') or not proof.get('environment'):
        raise StageBlockedError('Bootstrap repeat lacks code/environment evidence')
    evidence=[proof_path]
    for job in jobs:
        if (job.get('status')!='passed' or job.get('fits_requested')!=20000
                or not job.get('all_successful_5000')
                or job.get('max_absolute_difference')!=0):
            raise StageBlockedError('Bootstrap repeat is incomplete or numerically different')
        if not job.get('input_seal') or not job.get('outputs'):
            raise StageBlockedError('Bootstrap repeat lacks input/output hashes')
        verify_seal(job['input_seal']);verify_seal(job['outputs'])
        reference=Path(job['reference']).resolve();repeat=Path(job['repeat']).resolve()
        if reference==repeat:
            raise StageBlockedError('Bootstrap repeat must use a separate output directory')
        for name in ['07_eeg_cluster_bootstrap_5000.csv','08_eeg_bootstrap_failures.csv','bootstrap_draw_provenance.json']:
            left=reference/name;right=repeat/name
            if file_sha256(left)!=file_sha256(right):
                raise StageBlockedError('Bootstrap outputs/draws are not byte identical: '+name)
            evidence.extend([left,right])
        table=pd.read_csv(repeat/'07_eeg_cluster_bootstrap_5000.csv')
        if (len(table)!=60 or table[['outcome','term']].duplicated().any()
                or table.outcome.nunique()!=4 or not table.successful.eq(5000).all()
                or not table.failed_replicates.eq(0).all()):
            raise StageBlockedError('Bootstrap repeat lacks all four outcomes and 5000 successful fits')
        counts=inference_counts(table)
        if counts['invalid_inference_rows']:
            raise StageBlockedError('Bootstrap repeat contains invalid inference')
    if file_sha256(proof_path)!=proof_hash:
        raise StageBlockedError('Bootstrap repeat proof changed during verification')
    return {'status':'passed','windows':4,'fresh_repeated_fits':80000,
            'code_sha':proof['code_sha'],'evidence':seal_files(evidence)}

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
