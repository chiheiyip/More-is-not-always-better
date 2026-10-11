"""Compare declared repeated calculations and seal three-scope evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from paper_analysis.teacher.reproducibility import compare_tables,inference_counts,seal_files,verify_seal,verify_complete_bootstrap_repeat
from paper_analysis.teacher.runtime_lock import validate_analysis
from paper_analysis.teacher.state import StageBlockedError,file_sha256
import pandas as pd

def verify(request_path,out,*,exact_numeric=False,complete=False):
    request=json.loads(request_path.read_text(encoding='utf-8-sig'))
    config=json.loads(Path(request['config']).read_text(encoding='utf-8-sig'))
    environment=validate_analysis(config,require_matlab=True)
    if out.exists():raise StageBlockedError('Verification needs a new output directory')
    required={'eye','eeg','joint'}
    if {x['scope'] for x in request['comparisons']}!=required:
        raise StageBlockedError('Verification must explicitly cover eye, EEG and joint')
    if complete and not exact_numeric:
        raise StageBlockedError('Complete verification requires exact numeric comparison')
    bootstrap=None
    if complete:
        if not request.get('full_bootstrap_repeat'):
            raise StageBlockedError('Complete verification needs full_bootstrap_repeat evidence')
        bootstrap=verify_complete_bootstrap_repeat(request['full_bootstrap_repeat'])
    records=seal_files([request_path,request['config'],*[p for c in request['comparisons'] for p in (c['left'],c['right'])]])
    out.mkdir(parents=True)
    scopes={k:{'status':'passed','comparisons':0,'rows':0,'valid_inference_rows':0,'invalid_inference_rows':0} for k in sorted(required)}
    outputs=[]
    for i,c in enumerate(request['comparisons'],1):
        left,right=pd.read_csv(c['left']),pd.read_csv(c['right'])
        model=c.get('model',True)
        counts=[inference_counts(t) for t in (left,right)] if model else None
        result=compare_tables(left,right,c['keys'],model=model,exact_numeric=exact_numeric)
        path=out/f'{i:03d}_{c["scope"]}_comparison.csv';result.to_csv(path,index=False,encoding='utf-8-sig')
        outputs.append(path)
        scope=scopes[c['scope']];scope['comparisons']+=1
        scope['rows']+=len(left)
        if counts:
            if counts[0]!=counts[1]:scope['status']='failed'
            if any(t['invalid_inference_rows'] for t in counts):scope['status']='failed'
            for field in ('valid_inference_rows','invalid_inference_rows'):
                scope[field]+=counts[0][field]
        if not result.passed.all():scope['status']='failed'
    for scope in scopes.values():
        if not scope['valid_inference_rows']:scope['status']='failed'
    verify_seal(records)
    value={'status':'passed' if all(x['status']=='passed' for x in scopes.values()) else 'failed',
           'scopes':scopes,'environment':environment,'inputs':records,'evidence':seal_files(outputs),
           'verification_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           'comparison_mode':'exact_numeric' if exact_numeric else 'tolerance',
           'verification_level':'complete_declared_scope' if complete else 'declared_comparisons',
           'full_bootstrap_repeat':bootstrap,
           'scope_notes':request.get('scope_notes',[])}
    (out/'reproducibility_verification.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    if bootstrap:verify_seal(bootstrap['evidence'])
    if value['status']!='passed':raise StageBlockedError('Reproducibility differences; retain evidence and stop publication')
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request-config',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--exact-numeric',action='store_true',help='Reject every nonzero numeric difference')
    p.add_argument('--complete',action='store_true',help='Require exact comparisons and full four-window bootstrap repeat evidence')
    args=p.parse_args()
    print(json.dumps(verify(args.request_config,args.outdir,exact_numeric=args.exact_numeric,complete=args.complete),ensure_ascii=False))
