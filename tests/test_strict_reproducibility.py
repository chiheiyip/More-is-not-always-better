import json
from pathlib import Path

import pandas as pd
import pytest

from paper_analysis.teacher.reproducibility import (
    compare_tables, seal_files, verify_complete_bootstrap_repeat,
)
from paper_analysis.teacher.state import StageBlockedError


def test_exact_mode_rejects_a_nonzero_difference_accepted_by_tolerance():
    left=pd.DataFrame({'key':[1],'estimate':[.2],'p.value':[.3]})
    right=left.copy();right.loc[0,'estimate']+=1e-12
    assert compare_tables(left,right,['key']).passed.all()
    assert not compare_tables(left,right,['key'],exact_numeric=True).passed.all()


def test_exact_mode_pairs_shuffled_rows_and_retains_missing_and_infinite_values():
    left=pd.DataFrame({'key':[1,2,3],'estimate':[.2,float('nan'),-.1],
                       'p.value':[.3,float('nan'),.02],'df':[float('inf'),12.,13.]})
    assert compare_tables(left,left.iloc[::-1],['key'],exact_numeric=True).passed.all()
    right=left.copy();right.loc[1,'estimate']=0.
    assert not compare_tables(left,right,['key'],exact_numeric=True).passed.all()


def test_exact_mode_does_not_round_large_integer_measurements():
    left=pd.DataFrame({'key':[1],'sample_clock':[2**53+1]})
    right=pd.DataFrame({'key':[1],'sample_clock':[2**53+2]})
    assert not compare_tables(left,right,['key'],model=False,exact_numeric=True).passed.all()


@pytest.fixture
def complete_repeat(tmp_path):
    script=tmp_path/'model.R';script.write_text('registered model implementation')
    jobs=[]
    for trim in [0,5,10,15]:
        reference=tmp_path/f'reference_{trim}';repeat=tmp_path/f'repeat_{trim}'
        reference.mkdir();repeat.mkdir()
        table=pd.DataFrame([{'outcome':f'O{k}','term':f'T{j}','bootstrap_median':.2,
                             'ci_low':.1,'ci_high':.3,'successful':5000,'failed_replicates':0}
                            for k in range(4) for j in range(15)])
        for folder in [reference,repeat]:
            table.to_csv(folder/'07_eeg_cluster_bootstrap_5000.csv',index=False)
            (folder/'08_eeg_bootstrap_failures.csv').write_text('failed\n0\n')
            (folder/'bootstrap_draw_provenance.json').write_text('{"seed":20260726}')
        jobs.append({'status':'passed','trim_s':trim,'fits_requested':20000,
                     'all_successful_5000':True,'max_absolute_difference':0.,
                     'reference':str(reference),'repeat':str(repeat),
                     'input_seal':seal_files(list(reference.iterdir())),
                     'outputs':seal_files(list(repeat.iterdir()))})
    proof={'status':'passed','code_sha':'committed_model_sha','environment':{'R':'4.4.2'},
           'scripts':seal_files([script]),'fits_requested':80000,'jobs':jobs}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    return path,proof


def test_complete_bootstrap_repeat_checks_four_windows_and_real_output_hashes(complete_repeat):
    path,proof=complete_repeat
    result=verify_complete_bootstrap_repeat(path)
    assert result['fresh_repeated_fits']==80000
    modified=Path(proof['jobs'][0]['repeat'])/'07_eeg_cluster_bootstrap_5000.csv'
    modified.write_text(modified.read_text()+'changed')
    with pytest.raises(StageBlockedError,match='changed'):
        verify_complete_bootstrap_repeat(path)


@pytest.mark.parametrize('kind',['draw_replay','missing_window','duplicate_window','incomplete_fits','same_directory'])
def test_complete_status_rejects_replay_or_incomplete_fitting(complete_repeat,kind):
    path,proof=complete_repeat
    if kind=='draw_replay':proof={'status':'passed','jobs':[{'requested_draws':80000}]}
    elif kind=='missing_window':proof['jobs'].pop()
    elif kind=='duplicate_window':proof['jobs'][1]['trim_s']=0
    elif kind=='incomplete_fits':proof['jobs'][0]['fits_requested']=19999
    else:proof['jobs'][0]['repeat']=proof['jobs'][0]['reference']
    path.write_text(json.dumps(proof))
    with pytest.raises(StageBlockedError):verify_complete_bootstrap_repeat(path)


@pytest.fixture
def verifier_module(monkeypatch):
    import importlib.util
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('strict_verifier',repo/'scripts/verify_analysis_reproducibility.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'validate_analysis',lambda *a,**k:{'test_environment':'registered'})
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'test_sha\n')
    return module


def make_request(tmp_path,invalid=False):
    table=pd.DataFrame({'key':[1,2],'estimate':[.2,.3],'p.value':[.03,.04]})
    if invalid:table.loc[1,['estimate','p.value']]=float('nan')
    left=tmp_path/'left.csv';right=tmp_path/'right.csv'
    table.to_csv(left,index=False);table.to_csv(right,index=False)
    config=tmp_path/'config.json';config.write_text('{}')
    request=tmp_path/'request.json'
    request.write_text(json.dumps({'config':str(config),'comparisons':[
        {'scope':scope,'left':str(left),'right':str(right),'keys':['key']}
        for scope in ['eye','eeg','joint']]}))
    return request


def test_verifier_cannot_count_matching_invalid_inference_as_a_pass(tmp_path,verifier_module):
    request=make_request(tmp_path,invalid=True);out=tmp_path/'verification'
    with pytest.raises(StageBlockedError,match='differences'):
        verifier_module.verify(request,out,exact_numeric=True)
    value=json.loads((out/'reproducibility_verification.json').read_text())
    assert value['status']=='failed'
    assert all(s['invalid_inference_rows']==1 for s in value['scopes'].values())


def test_complete_verification_requires_both_exact_mode_and_bootstrap_proof(tmp_path,verifier_module):
    request=make_request(tmp_path)
    with pytest.raises(StageBlockedError,match='exact numeric'):
        verifier_module.verify(request,tmp_path/'verification',complete=True)
    with pytest.raises(StageBlockedError,match='full_bootstrap_repeat'):
        verifier_module.verify(request,tmp_path/'verification',complete=True,exact_numeric=True)


def test_exact_comparisons_do_not_claim_complete_verification_without_repeat_proof(tmp_path,verifier_module):
    request=make_request(tmp_path)
    value=verifier_module.verify(request,tmp_path/'verification',exact_numeric=True)
    assert value['status']=='passed' and value['comparison_mode']=='exact_numeric'
    assert value['verification_level']=='declared_comparisons'
    assert value['full_bootstrap_repeat'] is None
