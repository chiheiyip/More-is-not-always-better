import json
from pathlib import Path
import pandas as pd
import pytest
from paper_analysis.teacher import native_runtime
from paper_analysis.teacher.state import StageBlockedError
from paper_analysis.fusion.clock_sync import run_clock_synchronized_fusion

def test_native_thread_or_library_change_is_rejected(tmp_path,monkeypatch):
    (tmp_path/'configs').mkdir()
    expected={'libraries':{'blas.dll':{'sha256':'a','threads':16}},'thread_environment':{}}
    (tmp_path/'configs/analysis_native_python.lock.json').write_text(json.dumps({'snapshot':expected}))
    monkeypatch.setattr(native_runtime,'python_native_snapshot',lambda:expected)
    assert native_runtime.validate_python_native(tmp_path)['status']=='passed'
    for changed in [{'libraries':{'blas.dll':{'sha256':'b','threads':16}},'thread_environment':{}},
                    {'libraries':{'blas.dll':{'sha256':'a','threads':1}},'thread_environment':{}}]:
        monkeypatch.setattr(native_runtime,'python_native_snapshot',lambda:changed)
        with pytest.raises(StageBlockedError,match='DLL/thread'):native_runtime.validate_python_native(tmp_path)

def test_r_startup_locale_is_fixed_before_unicode_library_loading(monkeypatch):
    from paper_analysis.teacher.runtime_lock import locked_r_environment
    monkeypatch.setenv('LANG','C.UTF-8');monkeypatch.setenv('LC_ALL','C.UTF-8')
    monkeypatch.setattr('paper_analysis.teacher.runtime_lock.platform.system',lambda:'Windows')
    env=locked_r_environment()
    assert env['LANG']==env['LC_ALL']=='Chinese (Simplified)_China.utf8'

def test_python_install_and_runtime_locks_agree():
    repo=Path(__file__).resolve().parents[1]
    canonical=lambda n:n.lower().replace('_','-')
    requirements={canonical(line.split('==')[0]):line.split('==')[1] for line in
        (repo/'requirements-analysis.lock.txt').read_text().splitlines() if '==' in line}
    runtime={canonical(k):v for k,v in json.loads((repo/'configs/analysis_python.lock.json').read_text())['packages'].items()}
    assert requirements==runtime

def test_sync_rejects_duplicate_or_missing_keys_before_output(tmp_path):
    scene=tmp_path/'eye.csv';eeg=tmp_path/'eeg.csv';out=tmp_path/'out'
    pd.DataFrame([{'participant_id':'P1','scene_id':1,'eeg_sample_csv_path':'missing.csv'}]).to_csv(eeg,index=False)
    for records in [[{'participant_id':'P1','scene_id':1,'eye_csv_path':'missing.csv'}]*2,
                    [{'participant_id':None,'scene_id':1,'eye_csv_path':'missing.csv'}]]:
        pd.DataFrame(records).to_csv(scene,index=False)
        with pytest.raises(ValueError,match='Missing or duplicate'):run_clock_synchronized_fusion(scene,eeg,out)
        assert not out.exists()

def test_archive_tampering_and_path_escape_are_rejected(tmp_path):
    import importlib.util,zipfile
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('archive',repo/'scripts/archive_analysis_environment.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as archive:archive.writestr('../escape','x')
    with pytest.raises(ValueError,match='Unsafe'):m.unpack(z,tmp_path/'restore')
    (tmp_path/'archive_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest SHA256'):m.restore(tmp_path,tmp_path/'restore','wrong')

def test_comparison_preserves_family_keys_and_investigates_small_crossing():
    from paper_analysis.teacher.reproducibility import compare_tables
    a=pd.DataFrame({'key':[1,2],'estimate':[1.,-1.],'p.value':[.0499999,.3],'df':[float('inf'),12.]})
    assert compare_tables(a,a,['key']).passed.all()
    b=a.copy();b.loc[0,'p.value']=.0500001
    result=compare_tables(a,b,['key'])
    assert result.set_index('field').loc['p.value','outside_tolerance']==0
    assert not result.passed.all()
    with pytest.raises(StageBlockedError,match='identities'):compare_tables(a,b.iloc[:1],['key'])
    with pytest.raises(StageBlockedError,match='Empty/failed'):compare_tables(a.iloc[:0],b.iloc[:0],['key'])

@pytest.mark.parametrize('field',['p.value.likelihood','p.value.CR2','p.value.BH'])
def test_eye_probability_variants_use_probability_tolerance_and_boundary_check(field):
    from paper_analysis.teacher.reproducibility import compare_tables
    a=pd.DataFrame({'key':[1],field:[.0499999]})
    b=pd.DataFrame({'key':[1],field:[.0500001]})
    result=compare_tables(a,b,['key']).iloc[0]
    assert result.outside_tolerance==0
    assert result.p_q_crossings==1
    assert not result.passed
