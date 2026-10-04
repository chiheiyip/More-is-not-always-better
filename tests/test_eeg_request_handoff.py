from pathlib import Path
import importlib.util
import os
import json

import numpy as np
import pandas as pd
import pytest

from paper_analysis.teacher.request_handoff import (compare_cells, bh_checks,
    unique_keys, archive_publish, r_environment, independent_design_column)


def test_independent_cell_reader_detects_columns_values_and_missingness():
    original=pd.DataFrame({'Participant':['甲','乙'],'GlobalTrialOrder':['1','2'],'x':['','0.3']})
    assert compare_cells(original,original.iloc[::-1],keys=['Participant','GlobalTrialOrder'])==6
    with pytest.raises(ValueError,match='Column'):compare_cells(original,original[['x','Participant','GlobalTrialOrder']])
    changed=original.copy();changed.loc[0,'x']='NA'
    with pytest.raises(ValueError,match='Cell'):compare_cells(original,changed)


def test_duplicate_and_blank_identity_fail():
    with pytest.raises(ValueError):unique_keys(pd.DataFrame({'Participant':['甲','甲']}),['Participant'])
    with pytest.raises(ValueError):unique_keys(pd.DataFrame({'Participant':['']}),['Participant'])


def test_bh_invalid_family_not_shrunk():
    frame=pd.DataFrame({'onset_trim_s':[0,0], 'outcome':['x','y'],'model':['Model1']*2,'term':['x']*2,
        'family_id':['f']*2,'p.value':[.01,np.nan],'inference_valid':[True,False],
        'within_q':[np.nan]*2,'joint_q':[np.nan]*2})
    assert not bh_checks(frame)[0]['valid']
    frame.loc[0,'joint_q']=.01
    with pytest.raises(ValueError,match='BH'):bh_checks(frame)


def test_archive_publication_keeps_exact_bytes_and_no_history_in_current(tmp_path):
    stage=tmp_path/'stage';target=tmp_path/'delivery';archive=tmp_path/'history'/'run'
    stage.mkdir();target.mkdir();(target/'old.txt').write_bytes(b'original');(stage/'new.md').write_bytes(b'new')
    mapping=archive_publish(stage,target,archive)
    assert {p.name for p in target.iterdir()}=={'new.md'}
    assert (archive/'old.txt').read_bytes()==b'original'
    assert mapping[str(target/'old.txt')]==str(archive/'old.txt')


def test_archive_rejects_nested_target_and_rolls_back_move_failure(tmp_path,monkeypatch):
    stage=tmp_path/'stage';target=tmp_path/'delivery';stage.mkdir();target.mkdir()
    (target/'old.txt').write_bytes(b'original');(stage/'new.md').write_bytes(b'new')
    with pytest.raises(ValueError):archive_publish(stage,target,target/'history')
    import paper_analysis.teacher.request_handoff as module
    real=module.shutil.move
    def fail(src,dst):
        if Path(src).parent==stage:raise OSError('publish failed')
        return real(src,dst)
    monkeypatch.setattr(module.shutil,'move',fail)
    with pytest.raises(OSError):archive_publish(stage,target,tmp_path/'history')
    assert (target/'old.txt').read_bytes()==b'original'
    assert not (target/'new.md').exists()


def test_system_r_environment_keeps_isolated_and_original_user_library(monkeypatch):
    monkeypatch.setenv('R_HOME','wrong');monkeypatch.setenv('LC_CTYPE','C.UTF-8')
    env=r_environment({'r_library':'D:/isolated'})
    assert 'R_HOME' not in env and 'LC_CTYPE' not in env
    assert env['R_LIBS_USER'].startswith('D:/isolated;')


def test_archive_rolls_back_failed_postmove_validation(tmp_path):
    stage=tmp_path/'stage';target=tmp_path/'delivery';stage.mkdir();target.mkdir()
    (target/'old.txt').write_bytes(b'original');(stage/'new.md').write_bytes(b'new')
    def reject(current,archive):raise ValueError('hash mismatch')
    with pytest.raises(ValueError,match='hash mismatch'):
        archive_publish(stage,target,tmp_path/'history',reject)
    assert (target/'old.txt').read_bytes()==b'original'
    assert (stage/'new.md').read_bytes()==b'new'
    assert not (target/'new.md').exists()


def test_python_design_checks_reference_levels_and_interactions():
    frame=pd.DataFrame({'WWR':['WWR15','WWR45'],'Complexity':['C0','C1'],
        'ExperienceGroup':['High','Low'],'PositionWithinBlockCentered':[-2.5,.5]})
    assert independent_design_column(frame,'WWRWWR45:ComplexityC1').tolist()==[0,1]
    assert independent_design_column(frame,'ExperienceGroupLow').tolist()==[0,1]
    assert independent_design_column(frame,'PositionWithinBlockCentered').tolist()==[-2.5,.5]
    with pytest.raises(ValueError):independent_design_column(frame,'Unexpected')


def test_real_r_independent_grid_refits(tmp_path):
    root=Path(__file__).resolve().parents[1]
    r=Path('C:/Program Files/R/R-4.5.3/bin/x64/Rscript.exe')
    library=root/'.codex_tmp/r45-lib'
    if not r.is_file() or not library.exists():pytest.skip('local Windows R 4.5 verification runtime unavailable')
    from test_eeg_denominator import synthetic_config
    from paper_analysis.teacher.eeg_audit import prepare_inputs, FACTOR_LEVELS
    from paper_analysis.teacher.eeg_denominator import run_models
    from paper_analysis.teacher.request_handoff import r_call
    config=synthetic_config(tmp_path,unicode_ids=True)
    frames,_=prepare_inputs(config)
    contract=tmp_path/'contract.json';contract.write_text(json.dumps({'factor_levels':FACTOR_LEVELS,'seed':20260906}))
    production=tmp_path/'production';production.mkdir()
    reference=run_models(frames,config,root,production/'A',contract)
    import shutil
    shutil.copytree(production/'A',production/'B');shutil.copytree(production/'A',production/'C')
    requests=tmp_path/'requests.csv'
    pd.DataFrame([{'onset_trim_s':0,'outcome':'F_theta_relative','model':'Model1'}]).to_csv(requests,index=False)
    output=tmp_path/'refit';output.mkdir()
    independent={'rscript':str(r),'r_library':str(library)}
    r_call(independent,root,Path('analysis/r/eeg_request_model_verify.R'),
           {'source_run':str(production),'requests':str(requests),'output':str(output),'design':str(contract)},output,'models')
    actual=pd.read_csv(output/'r_refit_factors.csv')
    expected=reference['factor_tests'].loc[reference['factor_tests'].onset_trim_s.eq(0)&reference['factor_tests'].outcome.eq('F_theta_relative')]
    merged=actual.loc[actual.version.eq('A')].merge(expected,on=['onset_trim_s','outcome','model','term'],suffixes=('_new','_old'))
    assert len(merged)==6
    assert np.allclose(merged['p.value_new'],merged['p.value_old'],atol=1e-6,rtol=0)


def test_real_r_independent_csv_excel_mat_readers(tmp_path):
    root=Path(__file__).resolve().parents[1]
    r=Path('C:/Program Files/R/R-4.5.3/bin/x64/Rscript.exe');library=root/'.codex_tmp/r45-lib'
    if not r.is_file() or not library.exists():pytest.skip('local Windows R 4.5 verification runtime unavailable')
    from scipy.io import savemat
    from openpyxl import Workbook
    from paper_analysis.teacher.request_handoff import r_call,raw_csv
    inputs=tmp_path/'original';inputs.mkdir();output=tmp_path/'reader_check';output.mkdir()
    frame=pd.DataFrame({'Participant':['测试甲','测试乙'],'GlobalTrialOrder':[1,2],'blank':['',''],'number':['1e-6','0.23']})
    frame.to_csv(inputs/'original.csv',index=False,encoding='utf-8-sig')
    headers=['Q1.0_姓名：','Q1.4_乒乓球经验：','Q1.5_近 6 个月平均运动频率：']
    workbook=Workbook();sheet=workbook.active;sheet.append(headers);sheet.append(['测试甲','偶尔（每月1–2次）','经常'])
    workbook.save(inputs/'original.xlsx')
    spectra=np.empty((2,3),dtype=object);powers=[]
    for j in range(2):
        for k,roi in enumerate('FPO'):
            spectra[j,k]={'f':np.arange(0,51,dtype=float),'pxx':np.ones(51),'first_sample':j*100+1,'last_sample':j*100+100}
            powers.append({'Participant':'测试甲','GlobalTrialOrder':j+1,'onset_trim_s':0,'roi':roi,'first_sample':j*100+1,'last_sample':j*100+100})
    savemat(inputs/'spectra_001.mat',{'spectra':spectra,'metadata':{'source':{'participant':'测试甲'}}})
    pd.DataFrame(powers).to_csv(inputs/'paired_powers.csv',index=False,encoding='utf-8-sig')
    for version in 'ABC':
        (inputs/version).mkdir()
        family=pd.DataFrame({'onset_trim_s':[0,0],'outcome':['x','y'],'model':['Model1']*2,'term':['x']*2,'family_id':['f']*2,
            'p.value':[.01,.2],'inference_valid':[True,True]})
        for kind in ('coefficients','factors'):family.to_csv(inputs/version/f'family_{kind}.csv',index=False)
    r_call({'rscript':str(r),'r_library':str(library)},root,Path('analysis/r/eeg_request_read_verify.R'),
       {'reads':[{'id':'input','path':str(inputs/'original.csv')}],'questionnaire_file':str(inputs/'original.xlsx'),
        'questionnaire_columns':headers,'source_run':str(inputs),'cache':{'cache_dir':str(inputs),'sources':[{'participant':'测试甲'}]},'output':str(output)},output,'read')
    assert compare_cells(raw_csv(inputs/'original.csv'),raw_csv(output/'r_reads/input.csv'))==8
    integrals=pd.read_csv(output/'r_psd_integrals.csv')
    assert len(integrals)==6
    assert np.allclose(integrals.total_1_45,44)
    assert raw_csv(output/'r_Q1_4_groups.csv').iloc[0].ExperienceGroup=='Low'
