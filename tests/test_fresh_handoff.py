import json
from pathlib import Path

import pandas as pd

from paper_analysis.teacher import fresh_handoff as handoff
import pytest


def test_calculation_identity_is_preserved_when_publisher_changes(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'git_commit':'numerical-version'}))
    assert handoff.calculation_sha(tmp_path)=='numerical-version'


def test_generated_files_and_archived_results_cannot_become_new_sources(tmp_path):
    for name in ['fresh_summary.json','handoff_guide.json','formal_index_tables.json','previous_publication.json',
                 'artifact_data_verification.json','published_data_verification.json']:
        (tmp_path/name).write_text('{}')
    (tmp_path/'superseded_formal_results').mkdir()
    (tmp_path/'superseded_formal_results/old.csv').write_text('estimate\n99\n')
    (tmp_path/'current').mkdir()
    current=tmp_path/'current/model.csv';current.write_text('estimate\n1\n')
    sources=handoff.collect_sources(tmp_path)
    assert len(sources)==1 and sources[0]['source_path']==str(current)


def test_marginal_directions_use_coefficient_names_and_preserve_analysis_sha(tmp_path, monkeypatch):
    run=tmp_path/'run'; main=run/'10_eeg_denominator_sensitivity/main_1_45'
    main.mkdir(parents=True); primary=run/'06_eeg_primary';primary.mkdir()
    (run/'run_manifest.json').write_text(json.dumps({'git_commit':'analysis'}))
    monkeypatch.setattr(handoff,'git_sha',lambda _: 'verification')
    pd.DataFrame([{'outcome':'P_beta_relative','term':'WWRWWR45','estimate':.02,'p.value':.1}]).to_csv(primary/'06_eeg_CR2_results.csv',index=False)
    coefficients=[];contrasts=[]
    for trim in [0,5,10,15]:
        for term,value in [('WWRWWR75',-.01),('WWRWWR45',.02)]:
            coefficients.append({'onset_trim_s':trim,'model':'Model1','outcome':'P_beta_relative','term':term,'estimate':value,'p.value':.1})
        for row_id in [1,2]:
            for term in ['WWRWWR45','WWRWWR75']:
                contrasts.append({'onset_trim_s':trim,'outcome':'P_beta_relative','term':'WWR','contrast_row':row_id,
                                  'coefficient':term,'weight':int(term=={1:'WWRWWR45',2:'WWRWWR75'}[row_id])})
    pd.DataFrame(coefficients).to_csv(main/'coefficients.csv',index=False)
    pd.DataFrame(contrasts).to_csv(main/'contrast_matrices.csv',index=False)
    handoff.final_readonly_checks({},run,tmp_path)
    actual=pd.read_csv(run/'12_final_verification/P_beta_WWR_marginal_directions.csv')
    assert actual.equal_margin_difference.tolist()==[.02,-.01]*4
    identity=json.loads((run/'12_final_verification/final_readonly_checks.json').read_text())
    assert identity['analysis_sha']=='analysis' and identity['verification_sha']=='verification'
    coefficients[5]['p.value']=.049
    pd.DataFrame(coefficients).to_csv(main/'coefficients.csv',index=False)
    with pytest.raises(handoff.StageBlockedError,match='values differ'):
        handoff.final_readonly_checks({},run,tmp_path)


def test_formal_entries_replace_old_counts_and_preserve_deferred_questionnaire(tmp_path):
    outputs=tmp_path/'outputs';outputs.mkdir()
    run=outputs/'teacher_runs/new';run.mkdir(parents=True)
    (run/'run_manifest.json').write_text(json.dumps({'git_commit':'analysis'}))
    (run/'结果文件总索引.xlsx').write_bytes(b'new-index')
    (run/'fresh_summary.json').write_text(json.dumps({'eye':{'participants':46,'trials':434},'eeg':{'participants':41,'trials':453}}))
    pd.DataFrame([{'run_id':'old','questionnaire_participants':42,'eeg_participants':42}]).to_csv(outputs/'realdata_run_summary.csv',index=False)
    (outputs/'结果文件总索引.xlsx').write_bytes(b'old-index')
    (outputs/'code_publication.json').write_text('{"git_commit":"old"}')
    oldzip=str(tmp_path/'delivery/old.zip'); archived=str(tmp_path/'archive/old.zip')
    (outputs/'eye_eeg_incremental_latest.json').write_text(json.dumps({'zip':oldzip,'zip_sha256':'old-hash','git_sha':'original-audit'}))
    pointer={'publication_git_sha':'publisher','zip':'delivery/new.zip','zip_sha256':'new-hash','source_count':10,'delivery_root':'delivery'}
    handoff.refresh_formal_entrypoints(outputs,run,pointer,{oldzip:archived})
    actual=pd.read_csv(outputs/'realdata_run_summary.csv').iloc[0]
    assert actual.eeg_participants==41 and actual.eeg_common_trials==453
    assert actual.questionnaire_participants==42 and actual.questionnaire_status=='not_rerun'
    assert actual.git_commit=='analysis' and actual.publication_git_sha=='publisher'
    assert (outputs/'结果文件总索引.xlsx').read_bytes()==b'new-index'
    old=run/'superseded_formal_results/root_entrypoints'
    assert (old/'结果文件总索引.xlsx').read_bytes()==b'old-index'
    assert pd.read_csv(old/'realdata_run_summary.csv').iloc[0].eeg_participants==42
    updated=json.loads((outputs/'eye_eeg_incremental_latest.json').read_text(encoding='utf-8'))
    assert updated['zip']==pointer['zip'] and updated['zip_sha256']=='new-hash'
    history=updated['previous_audit_and_delivery']['previous_delivery']
    assert history['zip']==archived and history['zip_sha256']=='old-hash'
    assert history['git_sha']=='original-audit'
