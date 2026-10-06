import json
from pathlib import Path
import pandas as pd
import pytest
from paper_analysis.teacher import incremental_handoff as handoff


def config(tmp_path):
    return dict(outputs_root=str(tmp_path/'analysis'),run_id='incremental',
                delivery_root=str(tmp_path/'delivery'),archive_root=str(tmp_path/'archive'))


def test_preflight_is_read_only_and_rejects_nested_archive(tmp_path,monkeypatch):
    monkeypatch.setattr(handoff,'git_sha',lambda repo:'committed')
    cfg=config(tmp_path);plan=handoff.preflight(cfg,tmp_path)
    assert plan['missing_completed_stages'] and len(plan['sheets'])==9
    assert not (tmp_path/'analysis').exists()
    cfg['archive_root']=str(tmp_path/'delivery/history')
    with pytest.raises(ValueError,match='separate'):handoff.preflight(cfg,tmp_path)


def test_frozen_hash_or_reader_disagreement_blocks_publication(tmp_path,monkeypatch):
    monkeypatch.setattr(handoff,'git_sha',lambda repo:'committed')
    monkeypatch.setattr(handoff.subprocess,'check_output',lambda *args,**kwargs:'')
    cfg=config(tmp_path);root=handoff.location(cfg)
    (root/'eye').mkdir(parents=True);(root/'source_readers').mkdir()
    for file in ['eye_comparison_summary.json','eeg_comparison_summary.json','EEG_stage_reuse_manifest.json']:
        (root/file).write_text('{}')
    source=root/'eye/frozen.csv';source.write_text('original')
    record={'files':[{'path':str(source),'sha256':handoff.sha(source)}]}
    for file in ['independent_calculation_seal.json','canonical_calculation_seal.json']:
        (root/'eye'/file).write_text(json.dumps(record))
    path=root/'source_readers/Python_R_original_design_summary.csv'
    pd.DataFrame({'MismatchedRows':[0]}).to_csv(path,index=False)
    assert handoff.require_complete(cfg,tmp_path)==root
    source.write_text('changed')
    with pytest.raises(ValueError,match='Frozen'):handoff.require_complete(cfg,tmp_path)
    source.write_text('original');pd.DataFrame({'MismatchedRows':[1]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='readers'):handoff.require_complete(cfg,tmp_path)


def test_stale_artifact_qa_blocks_before_archiving(tmp_path,monkeypatch):
    cfg=config(tmp_path);root=handoff.location(cfg);stage=root/'handoff_staging'
    stage.mkdir(parents=True);(stage/'filesource_flat').mkdir()
    for name in ['论文数据分析结果报告.md','数据来源交接索引.xlsx','数据来源交接说明.docx']:
        (stage/name).write_text('artifact')
    (root/'artifact_verification.json').write_text(json.dumps({'files':{
        '数据来源交接索引.xlsx':{'visually_reviewed':True,'sha256':'stale'}}}))
    monkeypatch.setattr(handoff,'require_complete',lambda *args:root)
    with pytest.raises(ValueError,match='QA'):handoff.publish(cfg,tmp_path)
    assert not Path(cfg['archive_root']).exists()


def test_active_pointer_updates_zip_hash_and_preserves_archived_history(tmp_path):
    old={'zip':'old.zip','zip_sha256':'old-hash','git_sha':'original-calculation'}
    current={'zip':'new.zip','zip_sha256':'new-hash','delivery_root':'delivery','source_count':192,'publication_git_sha':'publisher'}
    actual=handoff.renew_delivery_pointer(old,current,{'old.zip':'archive/old.zip'},tmp_path)
    assert actual['zip']=='new.zip' and actual['zip_sha256']=='new-hash'
    assert actual['previous_delivery']['zip']=='archive/old.zip'
    assert actual['previous_delivery']['zip_sha256']=='old-hash'
    assert actual['git_sha']=='original-calculation' and actual['publication_git_sha']=='publisher'
    filename=str(tmp_path/'old.zip')
    doubled=filename.replace('\\','\\\\')
    actual=handoff.renew_delivery_pointer({'zip':doubled},current,{filename:'archive/old.zip'},tmp_path)
    assert actual['previous_delivery']['zip']=='archive/old.zip'


def test_missing_historical_zip_is_not_replaced_by_different_archive(tmp_path):
    archive=tmp_path/'archive';archive.mkdir();actual=archive/'renamed.zip';actual.write_bytes(b'actual-old-package')
    previous={'zip':str(tmp_path/'missing.zip'),'zip_sha256':'unavailable-original-hash'}
    record,status=handoff.verify_legacy_zip_reference(previous,{},archive)
    assert record==previous and status['state']=='unable_to_verify_historical_ZIP_reference'
    assert status['actual_archived_zip_candidates'][0]['sha256']==handoff.sha(actual)
    previous['zip_sha256']=handoff.sha(actual)
    record,status=handoff.verify_legacy_zip_reference(previous,{},archive)
    assert record['zip']==str(actual) and status['state']=='verified_by_recorded_hash'
