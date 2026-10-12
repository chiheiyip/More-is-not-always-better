import hashlib
import json
from pathlib import Path

import pytest

from paper_analysis.teacher import discussion_handoff as legacy
from paper_analysis.teacher import manuscript_handoff as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return str(path)


@pytest.fixture
def package(tmp_path, monkeypatch):
    delivery = tmp_path/'delivery';delivery.mkdir(); (delivery/'filesource_flat').mkdir()
    science = tmp_path/'science'; science.mkdir()
    original = tmp_path/'original.docx'; original.write_bytes(b'original-manuscript')
    data = science/'model.csv';data.write_bytes(b'term,estimate,p\nx,0.12,0.03\n')
    write(science/'run_manifest.json', {'git_commit': 'original-calculation'})
    for name in ['source_hashes_before.json', 'source_hashes_after.json']:
        write(science/name, [{'path': str(data), 'sha256': m.file_sha256(data)}])
    (delivery/'filesource_flat/SRC0001__model.csv').write_bytes(data.read_bytes())
    records = [{'Source ID': 'SRC0001', 'source_path': str(data), 'flat_name': 'SRC0001__model.csv', 'SHA256': m.file_sha256(data)}]
    manifest = write(science/'source_manifest.json', records)
    artifacts = {}
    for name in ['论文数据分析结果报告.md', '数据来源交接索引.xlsx', '数据来源交接说明.docx', 'results.docx', 'discussion.docx', 'manuscript.docx']:
        path = tmp_path/'authored'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(('new-'+name).encode());artifacts[name]=str(path)
        if name in m.BASE:(delivery/name).write_bytes(('old-'+name).encode())
    old_zip = delivery/'old.zip';old_zip.write_bytes(b'old-archive')
    proof = write(science/'reproducibility.json', {'status': 'passed', 'comparison_mode': 'exact_numeric', 'verification_level': 'complete_declared_scope', 'verification_sha': 'calculation-verifier', 'scopes': {k: {'status': 'passed', 'valid_inference_rows': 1, 'invalid_inference_rows': 0} for k in ['eye', 'eeg', 'joint']}, 'inputs': [], 'evidence': [], 'full_bootstrap_repeat': {'status': 'passed', 'fresh_repeated_fits': 80000, 'evidence': []}})
    environment = write(science/'environment.json', {})
    index = write(tmp_path/'index.json', {'tables': [{'name': n, 'columns': [], 'rows': []} for n in m.SHEETS]})
    script = tmp_path/'statistics.R';script.write_bytes(b'unchanged statistics')
    cfg = {'mode': 'manuscript', 'delivery_root': str(delivery), 'work_root': str(tmp_path/'work'), 'archive_path': str(tmp_path/'archive'), 'manuscript_file': str(original), 'source_run': str(science), 'source_manifest': manifest, 'analysis_config': environment, 'reproducibility_proof': proof, 'bootstrap_repeat_proof': proof, 'statistical_code_hashes': [{'path': str(script), 'sha256': m.file_sha256(script)}], 'previous_zip_name': 'old.zip', 'previous_zip_sha256': m.file_sha256(old_zip), 'zip_name': 'new.zip', 'artifacts': artifacts, 'index_payload': index, 'additions': [], 'formal_index_source': artifacts['数据来源交接索引.xlsx'], 'formal_targets': {}, 'pointer_files': []}
    cfg['inputs']=[{'path': p, 'sha256': m.file_sha256(p)} for p in [str(original), proof, environment, manifest, index]]
    monkeypatch.setattr(m, 'validate_analysis', lambda *a, **k: {'status': 'passed'})
    monkeypatch.setattr(m, 'verify_complete_bootstrap_repeat', lambda *a: {'status': 'passed'})
    monkeypatch.setattr(m, 'git_sha', lambda *a: 'authoring-sha')
    return cfg, tmp_path


def qa(config):
    stage = Path(config['work_root'])/'staging'
    write(Path(config['work_root'])/'artifact_verification.json', {'files': {n: {'sha256': m.file_sha256(stage/n), 'content_verified': True, 'visually_reviewed': True} for n in config['artifacts']}, 'source_and_index_checks_passed': True, 'manuscript_consistency_passed': True, 'formal_index_sha256': m.file_sha256(config['formal_index_source'])})


def test_complete_package_reuses_sources_without_relabelling_scientific_sha(package):
    config, root = package
    pointer = root/'latest.json';write(pointer, {'git_commit': 'original-calculation', 'zip': str(Path(config['delivery_root'])/'old.zip')})
    config['pointer_files']=[str(pointer)]
    previous = m.inventory(Path(config['delivery_root']))
    assert len(legacy.check(config)) == 1
    legacy.prepare(config, root);qa(config);result=legacy.publish(config, root)
    assert result['document_generation_sha']=='authoring-sha'
    assert m.inventory(Path(config['archive_path'])) == previous
    updated=json.loads(pointer.read_text());assert updated['git_commit']=='original-calculation'
    assert updated['zip'].endswith('new.zip') and len(updated['manuscript_delivery']['documents'])==4
    assert Path(config['manuscript_file']).read_bytes()==b'original-manuscript'


def test_incomplete_or_changed_proof_and_missing_hash_binding_are_rejected(package):
    config, root=package
    proof=m.load(config['reproducibility_proof']);proof['scopes']['eeg']['invalid_inference_rows']=1
    write(Path(config['reproducibility_proof']),proof)
    with pytest.raises(ValueError,match='hash'): m.check(config)
    for item in config['inputs']:item['sha256']=m.file_sha256(item['path'])
    with pytest.raises(ValueError,match='Invalid inference'):m.check(config)
    config['inputs']=[]
    with pytest.raises(ValueError,match='hash-bound'):m.check(config)


def test_stale_visual_qa_stops_publication(package):
    config,root=package;m.prepare(config,root);qa(config)
    stage=Path(config['work_root'])/'staging';(stage/'results.docx').write_bytes(b'changed-after-review')
    with pytest.raises(ValueError,match='stale'):m.publish(config,root)
    assert not Path(config['archive_path']).exists()


def test_entrypoint_failure_restores_old_delivery_and_formal_file(package,monkeypatch):
    config,root=package;target=root/'formal.docx';target.write_bytes(b'old-formal')
    config['formal_targets']={'manuscript.docx':[str(target)]}
    old=m.inventory(Path(config['delivery_root']));m.prepare(config,root);qa(config)
    original=Path.write_bytes;failed=False
    def fail_once(path,data):
        nonlocal failed
        if path==target.resolve() and not failed:
            failed=True;raise OSError('simulated formal write failure')
        return original(path,data)
    monkeypatch.setattr(Path,'write_bytes',fail_once)
    with pytest.raises(OSError,match='simulated'):m.publish(config,root)
    assert m.inventory(Path(config['delivery_root']))==old and target.read_bytes()==b'old-formal'


def test_original_and_scientific_targets_are_protected(package):
    config,_=package;config['formal_targets']={'manuscript.docx':[config['manuscript_file']]}
    with pytest.raises(ValueError,match='protected'):m.check(config)


def test_historical_zip_pointer_stays_unchanged():
    original={'git_commit':'science','zip':'current.zip','previous':{'zip':'historical.zip'}}
    new=m.updated_pointer(original,'current.zip',{'zip':'new.zip','zip_sha256':'digest','source_count':3})
    assert new['git_commit']=='science' and new['previous']==original['previous']
    assert original['zip']=='current.zip'
