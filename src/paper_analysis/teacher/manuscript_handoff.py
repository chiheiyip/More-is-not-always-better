"""Publish a reviewed multi-document manuscript package from sealed results."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .fresh import dump, git_sha, verify_inputs
from .state import file_sha256
from .reproducibility import verify_seal, verify_complete_bootstrap_repeat
from .runtime_lock import validate_analysis
from .discussion_handoff import BASE, inventory, refresh_package_hashes, validate_zip
from .request_handoff import archive_publish

SHEETS = ['先看这里', '段落来源映射', '关键数字核对', '文件总清单', '样本与数据粒度',
          '功率与统计复核', '差异与阴性结果', '未复制原始数据', '术语与字段说明']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_name(value):
    if not value or '/' in value or '\\' in value or value in {'.', '..'} or ':' in value:
        raise ValueError('Unsafe package filename')
    return value


def validate_paths(config):
    delivery, work, archive = (Path(config[k]).resolve() for k in ['delivery_root', 'work_root', 'archive_path'])
    for a, b in [(delivery, work), (delivery, archive), (work, archive)]:
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('Delivery, work and archive directories must be disjoint')
    original = Path(config['manuscript_file']).resolve()
    science = Path(config['source_run']).resolve()
    for target in config.get('formal_targets', {}).values():
        for raw in target:
            path = Path(raw).resolve()
            if path == original or path.is_relative_to(science) or path.is_relative_to(delivery) or path.is_relative_to(archive):
                raise ValueError('Formal target would overwrite protected source or delivery')
    for raw in config.get('pointer_files', []):
        path = Path(raw).resolve()
        if path == original or path.is_relative_to(science) or path.is_relative_to(delivery) or path.is_relative_to(archive):
            raise ValueError('Pointer target would overwrite protected source')


def verify_reuse(config):
    source = Path(config['source_run'])
    before = load(source/'source_hashes_before.json')
    if not before or before != load(source/'source_hashes_after.json'):
        raise ValueError('Scientific source inventory missing or changed')
    verify_inputs(before)
    proof = load(config['reproducibility_proof'])
    if proof.get('status') != 'passed' or proof.get('comparison_mode') != 'exact_numeric' or proof.get('verification_level') != 'complete_declared_scope':
        raise ValueError('Complete exact reproducibility proof required')
    if set(proof.get('scopes', {})) != {'eye', 'eeg', 'joint'}:
        raise ValueError('Missing analysis scope')
    for value in proof['scopes'].values():
        if value['status'] != 'passed' or value['invalid_inference_rows'] or not value['valid_inference_rows']:
            raise ValueError('Invalid inference cannot establish reuse')
    verify_seal(proof['inputs']); verify_seal(proof['evidence'])
    bootstrap = proof['full_bootstrap_repeat']
    if bootstrap['status'] != 'passed' or bootstrap['fresh_repeated_fits'] != 80000:
        raise ValueError('Complete actual bootstrap repeat required')
    verify_seal(bootstrap['evidence'])
    verify_complete_bootstrap_repeat(config['bootstrap_repeat_proof'])
    if not config['statistical_code_hashes']:
        raise ValueError('Statistical code identity required')
    verify_seal(config['statistical_code_hashes'])
    environment = validate_analysis(load(config['analysis_config']), require_matlab=True)
    return {'status': 'passed', 'scientific_source_run': config['source_run'],
            'original_calculation_sha': load(source/'run_manifest.json')['git_commit'],
            'original_verification_sha': proof['verification_sha'], 'scopes': proof['scopes'],
            'actual_bootstrap_repeat_fits': 80000, 'environment': environment,
            'statistical_code': config['statistical_code_hashes'], 'source_inventory': before,
            'note': 'Sealed scientific results reused; authoring is not relabelled as numerical calculation.'}


def check(config):
    validate_paths(config)
    delivery = Path(config['delivery_root'])
    required = {str(Path(config[k]).resolve()) for k in ['manuscript_file', 'reproducibility_proof', 'analysis_config', 'source_manifest', 'index_payload']}
    if not required.issubset({str(Path(item['path']).resolve()) for item in config['inputs']}):
        raise ValueError('Manuscript, scientific proof, environment, source manifest and index must be hash-bound inputs')
    for item in config['inputs']:
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('Input hash mismatch: ' + item['path'])
    expected = BASE | {safe_name(config['previous_zip_name'])} | set(config.get('previous_documents', []))
    if {p.name for p in delivery.iterdir()} != expected:
        raise ValueError('Unexpected existing delivery layout')
    if file_sha256(delivery/config['previous_zip_name']) != config['previous_zip_sha256']:
        raise ValueError('Existing ZIP changed')
    artifacts = config['artifacts']
    if not BASE.difference({'filesource_flat'}).issubset(artifacts) or len([n for n in artifacts if n.endswith('.docx')]) < 4:
        raise ValueError('Complete report, index, guide and three manuscript documents required')
    for name, path in artifacts.items():
        safe_name(name)
        if not Path(path).is_file(): raise ValueError('Missing artifact: ' + name)
    records = load(config['source_manifest'])
    ids = []
    for record in records:
        safe_name(record['flat_name']); ids.append(record['Source ID'])
        if file_sha256(record['source_path']) != record['SHA256'] or file_sha256(delivery/'filesource_flat'/record['flat_name']) != record['SHA256']:
            raise ValueError('Existing scientific source changed')
    for item in config['additions']:
        safe_name(item['flat_name']); ids.append(item['Source ID'])
        if not Path(item['path']).is_file(): raise ValueError('Missing additional source')
    if len(ids) != len(set(ids)) or len({r['flat_name'] for r in records} | {r['flat_name'] for r in config['additions']}) != len(ids):
        raise ValueError('Duplicate Source ID or flat filename')
    payload = load(config['index_payload'])
    if [t['name'] for t in payload['tables']] != SHEETS:
        raise ValueError('Nine-sheet index contract differs')
    verify_reuse(config)
    return records


def prepare(config, repo):
    records = check(config)
    work = Path(config['work_root']); stage = work/'staging'
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(config['delivery_root'])/'filesource_flat', stage/'filesource_flat')
    for name, path in config['artifacts'].items(): shutil.copy2(path, stage/name)
    added = []
    for item in config['additions']:
        path = Path(item['path'])
        row = {'Source ID': item['Source ID'], 'source_path': str(path), 'flat_name': item['flat_name'],
               'stage': item.get('stage', 'manuscript'), 'bytes': path.stat().st_size, 'SHA256': file_sha256(path)}
        shutil.copy2(path, stage/'filesource_flat'/row['flat_name']); added.append(row)
    dump(work/'source_manifest.json', records+added)
    dump(stage/'filesource_flat/source_manifest.json', records+added)
    dump(work/'reuse_verification.json', verify_reuse(config))
    dump(work/'preparation.json', {'code_sha': git_sha(repo), 'config_sha256': config_digest(config),
         'old_inventory': inventory(Path(config['delivery_root'])), 'original_source_count': len(records),
         'artifacts': {n: file_sha256(stage/n) for n in config['artifacts']}})
    return stage


def updated_pointer(value, old_zip, package):
    if isinstance(value, list): return [updated_pointer(v, old_zip, package) for v in value]
    if not isinstance(value, dict): return value
    result = {k: updated_pointer(v, old_zip, package) for k, v in value.items()}
    if result.get('zip') == old_zip:
        result.update(zip=package['zip'], zip_sha256=package['zip_sha256'], source_count=package['source_count'],
                      manuscript_delivery=package)
    return result


def publish(config, repo):
    work = Path(config['work_root']); stage = work/'staging'
    prepared = load(work/'preparation.json')
    if prepared['code_sha'] != git_sha(repo) or prepared['config_sha256'] != config_digest(config):
        raise ValueError('Code/configuration changed after preparation')
    check(config)
    if {p.name for p in stage.iterdir()} != set(config['artifacts']) | {'filesource_flat'}:
        raise ValueError('Unexpected complete-package staging layout')
    qa = load(work/'artifact_verification.json')
    for name in config['artifacts']:
        entry = qa['files'][name]
        if entry['sha256'] != file_sha256(stage/name) or not entry['content_verified']:
            raise ValueError('Artifact content QA missing or stale')
        if name.endswith(('.docx', '.xlsx')) and not entry['visually_reviewed']:
            raise ValueError('Visual QA missing')
    if not qa['source_and_index_checks_passed'] or not qa['manuscript_consistency_passed']:
        raise ValueError('Source/index/manuscript checks failed')
    if qa['formal_index_sha256'] != file_sha256(config['formal_index_source']):
        raise ValueError('Formal index QA stale')
    records = load(work/'source_manifest.json')
    for item in records:
        if file_sha256(item['source_path']) != item['SHA256'] or file_sha256(stage/'filesource_flat'/item['flat_name']) != item['SHA256']:
            raise ValueError('Source bytes changed')
    hashes = refresh_package_hashes(stage)
    zip_name = safe_name(config['zip_name']); zip_path = stage/zip_name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes: archive.write(stage/name, name)
    validate_zip(zip_path, hashes)
    delivery, historical = Path(config['delivery_root']), Path(config['archive_path'])
    if inventory(delivery) != prepared['old_inventory']:
        raise ValueError('Delivery changed since preparation')
    package = {'status': 'complete', 'mode': 'manuscript', 'document_generation_sha': git_sha(repo),
               'work_root': str(work), 'scientific_source_run': config['source_run'],
               'source_count': len(records), 'documents': [str(delivery/n) for n in config['artifacts'] if n.endswith('.docx')],
               'zip': str(delivery/zip_name), 'zip_sha256': file_sha256(zip_path), 'archive_path': str(historical),
               'scope': 'Validated scientific-result reuse and complete manuscript delivery; questionnaire not rerun'}
    writes = {}
    for name, targets in config.get('formal_targets', {}).items():
        source = Path(config['formal_index_source']) if name == 'formal_index' else stage/safe_name(name)
        if not source.is_file(): raise ValueError('Missing formal artifact source')
        for target in targets:
            resolved = str(Path(target).resolve())
            if resolved in writes: raise ValueError('Duplicate formal target')
            writes[resolved] = source.read_bytes()
    old_zip = str(delivery/config['previous_zip_name'])
    for target in config.get('pointer_files', []):
        resolved = str(Path(target).resolve())
        if resolved in writes: raise ValueError('Duplicate pointer/formal target')
        value = updated_pointer(load(target), old_zip, package)
        writes[resolved] = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')
    backups = {}
    for i, target in enumerate(writes):
        path = Path(target); backup = work/'previous_entrypoints'/f'{i:03d}__{path.name}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): shutil.copy2(path, backup); backups[target] = backup
        else: backups[target] = None
    current_hashes = inventory(stage)
    def validate(current, old):
        if inventory(current) != current_hashes or inventory(old) != prepared['old_inventory']:
            raise ValueError('Published/archived bytes differ')
    mapping = archive_publish(stage, delivery, historical, validate)
    try:
        for target, data in writes.items():
            path = Path(target); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        for target, data in writes.items():
            if Path(target).read_bytes() != data: raise ValueError('Formal entrypoint bytes differ')
    except BaseException:
        for target, backup in backups.items():
            path = Path(target)
            if backup is None:
                if path.exists(): path.unlink()
            else: shutil.copy2(backup, path)
        failed = work/'failed_delivery'
        if failed.exists(): raise ValueError('Cannot safely restore: failed-delivery target exists')
        shutil.move(str(delivery), str(failed)); shutil.move(str(historical), str(delivery))
        raise
    dump(work/'archive_path_mapping.json', mapping)
    dump(work/'publication.json', package)
    return package
