"""Append a reviewed Discussion artifact without relabeling numerical results.

Authoring runtimes are invoked separately. This module prepares immutable source
copies and index additions, then publishes only after hash-bound artifact QA.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .fresh import dump, git_sha
from .request_handoff import archive_publish
from .state import file_sha256

BASE = {'论文数据分析结果报告.md', '数据来源交接索引.xlsx', '数据来源交接说明.docx', 'filesource_flat'}


def inventory(root):
    return {p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def check(config):
    delivery, work = Path(config['delivery_root']), Path(config['work_root'])
    if work.resolve() == delivery.resolve() or work.resolve().is_relative_to(delivery.resolve()):
        raise ValueError('Supplement must be built outside delivery')
    for item in config['inputs']:
        if file_sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Input hash mismatch: ' + item['path'])
    expected = BASE | {config['previous_zip_name']}
    if {p.name for p in delivery.iterdir()} != expected:
        raise ValueError('Unexpected existing delivery layout')
    if file_sha256(delivery/config['previous_zip_name']) != config['previous_zip_sha256']:
        raise ValueError('Existing ZIP changed')
    records = json.loads(Path(config['source_manifest']).read_text(encoding='utf-8-sig'))
    for item in records:
        if Path(item['flat_name']).name != item['flat_name']:
            raise ValueError('Unsafe source filename')
        if file_sha256(delivery/'filesource_flat'/item['flat_name']) != item['SHA256']:
            raise ValueError('Existing source copy changed')
    ids = [item['Source ID'] for item in records]
    for item in config['additions']:
        if Path(item['flat_name']).name != item['flat_name']:
            raise ValueError('Unsafe supplement filename')
        ids.append(item['Source ID'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate Source ID')
    return records


def extend_tables(payload, config, records):
    result = copy.deepcopy(payload)
    tables = {t['name']: t for t in result['tables']}
    tables['先看这里']['rows'].append(['Discussion 新增成果', config['document_name'] + '；原稿4.1—4.5核对及英文框架；不新增统计计算'])
    tables['文件总清单']['rows'].extend([[r.get(c) for c in tables['文件总清单']['columns']] for r in records])
    for row in config['paragraph_rows']:
        tables['段落来源映射']['rows'].append([row.get(c) for c in tables['段落来源映射']['columns']])
    tables['术语与字段说明']['rows'].extend(config.get('terminology_rows', []))
    return result


def prepare(config, repo):
    records = check(config)
    work = Path(config['work_root']); stage = work/'staging'
    stage.mkdir(parents=True, exist_ok=False)
    delivery = Path(config['delivery_root'])
    old = inventory(delivery)
    for name in BASE:
        source = delivery/name
        if source.is_dir(): shutil.copytree(source, stage/name)
        else: shutil.copy2(source, stage/name)
    additions = []
    for item in config['additions']:
        p = Path(item['path'])
        r = {'Source ID': item['Source ID'], 'source_path': str(p), 'flat_name': item['flat_name'],
             'stage': 'discussion', 'bytes': p.stat().st_size, 'SHA256': file_sha256(p)}
        shutil.copy2(p, stage/'filesource_flat'/r['flat_name'])
        additions.append(r)
    tables = json.loads(Path(config['index_payload']).read_text(encoding='utf-8-sig'))
    updated = extend_tables(tables, config, additions)
    dump(work/'index_tables.json', updated)
    formal = copy.deepcopy(updated)
    formal['linkPrefix'] = (delivery/'filesource_flat').as_posix() + '/'
    dump(work/'formal_index_tables.json', formal)
    dump(work/'source_manifest.json', records + additions)
    dump(stage/'filesource_flat/source_manifest.json', records + additions)
    dump(work/'preparation.json', {'code_sha': git_sha(repo), 'old_inventory': old,
         'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
         'original_source_count': len(records), 'added_source_count': len(additions)})
    return stage


def validate_zip(path, hashes):
    with zipfile.ZipFile(path) as z:
        if set(z.namelist()) != set(hashes) or len(z.namelist()) != len(hashes) or z.testzip() is not None:
            raise ValueError('ZIP inventory/integrity differs')
        for name, digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('ZIP bytes differ: ' + name)


def refresh_package_hashes(stage):
    """The portable manifest describes this package, excluding its own bytes."""
    manifest = stage/'filesource_flat/交付文件哈希.json'
    hashes = {name: digest for name, digest in inventory(stage).items()
              if name != manifest.relative_to(stage).as_posix()}
    dump(manifest, [{'path': name, 'sha256': digest} for name, digest in hashes.items()])
    return inventory(stage)


def update_pointer_tree(value, old_zip, new_zip, digest, supplement):
    """Update current delivery references only; preserve archived historical ZIPs."""
    if isinstance(value, list):
        return [update_pointer_tree(v, old_zip, new_zip, digest, supplement) for v in value]
    if not isinstance(value, dict): return value
    out = {k: update_pointer_tree(v, old_zip, new_zip, digest, supplement) for k, v in value.items()}
    if out.get('zip') == old_zip:
        out.update(zip=new_zip, zip_sha256=digest, discussion_supplement=supplement)
        if out.get('source_count') == supplement.get('original_source_count'):
            out['original_source_count'] = out['source_count']
            out['source_count'] = supplement['source_count']
    return out


def publish(config, repo):
    work = Path(config['work_root']); stage = work/'staging'
    preparation = json.loads((work/'preparation.json').read_text(encoding='utf-8'))
    if preparation['code_sha'] != git_sha(repo): raise ValueError('Code changed after preparation')
    expected_config = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if preparation['config_sha256'] != expected_config: raise ValueError('Configuration changed')
    check(config)
    expected = BASE | {config['document_name']}
    if {p.name for p in stage.iterdir()} != expected: raise ValueError('Unexpected supplement staging layout')
    qa = json.loads((work/'artifact_verification.json').read_text(encoding='utf-8'))
    for name in [config['document_name'], '数据来源交接索引.xlsx']:
        if qa['files'][name]['sha256'] != file_sha256(stage/name) or not qa['files'][name]['visually_reviewed']:
            raise ValueError('Artifact QA missing or stale')
    if not qa['source_and_index_checks_passed']: raise ValueError('Data/source checks failed')
    if qa['formal_index_sha256'] != file_sha256(work/'formal_index.xlsx'):
        raise ValueError('Formal index QA stale')
    for name in ['论文数据分析结果报告.md', '数据来源交接说明.docx']:
        if file_sha256(stage/name) != preparation['old_inventory'][name]: raise ValueError('Unrelated artifact changed')
    records = json.loads((work/'source_manifest.json').read_text(encoding='utf-8'))
    for item in records:
        if file_sha256(stage/'filesource_flat'/item['flat_name']) != item['SHA256']:
            raise ValueError('Source bytes changed')
    payload_hashes = refresh_package_hashes(stage)
    zpath = stage/config['zip_name']
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in payload_hashes: z.write(stage/name, name)
    validate_zip(zpath, payload_hashes)
    current_hashes = inventory(stage)
    delivery, archive = Path(config['delivery_root']), Path(config['archive_path'])
    def validate(current, historical):
        if inventory(current) != current_hashes or inventory(historical) != preparation['old_inventory']:
            raise ValueError('Published or archived bytes differ')
    if inventory(delivery) != preparation['old_inventory']: raise ValueError('Delivery changed since preparation')
    mapping = archive_publish(stage, delivery, archive, validate)
    dump(work/'archive_path_mapping.json', mapping)
    supplement = {'document': str(delivery/config['document_name']), 'code_sha': git_sha(repo),
                  'work_root': str(work), 'scope': 'Discussion only; numerical calculations unchanged',
                  'archive_path': str(archive),
                  'original_source_count': preparation['original_source_count'], 'source_count': len(records)}
    old_zip, new_zip = str(delivery/config['previous_zip_name']), str(delivery/config['zip_name'])
    digest = file_sha256(Path(new_zip))
    for raw in config['pointer_files']:
        p = Path(raw)
        value = json.loads(p.read_text(encoding='utf-8-sig'))
        backup = work/'previous_entrypoints'/p.parent.name/p.name
        backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, backup)
        dump(p, update_pointer_tree(value, old_zip, new_zip, digest, supplement))
    for raw in config.get('formal_document_targets', []):
        shutil.copy2(delivery/config['document_name'], Path(raw))
    for raw in config.get('formal_index_targets', []):
        p = Path(raw); backup = work/'previous_entrypoints'/p.parent.name/p.name
        backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, backup)
        shutil.copy2(work/'formal_index.xlsx', p)
    dump(work/'publication.json', {**supplement, 'zip': new_zip, 'zip_sha256': digest,
        'archive_path': str(archive), 'files': current_hashes, 'status': 'complete'})
    return supplement
