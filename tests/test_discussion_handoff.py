import hashlib
import json
import zipfile

import pytest

from paper_analysis.teacher.discussion_handoff import extend_tables, refresh_package_hashes, update_pointer_tree, validate_zip


def test_extension_preserves_existing_numeric_values_and_source_ids():
    payload = {'tables': [
        {'name': '先看这里', 'columns': ['项目', '说明'], 'rows': [['old', 'analysis-sha']]},
        {'name': '文件总清单', 'columns': ['Source ID', 'SHA256'], 'rows': [['SRC0001', 'original-hash']]},
        {'name': '段落来源映射', 'columns': ['内容', 'Source ID'], 'rows': [['original', 'SRC0001']]},
        {'name': '关键数字核对', 'columns': ['p', 'q'], 'rows': [[.049, .071]]},
        {'name': '术语与字段说明', 'columns': ['术语', '说明'], 'rows': []}]}
    original = json.loads(json.dumps(payload))
    changed = extend_tables(payload, {'document_name': 'Discussion.docx', 'paragraph_rows': [{'内容': 'new', 'Source ID': 'SRC0002'}]},
                            [{'Source ID': 'SRC0002', 'SHA256': 'new-hash'}])
    assert payload == original
    for before, after in zip(original['tables'], changed['tables']):
        assert after['rows'][:len(before['rows'])] == before['rows']
    assert changed['tables'][1]['rows'][-1] == ['SRC0002', 'new-hash']


def test_only_current_zip_references_are_updated_without_relabeling_calculations():
    original = {'git_commit': 'numerical', 'publication_git_sha': 'original-publisher', 'zip': 'current.zip',
                'nested': {'zip': 'current.zip'}, 'previous': {'zip': 'archive/old.zip', 'zip_sha256': 'historical'}}
    changed = update_pointer_tree(original, 'current.zip', 'new.zip', 'new-sha', {'code_sha': 'discussion'})
    assert changed['git_commit'] == 'numerical' and changed['publication_git_sha'] == 'original-publisher'
    assert changed['nested']['zip'] == 'new.zip' and changed['previous'] == original['previous']
    assert original['zip'] == 'current.zip'


def test_zip_rejects_extra_members_and_changed_bytes(tmp_path):
    target = tmp_path/'delivery.zip'
    expected = {'中文.md': hashlib.sha256(b'original').hexdigest()}
    with zipfile.ZipFile(target, 'w') as z: z.writestr('中文.md', b'original')
    validate_zip(target, expected)
    with zipfile.ZipFile(target, 'a') as z: z.writestr('unexpected.txt', b'extra')
    with pytest.raises(ValueError, match='inventory'): validate_zip(target, expected)
    with zipfile.ZipFile(target, 'w') as z: z.writestr('中文.md', b'changed')
    with pytest.raises(ValueError, match='bytes'): validate_zip(target, expected)


def test_portable_hash_manifest_replaces_stale_index_digest(tmp_path):
    (tmp_path/'filesource_flat').mkdir()
    (tmp_path/'index.xlsx').write_bytes(b'updated-index')
    manifest = tmp_path/'filesource_flat/交付文件哈希.json'
    manifest.write_text('[{"path":"index.xlsx","sha256":"old"}]')
    hashes = refresh_package_hashes(tmp_path)
    entries = json.loads(manifest.read_text(encoding='utf-8'))
    assert entries == [{'path': 'index.xlsx', 'sha256': hashlib.sha256(b'updated-index').hexdigest()}]
    assert 'filesource_flat/交付文件哈希.json' in hashes
