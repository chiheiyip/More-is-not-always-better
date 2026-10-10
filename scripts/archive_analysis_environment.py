"""Create a sealed offline archive or restore it into a new directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from paper_analysis.teacher.runtime_lock import validate_python, validate_r
from paper_analysis.teacher.state import file_sha256

def pack(source, target):
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(source.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(source).as_posix())

def unpack(source, target):
    with zipfile.ZipFile(source) as z:
        for entry in z.infolist():
            p=(target/entry.filename).resolve()
            if not p.is_relative_to(target.resolve()):raise ValueError('Unsafe archive path')
        z.extractall(target)

def create(root, independent_r, independent_library):
    validate_python()
    validate_r(str(REPO/'scripts/portable_rscript.cmd'),profile='primary')
    validate_r(str(independent_r/'bin/x64/Rscript.exe'),profile='independent',library=str(independent_library))
    root.mkdir(parents=True,exist_ok=False)
    wheels=root/'wheels'; wheels.mkdir()
    subprocess.run([sys.executable,'-m','pip','download','--only-binary=:all:','--no-deps',
                    '-r',str(REPO/'requirements-analysis.lock.txt'),'-d',str(wheels)],check=True)
    pack(REPO/'.r-env',root/'r-primary.zip')
    pack(independent_r,root/'r-independent.zip')
    pack(independent_library,root/'r-independent-library.zip')
    pack(Path(sys.base_prefix),root/'python-base.zip')
    # Hash-enforced requirements bind each package version to its archived wheel.
    import email
    requirements=[]
    for path in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(path) as z:
            metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        requirements.append(f"{metadata['Name']}=={metadata['Version']} --hash=sha256:{file_sha256(path)}")
    (root/'requirements-offline.txt').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
    files={p.relative_to(root).as_posix():file_sha256(p) for p in root.rglob('*') if p.is_file()}
    lock_paths=['requirements-analysis.lock.txt','configs/analysis_python.lock.json',
                'configs/analysis_native_python.lock.json','analysis/r/runtime-versions.lock.json']
    manifest={'schema':1,'python':sys.version,'files':files,
              'locks':{p:file_sha256(REPO/p) for p in lock_paths}}
    (root/'archive_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'archive':str(root),'manifest_sha256':file_sha256(root/'archive_manifest.json')}))

def restore(root, target, expected_manifest_sha):
    if file_sha256(root/'archive_manifest.json')!=expected_manifest_sha:
        raise ValueError('Archive manifest SHA256 mismatch')
    manifest=json.loads((root/'archive_manifest.json').read_text(encoding='utf-8'))
    for relative,digest in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=digest:
            raise ValueError('Archived installation file changed: '+relative)
    for relative,digest in manifest['locks'].items():
        if file_sha256(REPO/relative)!=digest:raise ValueError('Environment lock changed: '+relative)
    if target.exists():raise ValueError('Restore requires a new directory; existing environments are protected')
    if sys.version_info[:3]!=(3,12,10):raise ValueError('Restore requires Python 3.12.10')
    target.mkdir(parents=True)
    unpack(root/'python-base.zip',target/'python-base')
    subprocess.run([str(target/'python-base/python.exe'),'-m','venv',str(target/'python')],check=True)
    python=target/'python/Scripts/python.exe'
    subprocess.run([str(python),'-m','pip','install','--no-index','--require-hashes',
        '--find-links',str(root/'wheels'),'-r',str(root/'requirements-offline.txt')],check=True)
    for name in ('r-primary','r-independent','r-independent-library'):
        unpack(root/(name+'.zip'),target/name)
    rroot=target/'r-primary'
    launcher=target/'portable_rscript.cmd'
    launcher.write_text('@echo off\nset "R_HOME='+str(rroot/'Lib/R')+'"\nset "PATH='+
        str(rroot/'Lib/R/bin/x64')+';'+str(rroot/'Library/bin')+';%PATH%"\n"'+
        str(rroot/'Lib/R/bin/Rscript.exe')+'" %*\n',encoding='utf-8')
    env=os.environ.copy();env['PYTHONPATH']=str(REPO/'src')
    expression=('import json;from paper_analysis.teacher.runtime_lock import validate_python;'
                'print(json.dumps(validate_python()))')
    p=subprocess.run([str(python),'-c',expression],env=env,capture_output=True,text=True,check=True)
    proof={'python':json.loads(p.stdout),'primary':validate_r(str(launcher),profile='primary'),
        'independent':validate_r(str(target/'r-independent/bin/x64/Rscript.exe'),
            profile='independent',library=str(target/'r-independent-library')),
        'archive_manifest_sha256':expected_manifest_sha}
    (target/'restore_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('Verified offline restoration: '+str(target))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['archive','restore'],required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--target',type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--independent-r',type=Path)
    parser.add_argument('--independent-library',type=Path)
    args=parser.parse_args()
    if args.phase=='archive':
        if not args.independent_r or not args.independent_library:parser.error('Archive requires both independent R paths')
        create(args.archive,args.independent_r,args.independent_library)
    else:
        if not args.target or not args.manifest_sha256:parser.error('Restore requires target and trusted manifest SHA256')
        restore(args.archive,args.target,args.manifest_sha256)

if __name__=='__main__':main()
