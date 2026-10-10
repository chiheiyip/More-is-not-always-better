"""Resolved numerical libraries and MATLAB provenance; no automatic relocking."""
from __future__ import annotations
import ctypes
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import scipy
from .state import StageBlockedError, file_sha256

THREAD_VARIABLES = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')

def python_native_snapshot():
    libraries = {}
    for module in (np, scipy):
        for path in sorted(Path(module.__file__).parent.parent.glob(module.__name__+'.libs/*.dll')):
            entry={'sha256':file_sha256(path)}
            if 'openblas' in path.name:
                library=ctypes.CDLL(str(path))
                names=('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                       'openblas_get_num_threads64_', 'openblas_get_num_threads')
                for name in names:
                    try:
                        function=getattr(library,name); function.restype=ctypes.c_int
                        entry['threads']=function(); break
                    except AttributeError:
                        continue
                if 'threads' not in entry:
                    raise StageBlockedError('Cannot verify OpenBLAS thread count: '+path.name)
            libraries[path.name]=entry
    return {'libraries':libraries,'thread_environment':{k:os.getenv(k,'') for k in THREAD_VARIABLES}}

def validate_python_native(repo):
    path=Path(repo)/'configs/analysis_native_python.lock.json'
    expected=json.loads(path.read_text(encoding='utf-8'))['snapshot']
    actual=python_native_snapshot()
    if actual != expected:
        raise StageBlockedError('Python numerical DLL/thread environment mismatch; restore registered runtime')
    return {'status':'passed','lock_sha256':file_sha256(path),'snapshot':actual}

def validate_matlab(matlab, eeglab_root, repo):
    repo=Path(repo); lock=repo/'configs/analysis_matlab.lock.json'
    expected=json.loads(lock.read_text(encoding='utf-8'))['snapshot']
    quote=lambda p: str(p).replace('\\','/').replace("'","''")
    expression=(f"addpath('{quote(repo/'matlab')}'); "
                f"s=analysis_runtime_snapshot('{quote(eeglab_root)}'); "
                "fprintf('ANALYSIS_MATLAB_LOCK=%s\\n',jsonencode(s));")
    try:
        result=subprocess.run([str(matlab),'-batch',expression],capture_output=True,
            text=True,encoding='utf-8',errors='replace',check=True,timeout=180)
        lines=[line.split('=',1)[1] for line in result.stdout.splitlines()
               if line.startswith('ANALYSIS_MATLAB_LOCK=')]
        if len(lines)!=1: raise ValueError('Missing unique MATLAB snapshot')
        actual=json.loads(lines[0])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise StageBlockedError('MATLAB environment probe failed: '+str(exc)) from exc
    if actual != expected:
        raise StageBlockedError('MATLAB/EEGLAB function path, version, hash or numerical settings mismatch')
    return {'status':'passed','lock_sha256':file_sha256(lock),'snapshot':actual}
