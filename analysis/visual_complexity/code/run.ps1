$ErrorActionPreference = 'Stop'
$taskCode = $PSScriptRoot
& (Join-Path $taskCode '.venv\Scripts\python.exe') -u (Join-Path $taskCode 'source\run_panorama_local.py') @args
if ($LASTEXITCODE -ne 0) { throw "Calculation failed with exit code $LASTEXITCODE" }
