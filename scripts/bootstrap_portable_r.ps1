[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][string]$ManifestSHA256,
    [Parameter(Mandatory=$true)][string]$TargetRoot
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv-analysis\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Python 3.12.10 calculation runtime is required.' }
# No latest-version downloads, substitutions or removal of existing environments.
& $python (Join-Path $PSScriptRoot 'archive_analysis_environment.py') `
    --phase restore --archive $Archive --manifest-sha256 $ManifestSHA256 --target $TargetRoot
if ($LASTEXITCODE -ne 0) { throw 'Locked offline environment restoration failed.' }
