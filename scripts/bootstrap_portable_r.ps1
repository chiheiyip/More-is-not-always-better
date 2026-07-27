[CmdletBinding()]
param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$toolsRoot = Join-Path $repoRoot ".tools"
$archivePath = Join-Path $toolsRoot "micromamba-win-64.tar.bz2"
$micromamba = Join-Path $toolsRoot "Library\bin\micromamba.exe"
$mambaRoot = Join-Path $toolsRoot "mamba-root"
$environmentPath = Join-Path $repoRoot ".r-env"
$environmentFile = Join-Path $repoRoot "analysis\r\environment.yml"

New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null

if (-not (Test-Path -LiteralPath $micromamba)) {
    Invoke-WebRequest `
        -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" `
        -OutFile $archivePath
    tar -xf $archivePath -C $toolsRoot
}

$env:MAMBA_ROOT_PREFIX = $mambaRoot

if ($Recreate -and (Test-Path -LiteralPath $environmentPath)) {
    $resolvedRepo = [System.IO.Path]::GetFullPath($repoRoot)
    $resolvedEnvironment = [System.IO.Path]::GetFullPath($environmentPath)
    if (-not $resolvedEnvironment.StartsWith(
        $resolvedRepo + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove an R environment outside the repository."
    }
    & $micromamba remove -y -p $environmentPath --all
}

& $micromamba create -y -p $environmentPath -f $environmentFile `
    --repodata-ttl 86400 --no-exp-repodata-parsing
if ($LASTEXITCODE -ne 0) {
    throw "Portable R environment creation failed with exit code $LASTEXITCODE."
}

$rscriptCandidates = @(
    (Join-Path $environmentPath "Scripts\Rscript.exe"),
    (Join-Path $environmentPath "Lib\R\bin\x64\Rscript.exe"),
    (Join-Path $environmentPath "Lib\R\bin\Rscript.exe")
)
$rscript = $rscriptCandidates | Where-Object {
    Test-Path -LiteralPath $_
} | Select-Object -First 1

if (-not $rscript) {
    throw "Portable R was created, but Rscript.exe could not be located."
}

& $rscript -e @"
if (!requireNamespace("clubSandwich", quietly = TRUE)) {
    install.packages("clubSandwich", repos = "https://cloud.r-project.org")
}
required <- c("lme4", "glmmTMB", "emmeans", "clubSandwich", "broom.mixed", "renv")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
installed <- vapply(required, function(package) as.character(packageVersion(package)), character(1))
cat(R.version.string, "\n")
for (package in required) cat(package, installed[[package]], "\n")
"@

Write-Output "Portable R is ready: $rscript"
Write-Output "Delete '$environmentPath' and '$toolsRoot' to remove it completely."
