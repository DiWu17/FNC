param([string]$Output = 'runs\action_v3', [string]$Dataset = 'data\action_v3')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe' }
$manifest = Join-Path $Dataset 'manifest.jsonl'
if (-not (Test-Path -LiteralPath $manifest)) {
    & $runtime -c "from clickagent.action_dataset import build_action_dataset; build_action_dataset(r'$Dataset')"
    if ($LASTEXITCODE -ne 0) { throw 'Action data generation failed' }
}
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
& $runtime -X utf8 -m clickagent.action_benchmark --manifest $manifest --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Action benchmark failed; completed raw features and episodes are saved' }
