param([string]$Output = 'runs\pilot_v3', [int]$Screenshots = 24, [int]$MaxNewTokens = 8)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Run scripts\setup.ps1 first' }
$datasetDir = Join-Path $Output 'dataset'
$manifest = Join-Path $datasetDir 'manifest.jsonl'
if (-not (Test-Path -LiteralPath $manifest)) {
    & $runtime -m clickagent dataset --output $datasetDir --screenshots $Screenshots --samples-per-image 3
    if ($LASTEXITCODE -ne 0) { throw 'Dataset generation failed' }
}
& $runtime -m clickagent run --manifest $manifest --output $Output --gpu-memory 13GiB --max-new-tokens $MaxNewTokens --conditions prefill post_decision
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; features are checkpointed and can be resumed' }
