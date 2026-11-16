param(
    [string]$Output = 'runs\visual_state_v2_matched',
    [string]$Dataset = 'data\visual_state_v2_matched',
    [int]$Scenarios = 300,
    [int]$Seed = 43,
    [int]$MaxNewTokens = 12,
    [switch]$SkipNoImage,
    [switch]$SkipPermutation
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Run scripts\setup.ps1 first' }
$manifest = Join-Path $Dataset 'manifest.jsonl'
if (-not (Test-Path -LiteralPath $manifest)) {
    & $runtime -m clickagent visual-state-dataset --output $Dataset --scenarios $Scenarios --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw 'Visual-state dataset generation failed' }
}
$dataSummary = Get-Content -LiteralPath (Join-Path $Dataset 'summary.json') -Raw | ConvertFrom-Json
if ($dataSummary.num_scenarios -ne $Scenarios -or $dataSummary.seed -ne $Seed) {
    throw 'Existing dataset has different Scenarios/Seed. Select a new -Dataset and -Output directory.'
}
& $runtime -m clickagent run --manifest $manifest --task visual_state --output $Output --seed $Seed --gpu-memory 13GiB --max-new-tokens $MaxNewTokens --conditions prefill post_decision --min-common-coverage 1
if ($LASTEXITCODE -ne 0) { throw 'Visual extraction/evaluation failed; completed samples are checkpointed' }
if (-not $SkipNoImage) {
    $noImage = Join-Path $Output 'no_image'
    & $runtime -m clickagent constant-no-image --manifest $manifest --output (Join-Path $noImage 'features') --seed $Seed --gpu-memory 13GiB --max-new-tokens $MaxNewTokens
    if ($LASTEXITCODE -ne 0) { throw 'Constant-input no-image extraction failed' }
    & $runtime -m clickagent evaluate --manifest $manifest --task visual_state --features (Join-Path $noImage 'features') --output (Join-Path $noImage 'evaluation') --seed $Seed --min-common-coverage 1
    if ($LASTEXITCODE -ne 0) { throw 'No-image control failed' }
}
if (-not $SkipPermutation) {
    & $runtime scripts\run_permutation_control.py --manifest $manifest --features (Join-Path $Output 'features') --output (Join-Path $Output 'permutation')
    if ($LASTEXITCODE -ne 0) { throw 'Permutation control failed' }
}
