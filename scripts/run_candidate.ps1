param([string]$Output = 'runs\candidate_v2', [switch]$WaitForCollection)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$runtime = Join-Path $projectRoot '.venv\python.exe'
if ($WaitForCollection) {
    while (-not (Test-Path -LiteralPath (Join-Path $Output 'collection_complete.json'))) {
        Start-Sleep -Seconds 5
    }
    Start-Sleep -Seconds 3
} else {
    & $runtime -X utf8 -m clickagent.candidate_experiment --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Candidate collection failed' }
}
& $runtime -X utf8 -m clickagent.candidate_augment --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Training counterfactual collection failed' }
if (-not (Test-Path -LiteralPath (Join-Path $Output 'training_records.json'))) {
    & $runtime -X utf8 -m clickagent.candidate_probes --output $Output --train-only
    if ($LASTEXITCODE -ne 0) { throw 'Readout training/evaluation failed' }
}
if (-not (Test-Path -LiteralPath (Join-Path $Output 'training_weights_manifest.json'))) {
    & $runtime -X utf8 scripts/freeze_candidate_heads.py --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Head freeze record failed' }
}
& $runtime -X utf8 -m clickagent.candidate_probes --output $Output --evaluate-only
if ($LASTEXITCODE -ne 0) { throw 'Frozen-head evaluation failed' }
& $runtime -X utf8 scripts/summarize_candidates.py --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Result audit/report failed' }
& $runtime -X utf8 scripts/diagnose_candidate_first.py --output $Output
if ($LASTEXITCODE -ne 0) { throw 'First-proposal diagnostics failed' }
