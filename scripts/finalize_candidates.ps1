param([string]$Output = 'runs\candidate_v2')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
while (-not ((Test-Path -LiteralPath (Join-Path $Output 'collection_complete.json')) -and
             (Test-Path -LiteralPath (Join-Path $Output 'training_records.json')))) {
    Start-Sleep -Seconds 5
}
& .venv/python.exe -X utf8 -m clickagent.candidate_probes --output $Output --evaluate-only
if ($LASTEXITCODE -ne 0) { throw 'Frozen verifier evaluation failed' }
& .venv/python.exe -X utf8 scripts/summarize_candidates.py --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Verifier result audit failed' }
