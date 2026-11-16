param([string]$Manifest = 'data\public_v1\manifest.jsonl', [string]$Output = 'runs\public_v1', [int]$PerDataset = 200)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe' }
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
& $runtime -X utf8 -m clickagent.public_benchmark --manifest $Manifest --output $Output --per-dataset $PerDataset
if ($LASTEXITCODE -ne 0) { throw 'Public evaluation failed; committed predictions and features are saved for resume' }
