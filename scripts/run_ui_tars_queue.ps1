$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
while (-not (Test-Path -LiteralPath 'models/UI-TARS-1.5-7B/clickagent_download_verified.json') -or
       -not (Test-Path -LiteralPath 'runs/semantic_target_v2/completion.json')) {
    Start-Sleep -Seconds 5
}
& .venv/python.exe -u -X utf8 scripts/run_ui_tars_baseline.py --split dev --limit 3 *> runs/ui_tars_dev_check.log
if ($LASTEXITCODE -ne 0) { throw 'UI-TARS development interface check failed' }
& .venv/python.exe -u -X utf8 scripts/run_ui_tars_baseline.py *> runs/ui_tars_test_console.log
if ($LASTEXITCODE -ne 0) { throw 'UI-TARS test failed' }
& .venv/python.exe -u -X utf8 scripts/summarize_semantic_target.py *> runs/semantic_target_v2_comparison.log
if ($LASTEXITCODE -ne 0) { throw 'Matched comparison failed' }
