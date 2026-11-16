$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

function Invoke-ExperimentStage {
    param([string]$Name, [string[]]$PythonArguments)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$stamp START $Name" | Add-Content -LiteralPath 'runs/conditioned_v3_pipeline.log'
    & .venv/python.exe -u -X utf8 @PythonArguments *> "runs/conditioned_v3_$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name; see runs/conditioned_v3_$Name.log" }
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$stamp DONE $Name" | Add-Content -LiteralPath 'runs/conditioned_v3_pipeline.log'
}

# Run only after inspecting the separate four-sample real-model pilot.
Invoke-ExperimentStage -Name 'extract_trainval' -PythonArguments @('scripts/extract_conditioned_features.py')
Invoke-ExperimentStage -Name 'audit_trainval' -PythonArguments @('scripts/audit_conditioned_features.py', '--require-complete', 'trainval')
if (-not (Test-Path -LiteralPath 'runs/conditioned_v3/frozen_readouts.json')) {
    Invoke-ExperimentStage -Name 'train' -PythonArguments @('scripts/train_conditioned_readouts.py')
}
Invoke-ExperimentStage -Name 'extract_test' -PythonArguments @('scripts/extract_conditioned_features.py', '--split', 'test')
Invoke-ExperimentStage -Name 'audit_features' -PythonArguments @('scripts/audit_conditioned_features.py', '--require-complete', 'all')
Invoke-ExperimentStage -Name 'evaluate' -PythonArguments @('scripts/evaluate_conditioned_readouts.py')
Invoke-ExperimentStage -Name 'ui_tars' -PythonArguments @('scripts/run_ui_tars_baseline.py', '--output', 'runs/conditioned_v3')
Invoke-ExperimentStage -Name 'audit_ui_tars' -PythonArguments @('scripts/audit_ui_tars_baseline.py', '--output', 'runs/conditioned_v3')
Invoke-ExperimentStage -Name 'report' -PythonArguments @('scripts/report_conditioned_readouts.py')
Invoke-ExperimentStage -Name 'final_audit' -PythonArguments @('scripts/audit_conditioned_experiment.py')
Invoke-ExperimentStage -Name 'plots' -PythonArguments @('scripts/plot_conditioned_results.py')
'COMPLETE conditioned spatial experiment' | Add-Content -LiteralPath 'runs/conditioned_v3_pipeline.log'
