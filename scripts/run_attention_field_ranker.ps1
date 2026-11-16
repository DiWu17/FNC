$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv/python.exe'
$runPath = Join-Path $projectRoot 'runs/attention_field_v6'

function Invoke-ExperimentStage([string]$stageName, [string[]]$pythonArguments) {
    Write-Output "Running $stageName"
    & $pythonPath @pythonArguments *> (Join-Path $runPath "$stageName.log")
    if ($LASTEXITCODE -ne 0) { throw "$stageName failed; see runs/attention_field_v6/$stageName.log" }
}

if (-not (Test-Path -LiteralPath (Join-Path $runPath 'protocol.json'))) {
    throw 'The sealed runs/attention_field_v6/protocol.json is required.'
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'fields_trainval_completion.json'))) {
    Invoke-ExperimentStage 'cache_trainval' @('scripts/attention_field_data.py', '--split', 'trainval')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'frozen_readouts.json'))) {
    Invoke-ExperimentStage 'train' @('scripts/train_attention_field_ranker.py')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'fields_test_completion.json'))) {
    Invoke-ExperimentStage 'cache_test' @('scripts/attention_field_data.py', '--split', 'test')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'evaluation/summary.json'))) {
    Invoke-ExperimentStage 'evaluate' @('scripts/evaluate_attention_field_ranker.py')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'evaluation/cli_smoke.json'))) {
    Invoke-ExperimentStage 'cli_smoke' @('scripts/audit_attention_field_cli.py')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'final_audit.json'))) {
    Invoke-ExperimentStage 'audit' @('scripts/audit_attention_field_ranker.py')
}
Invoke-ExperimentStage 'report' @('scripts/report_attention_field_ranker.py')
Write-Output 'COMPLETE attention field experiment; diagnostic reuse of v5 test.'
