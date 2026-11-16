$ErrorActionPreference = 'Stop'
$budgetProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $budgetProjectRoot
$budgetPython = Join-Path $budgetProjectRoot '.venv/python.exe'
function Invoke-BudgetPython([string[]]$BudgetArguments) {
    & $budgetPython @BudgetArguments
    if ($LASTEXITCODE -ne 0) { throw "Candidate-budget command failed: $BudgetArguments" }
}
Invoke-BudgetPython @('-u','scripts/run_candidate_budget.py','--stage','prepare')
Invoke-BudgetPython @('-u','scripts/run_candidate_budget.py','--stage','infer')
Invoke-BudgetPython @('-u','scripts/evaluate_candidate_budget.py')
Invoke-BudgetPython @('-u','scripts/audit_candidate_budget.py')
Invoke-BudgetPython @('-u','scripts/report_candidate_budget.py')
