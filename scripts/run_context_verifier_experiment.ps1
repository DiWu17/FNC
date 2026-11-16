$ErrorActionPreference = 'Stop'
$contextExperimentRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $contextExperimentRoot
$contextPython = Join-Path $contextExperimentRoot '.venv/python.exe'
function Invoke-ContextPython([string[]]$ContextArguments) {
    & $contextPython @ContextArguments
    if ($LASTEXITCODE -ne 0) { throw "V9 command failed: $ContextArguments" }
}
Invoke-ContextPython @('-u','scripts/prepare_context_verifier.py')
Invoke-ContextPython @('-u','scripts/seal_context_verifier.py','--stage','protocol')
if (-not (Test-Path -LiteralPath 'runs/context_verifier_v9/critic_cache_val.json')) {
    Invoke-ContextPython @('-u','scripts/train_context_point_critic.py','--stage','cache')
}
if (-not ((Test-Path -LiteralPath 'runs/context_verifier_v9/models/point_critic_s1.json') -and (Test-Path -LiteralPath 'runs/context_verifier_v9/models/point_critic_s2.json'))) {
    Invoke-ContextPython @('-u','scripts/train_context_point_critic.py','--stage','train')
}
Invoke-ContextPython @('-u','scripts/run_context_verifier.py','--stage','validation')
Invoke-ContextPython @('-u','scripts/run_gui_verifier.py','--stage','validation')
Invoke-ContextPython @('-u','scripts/evaluate_context_verifier.py','--stage','validation')
Invoke-ContextPython @('-u','scripts/audit_context_verifier.py','--stage','validation')
Invoke-ContextPython @('-u','scripts/seal_context_verifier.py','--stage','freeze')
Invoke-ContextPython @('-u','scripts/run_context_verifier.py','--stage','test')
Invoke-ContextPython @('-u','scripts/run_gui_verifier.py','--stage','test')
Invoke-ContextPython @('-u','scripts/run_ui_tars_baseline.py','--output','runs/context_verifier_v9')
Invoke-ContextPython @('-u','scripts/evaluate_context_verifier.py')
Invoke-ContextPython @('-u','scripts/audit_ui_tars_baseline.py','--output','runs/context_verifier_v9')
Invoke-ContextPython @('-u','scripts/audit_context_verifier.py')
Invoke-ContextPython @('-u','scripts/report_context_verifier.py')
