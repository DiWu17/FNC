$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $experimentRoot
$experimentPython = Join-Path $experimentRoot '.venv/python.exe'
function Invoke-V8Python([string[]]$PythonArguments) {
    & $experimentPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "V8 Python failed: $PythonArguments" }
}
Invoke-V8Python @('-u','scripts/prepare_point_aligned.py')
Invoke-V8Python @('-u','scripts/seal_point_aligned.py')
$experimentWorkers = @()
foreach ($seedSpec in @(@('s1','cuda:0'),@('s2','cuda:1'))) {
    $worker = Start-Process -FilePath $experimentPython -ArgumentList @('-u','scripts/train_point_aligned.py','--suffix',$seedSpec[0],'--device',$seedSpec[1]) -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentRoot "runs/point_aligned_v8/train_$($seedSpec[0]).log") -RedirectStandardError (Join-Path $experimentRoot "runs/point_aligned_v8/train_$($seedSpec[0]).err") -PassThru
    $null = $worker.Handle
    $experimentWorkers += $worker
}
foreach ($worker in $experimentWorkers) {
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw "Training process $($worker.Id) failed: $($worker.ExitCode)" }
}
Invoke-V8Python @('-u','scripts/train_point_aligned.py','--freeze')
Invoke-V8Python @('-u','scripts/run_point_aligned_inference.py','--stage','smoke')
Invoke-V8Python @('-u','scripts/run_point_aligned_inference.py','--stage','seal')
Invoke-V8Python @('-u','scripts/run_point_aligned_inference.py','--stage','test')
Invoke-V8Python @('-u','scripts/run_ui_tars_baseline.py','--output','runs/point_aligned_v8')
Invoke-V8Python @('-u','scripts/evaluate_point_aligned.py')
Invoke-V8Python @('-u','scripts/diagnose_point_aligned_routing.py')
Invoke-V8Python @('-u','scripts/audit_ui_tars_baseline.py','--output','runs/point_aligned_v8')
Invoke-V8Python @('-u','scripts/audit_point_aligned.py')
Invoke-V8Python @('-u','scripts/report_point_aligned.py')
