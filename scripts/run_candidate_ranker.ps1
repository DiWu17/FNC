$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

function Invoke-RankerStage {
    param([string]$StageName, [string[]]$PythonArguments)
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $StageName" | Add-Content -LiteralPath 'runs/candidate_ranker_v5/pipeline.log'
    $rankerPython = (Resolve-Path -LiteralPath '.venv/python.exe').Path
    $rankerArguments = @('-u', '-X', 'utf8') + $PythonArguments
    $rankerProcess = Start-Process -FilePath $rankerPython -ArgumentList $rankerArguments -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput "runs/candidate_ranker_v5/$StageName.log" -RedirectStandardError "runs/candidate_ranker_v5/$StageName.stderr.log" -Wait -PassThru
    if ($rankerProcess.ExitCode -ne 0) { throw "Candidate ranker stage failed: $StageName; exit $($rankerProcess.ExitCode)" }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') DONE $StageName" | Add-Content -LiteralPath 'runs/candidate_ranker_v5/pipeline.log'
}

while (-not (Test-Path -LiteralPath 'runs/candidate_ranker_v5/candidates_trainval_completion.json')) { Start-Sleep -Seconds 5 }
if (-not (Test-Path -LiteralPath 'runs/candidate_ranker_v5/frozen_readouts.json')) {
    Invoke-RankerStage -StageName 'train' -PythonArguments @('scripts/train_candidate_ranker.py')
}
Invoke-RankerStage -StageName 'pin_backbone' -PythonArguments @('scripts/pin_conditioned_backbone.py')
Invoke-RankerStage -StageName 'extract_test' -PythonArguments @('scripts/extract_conditioned_features.py', '--output', 'runs/candidate_ranker_v5', '--manifest', 'runs/candidate_ranker_v5/manifest.jsonl', '--split', 'test')
Invoke-RankerStage -StageName 'cache_test' -PythonArguments @('scripts/ranker_data.py', '--split', 'test')
Invoke-RankerStage -StageName 'ui_tars' -PythonArguments @('scripts/run_ui_tars_baseline.py', '--output', 'runs/candidate_ranker_v5')
Invoke-RankerStage -StageName 'audit_ui_tars' -PythonArguments @('scripts/audit_ui_tars_baseline.py', '--output', 'runs/candidate_ranker_v5')
Invoke-RankerStage -StageName 'evaluate' -PythonArguments @('scripts/evaluate_candidate_ranker.py')
Invoke-RankerStage -StageName 'audit' -PythonArguments @('scripts/audit_candidate_ranker.py')
Invoke-RankerStage -StageName 'report' -PythonArguments @('scripts/report_candidate_ranker.py')
'COMPLETE candidate ranker experiment' | Add-Content -LiteralPath 'runs/candidate_ranker_v5/pipeline.log'
