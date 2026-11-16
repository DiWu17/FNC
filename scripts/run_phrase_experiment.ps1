$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

function Invoke-PhraseStage {
    param([string]$StageName, [string[]]$PythonArguments)
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START $StageName" | Add-Content -LiteralPath 'runs/phrase_v4/pipeline.log'
    # Windows PowerShell 5 treats harmless native stderr (e.g. tqdm) as a
    # terminating NativeCommandError under ErrorActionPreference=Stop.
    # Separate streams and trust the process exit code instead.
    $phrasePython = (Resolve-Path -LiteralPath '.venv/python.exe').Path
    $phraseArguments = @('-u', '-X', 'utf8') + $PythonArguments
    $phraseProcess = Start-Process -FilePath $phrasePython -ArgumentList $phraseArguments -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput "runs/phrase_v4/$StageName.log" -RedirectStandardError "runs/phrase_v4/$StageName.stderr.log" -Wait -PassThru
    if ($phraseProcess.ExitCode -ne 0) { throw "Phrase experiment stage failed: $StageName; exit $($phraseProcess.ExitCode)" }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') DONE $StageName" | Add-Content -LiteralPath 'runs/phrase_v4/pipeline.log'
}

if (-not (Test-Path -LiteralPath 'runs/phrase_v4/frozen_readouts.json')) {
    throw 'Finish and inspect training before starting the sealed test.'
}
if (-not (Test-Path -LiteralPath 'runs/phrase_v4/backbone_pin_complete.txt')) {
    Invoke-PhraseStage -StageName 'pin_backbone' -PythonArguments @('scripts/pin_conditioned_backbone.py')
    'verified' | Set-Content -LiteralPath 'runs/phrase_v4/backbone_pin_complete.txt'
}
Invoke-PhraseStage -StageName 'extract_test' -PythonArguments @('scripts/extract_conditioned_features.py', '--output', 'runs/phrase_v4', '--manifest', 'runs/phrase_v4/manifest.jsonl', '--split', 'test')
Invoke-PhraseStage -StageName 'ui_tars' -PythonArguments @('scripts/run_ui_tars_baseline.py', '--output', 'runs/phrase_v4')
Invoke-PhraseStage -StageName 'audit_ui_tars' -PythonArguments @('scripts/audit_ui_tars_baseline.py', '--output', 'runs/phrase_v4')
Invoke-PhraseStage -StageName 'evaluate' -PythonArguments @('scripts/evaluate_phrase_readouts.py')
Invoke-PhraseStage -StageName 'cli_smoke' -PythonArguments @('scripts/audit_phrase_cli.py')
Invoke-PhraseStage -StageName 'audit' -PythonArguments @('scripts/audit_phrase_experiment.py')
Invoke-PhraseStage -StageName 'report' -PythonArguments @('scripts/report_phrase_experiment.py')
'COMPLETE phrase experiment' | Add-Content -LiteralPath 'runs/phrase_v4/pipeline.log'
