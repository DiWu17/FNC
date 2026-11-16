$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$runPath = Join-Path $projectRoot 'runs/semantic_head_v7'
$pythonPath = (Resolve-Path -LiteralPath '.venv/python.exe').Path

function Set-Stage([string]$name, [string]$status) {
    @{ stage = $name; status = $status; time = (Get-Date -Format 'o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath 'pipeline_status.json') -Encoding utf8
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $status $name" | Add-Content -LiteralPath (Join-Path $runPath 'pipeline.log')
}

function Invoke-Stage([string]$name, [string[]]$pythonArguments) {
    Set-Stage $name 'START'
    $arguments = @('-u', '-X', 'utf8') + $pythonArguments
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath "$name.log") -RedirectStandardError (Join-Path $runPath "$name.stderr.log") -Wait -PassThru
    if ($process.ExitCode -ne 0) { Set-Stage $name 'FAILED'; throw "$name failed ($($process.ExitCode)); inspect its log" }
    Set-Stage $name 'DONE'
}

if (-not (Test-Path -LiteralPath (Join-Path $runPath 'protocol.json'))) { throw 'The experiment protocol must be sealed first.' }
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'features_extra_train_completion.json'))) {
    Invoke-Stage 'extract_extra_train' @('scripts/extract_semantic_head_features.py', '--stage', 'extra_train')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'point_cache_train_completion.json'))) {
    Invoke-Stage 'cache_train' @('scripts/semantic_head_data.py', '--split', 'train', '--device', 'cuda:0')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'point_cache_val_completion.json'))) {
    Invoke-Stage 'cache_val' @('scripts/semantic_head_data.py', '--split', 'val', '--device', 'cuda:0')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'frozen_readouts.json'))) {
    Set-Stage 'train_two_seeds' 'START'
    # Seed 1 initializes the source snapshot; wait for that small file before
    # seed 2 starts, then both train independently on different GPUs.
    $seed1Arguments = @('-u', '-X', 'utf8', 'scripts/train_semantic_head_experiment.py', '--seed', '20261104', '--device', 'cuda:0')
    $seed1 = Start-Process -FilePath $pythonPath -ArgumentList $seed1Arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath 'train_s1.log') -RedirectStandardError (Join-Path $runPath 'train_s1.stderr.log') -PassThru
    # Keep a process handle before polling HasExited; otherwise Windows
    # PowerShell can leave ExitCode null even after a successful worker exits.
    $null = $seed1.Handle
    while (-not (Test-Path -LiteralPath (Join-Path $runPath 'training_identity.json'))) {
        if ($seed1.HasExited) { throw 'Seed 1 stopped before creating the training identity.' }
        Start-Sleep -Seconds 2
    }
    $seed2Arguments = @('-u', '-X', 'utf8', 'scripts/train_semantic_head_experiment.py', '--seed', '20261105', '--device', 'cuda:1')
    $seed2 = Start-Process -FilePath $pythonPath -ArgumentList $seed2Arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runPath 'train_s2.log') -RedirectStandardError (Join-Path $runPath 'train_s2.stderr.log') -PassThru
    $null = $seed2.Handle
    while (-not $seed1.HasExited -or -not $seed2.HasExited) { Start-Sleep -Seconds 5 }
    $seed1.WaitForExit(); $seed2.WaitForExit()
    if ($seed1.ExitCode -ne 0 -or $seed2.ExitCode -ne 0) { Set-Stage 'train_two_seeds' 'FAILED'; throw 'A training worker failed; inspect train_s*.stderr.log.' }
    Set-Stage 'train_two_seeds' 'DONE'
    Invoke-Stage 'freeze' @('scripts/train_semantic_head_experiment.py', '--freeze-only')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'features_test_completion.json'))) {
    Invoke-Stage 'extract_test' @('scripts/extract_semantic_head_features.py', '--stage', 'test')
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'ui_tars/summary.json'))) {
    Invoke-Stage 'ui_tars' @('scripts/run_ui_tars_baseline.py', '--output', 'runs/semantic_head_v7')
}
Invoke-Stage 'audit_ui_tars' @('scripts/audit_ui_tars_baseline.py', '--output', 'runs/semantic_head_v7')
foreach ($cohort in @('diagnostic', 'test')) {
    if (-not (Test-Path -LiteralPath (Join-Path $runPath "point_cache_${cohort}_completion.json"))) {
        Invoke-Stage "cache_$cohort" @('scripts/semantic_head_data.py', '--split', $cohort, '--device', 'cuda:0')
    }
}
foreach ($cohort in @('diagnostic', 'test')) {
    if (-not (Test-Path -LiteralPath (Join-Path $runPath "evaluation/$cohort/summary.json"))) {
        Invoke-Stage "evaluate_$cohort" @('scripts/evaluate_semantic_head_experiment.py', '--split', $cohort)
    }
}
Invoke-Stage 'audit' @('scripts/audit_semantic_head_experiment.py')
Invoke-Stage 'report' @('scripts/report_semantic_head_experiment.py')
Set-Stage 'experiment' 'COMPLETE'
