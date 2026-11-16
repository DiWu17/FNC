$ErrorActionPreference = 'Stop'
$v11ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $v11ProjectRoot
$v11Python = Join-Path $v11ProjectRoot '.venv/python.exe'
$v11RunRoot = Join-Path $v11ProjectRoot 'runs/candidate_scorer_v11'
$script:v11Children = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

function Write-V11Stage([string]$Stage, [string]$Status = 'running') {
    @{stage=$Stage; status=$Status; updated_utc=[DateTime]::UtcNow.ToString('o'); pid=$PID} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $v11RunRoot 'experiment_progress.json') -Encoding utf8
}

function Start-V11Python([string]$LogName, [string[]]$PythonArguments) {
    $child = Start-Process -FilePath $v11Python -ArgumentList (@('-u') + $PythonArguments) -WorkingDirectory $v11ProjectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $v11RunRoot ($LogName + '.log')) `
        -RedirectStandardError (Join-Path $v11RunRoot ($LogName + '.err'))
    $script:v11Children.Add($child)
    @{name=$LogName; pid=$child.Id; arguments=$PythonArguments; started_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $v11RunRoot 'processes.jsonl') -Encoding utf8
    return $child
}

function Complete-V11Process($Child, [string]$Name) {
    $Child.WaitForExit()
    $Child.Refresh()
    if ($Child.ExitCode -ne 0) { throw "V11 $Name failed with exit $($Child.ExitCode); inspect its log/err" }
}

function Invoke-V11Python([string]$LogName, [string[]]$PythonArguments) {
    Write-V11Stage $LogName
    $child = Start-V11Python $LogName $PythonArguments
    Complete-V11Process $child $LogName
}

try {
    Invoke-V11Python 'local_trainval' @('scripts/extract_candidate_scorer.py','--stage','local_trainval')
    Invoke-V11Python 'cache_internal_train' @('scripts/cache_candidate_scorer.py','--stage','internal','--split','train')
    Invoke-V11Python 'cache_internal_val' @('scripts/cache_candidate_scorer.py','--stage','internal','--split','val')
    Write-V11Stage 'training_and_extra_visual_cache'
    $v11Seed1 = Start-V11Python 'train_s1' @('scripts/train_candidate_scorer_worker.py','--seed','s1','--device','cuda:0')
    $v11Seed2 = Start-V11Python 'visual_cache_and_train_s2' @('scripts/cache_visual_trainval.py')
    Complete-V11Process $v11Seed2 'visual_cache_and_train_s2'
    Complete-V11Process $v11Seed1 'train_s1'
    Invoke-V11Python 'freeze' @('scripts/candidate_scorer_protocol.py','--stage','freeze')
    Invoke-V11Python 'fresh_test_features' @('scripts/extract_candidate_scorer.py','--stage','test')
    Invoke-V11Python 'cache_internal_test' @('scripts/cache_candidate_scorer.py','--stage','internal','--split','test')
    Write-V11Stage 'fresh_test_visual_and_reference'
    $v11VisualTest = Start-V11Python 'cache_visual_test' @('scripts/cache_candidate_scorer.py','--stage','visual','--split','test','--device','cuda:1')
    $v11Reference = Start-V11Python 'reference_test' @('scripts/run_candidate_scorer.py','--stage','reference','--split','test')
    Complete-V11Process $v11VisualTest 'cache_visual_test'
    Complete-V11Process $v11Reference 'reference_test'
    Write-V11Stage 'fresh_test_heads_and_ui_tars'
    $v11Heads = Start-V11Python 'heads_test' @('scripts/run_candidate_scorer.py','--stage','heads','--split','test')
    $v11UITars = Start-V11Python 'ui_tars_test' @('scripts/run_ui_tars_baseline.py','--output','runs/candidate_scorer_v11')
    Complete-V11Process $v11Heads 'heads_test'
    Complete-V11Process $v11UITars 'ui_tars_test'
    Invoke-V11Python 'diagnostic_pools' @('scripts/extract_candidate_scorer.py','--stage','diagnostic')
    Invoke-V11Python 'cache_internal_diagnostic' @('scripts/cache_candidate_scorer.py','--stage','internal','--split','diagnostic')
    Invoke-V11Python 'cache_visual_diagnostic' @('scripts/cache_candidate_scorer.py','--stage','visual','--split','diagnostic','--device','cuda:1')
    Invoke-V11Python 'heads_diagnostic' @('scripts/run_candidate_scorer.py','--stage','heads','--split','diagnostic')
    Invoke-V11Python 'reference_diagnostic' @('scripts/run_candidate_scorer.py','--stage','reference','--split','diagnostic')
    Invoke-V11Python 'evaluate' @('scripts/evaluate_candidate_scorer.py')
    Invoke-V11Python 'audit' @('scripts/audit_candidate_scorer.py')
    Invoke-V11Python 'report' @('scripts/report_candidate_scorer.py')
    Write-V11Stage 'complete' 'complete'
} catch {
    foreach ($child in $script:v11Children) {
        if (-not $child.HasExited) { Stop-Process -Id $child.Id -ErrorAction SilentlyContinue }
    }
    Write-V11Stage 'failed' 'failed'
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $v11RunRoot 'experiment_failure.txt') -Encoding utf8
    throw
}
