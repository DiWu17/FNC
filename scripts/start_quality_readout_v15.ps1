$ErrorActionPreference = 'Stop'
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runPath = Join-Path $workspacePath 'runs\quality_readout_v15'
$launchPath = Join-Path $runPath 'launch.json'
$logPath = Join-Path $runPath 'logs'
New-Item -ItemType Directory -Force -Path $logPath | Out-Null

if (Test-Path -LiteralPath $launchPath) {
    $previousLaunch = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
    $previousProcess = Get-Process -Id $previousLaunch.pid -ErrorAction SilentlyContinue
    if ($null -ne $previousProcess) {
        throw "A recorded pipeline process still exists (PID $($previousLaunch.pid)); inspect its status before resuming."
    }
}

# Preserve previous logs when explicitly restarting a completed or failed job.
$attemptStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($logName in @('pipeline.log', 'pipeline.stderr.log')) {
    $existingLog = Join-Path $logPath $logName
    if (Test-Path -LiteralPath $existingLog) {
        Move-Item -LiteralPath $existingLog -Destination (Join-Path $logPath "$attemptStamp.$logName")
    }
}
if (Test-Path -LiteralPath $launchPath) {
    Copy-Item -LiteralPath $launchPath -Destination (Join-Path $logPath "$attemptStamp.launch.json")
}

$interpreterPath = Join-Path $workspacePath '.venv\python.exe'
$pipelineProcess = Start-Process -FilePath $interpreterPath `
    -ArgumentList @('-B', '-X', 'utf8', '-u', 'scripts/run_quality_readout_v15.py', '--stage', 'all') `
    -WorkingDirectory $workspacePath -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logPath 'pipeline.log') `
    -RedirectStandardError (Join-Path $logPath 'pipeline.stderr.log')
$launchRecord = [ordered]@{
    pid = $pipelineProcess.Id
    started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    command = '.venv\python.exe -B -X utf8 -u scripts/run_quality_readout_v15.py --stage all'
    protocol_sha256 = (Get-FileHash -LiteralPath (Join-Path $runPath 'protocol.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    runs_without_further_prompts = $true
    sequence = @('old-data matched-step controls', 'all new train feature extraction', 'six expanded-data models', 'freeze all models', 'new reserve feature extraction', 'old/new development evaluation')
    stdout = (Join-Path $logPath 'pipeline.log')
    stderr = (Join-Path $logPath 'pipeline.stderr.log')
}
$launchRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding utf8
$launchRecord | ConvertTo-Json -Depth 5

# Keep this tool-managed shell alive while the worker runs. The execution host
# cleans up detached descendants when a short-lived launching shell exits.
$pipelineProcess.WaitForExit()
$exitRecord = [ordered]@{
    pid = $pipelineProcess.Id
    exit_code = $pipelineProcess.ExitCode
    exited_at_utc = (Get-Date).ToUniversalTime().ToString('o')
}
$exitRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath 'process_exit.json') -Encoding utf8
exit $pipelineProcess.ExitCode
