param([string]$DataRoot = 'data\public_v1', [string]$OutputRoot = 'runs\public_v1', [switch]$WaitForV2, [string[]]$DatasetOrder = @())
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
if ($WaitForV2) {
    $prior = Join-Path $OutputRoot 'screenspot_v2\completion.json'
    while (-not (Test-Path -LiteralPath $prior)) { Start-Sleep -Seconds 10 }
    $state = Get-Content -LiteralPath $prior -Raw | ConvertFrom-Json
    if (-not $state.complete) { throw 'ScreenSpot-v2 has not completed' }
}
$datasets = if ($DatasetOrder.Count -gt 0) { $DatasetOrder } elseif ($WaitForV2) { @('osworld_g', 'screenspot_pro', 'venusbench_gd', 'conflictgui') } else { @('screenspot_v2', 'osworld_g', 'screenspot_pro', 'venusbench_gd', 'conflictgui') }
foreach ($name in $datasets) {
    if ($name -notin @('screenspot_v2', 'osworld_g', 'screenspot_pro', 'venusbench_gd', 'conflictgui')) { throw "Unknown dataset: $name" }
}
foreach ($dataset in $datasets) {
    $manifest = Join-Path $DataRoot "manifests\$dataset.jsonl"
    $auditPath = Join-Path $DataRoot "audits\$dataset.json"
    $deadline = (Get-Date).AddHours(3)
    while ($true) {
        $ready = $false
        if ((Test-Path -LiteralPath $manifest) -and (Test-Path -LiteralPath $auditPath)) {
            $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
            $ready = ($audit.samples -gt 0) -and ($audit.images.ok -eq $audit.samples)
        }
        if ($ready) { break }
        if ((Get-Date) -gt $deadline) { throw "Data readiness timed out: $dataset" }
        Start-Sleep -Seconds 10
    }
    $output = Join-Path $OutputRoot $dataset
    $console = Join-Path $OutputRoot "$dataset.console.log"
    Write-Output "Starting fixed 200-case cohort: $dataset"
    & $runtime -X utf8 -m clickagent.public_benchmark --manifest $manifest --output $output --per-dataset 200 --bootstrap-iterations 1000 *> $console
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $dataset; see $console" }
    Write-Output "Completed: $dataset"
}
