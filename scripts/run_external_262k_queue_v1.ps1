$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'H:\wdProjs\ClickAgent'
$run = 'runs/external_262k_panel_v1'
$python = '.venv\python.exe'
$script = 'scripts/run_external_262k_panel_v1.py'
$stages = @(
    @('qwen','mmb'),
    @('qwen','sspro'),
    @('mvp','mmb'),
    @('mvp','sspro'),
    @('tag','pro'),
    @('tag','sheets'),
    @('tag','mmb'),
    @('tag','sspro')
)

# The first stage was started interactively before this queue was launched.
# Wait for it to exit, then invoke the same resumable command so an interrupted
# stage is completed and verified before advancing.
Wait-Process -Id 22704 -ErrorAction SilentlyContinue
foreach ($pair in $stages) {
    $method, $suite = $pair
    $log = Join-Path $run "queue_${method}_${suite}.log"
    "$(Get-Date -Format o) START $method $suite" | Add-Content -LiteralPath (Join-Path $run 'queue_events.log')
    & $python -B -X utf8 -u $script --stage run --method $method --suite $suite *>> $log
    if ($LASTEXITCODE -ne 0) {
        "$(Get-Date -Format o) FAILED $method $suite exit=$LASTEXITCODE" | Add-Content -LiteralPath (Join-Path $run 'queue_events.log')
        throw "External baseline stage failed: $method $suite"
    }
    "$(Get-Date -Format o) COMPLETE $method $suite" | Add-Content -LiteralPath (Join-Path $run 'queue_events.log')
}
& $python -B -X utf8 -u $script --stage verify-gate *>> (Join-Path $run 'queue_gate.log')
if ($LASTEXITCODE -ne 0) { throw 'External baseline public seal failed' }
"$(Get-Date -Format o) PUBLIC_SEALED" | Add-Content -LiteralPath (Join-Path $run 'queue_events.log')
