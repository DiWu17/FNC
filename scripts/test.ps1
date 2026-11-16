$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe' }
# Keep pytest temporary files in the workspace; use a fresh path per run.
$testTemp = Join-Path $projectRoot ('.cache\pytest-all-' + [guid]::NewGuid().ToString('N'))
& $runtime -m pytest -q --basetemp $testTemp
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
