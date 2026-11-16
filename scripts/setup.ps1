param([string]$Python = 'python', [switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $runtime)) {
    & $Python -c "import sys; assert (3,10)<=sys.version_info[:2]<(3,13), 'Provide Python 3.10-3.12 using -Python PATH (3.11 recommended)'"
    if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
    & $Python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
$torchIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu118' }
& $runtime -m pip install torch==2.7.1 torchvision==0.22.1 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $runtime -m pip install -e '.[model,dev]' --index-url https://mirrors.aliyun.com/pypi/simple
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed' }
& $runtime -m clickagent doctor

