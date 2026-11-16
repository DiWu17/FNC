$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
if (-not (Test-Path -LiteralPath 'data/conditioned_v3/qwen_weight_identity.json')) {
    & .venv/python.exe -u -X utf8 scripts/pin_conditioned_backbone.py *> runs/conditioned_v3_backbone_hashes.log
    if ($LASTEXITCODE -ne 0) { throw 'Local Qwen weight pinning failed' }
}
while (-not (Test-Path -LiteralPath 'data/conditioned_v3/training_download_verified.json')) {
    Start-Sleep -Seconds 5
}
& .venv/python.exe -u -X utf8 scripts/prepare_conditioned_data.py *> runs/conditioned_v3_prepare.log
if ($LASTEXITCODE -ne 0) { throw 'Training/test data preparation failed' }
& .venv/python.exe -u -X utf8 scripts/extract_conditioned_features.py --output runs/conditioned_v3_pilot --limit 4 *> runs/conditioned_v3_pilot.log
if ($LASTEXITCODE -ne 0) { throw 'Frozen model spatial feature pilot failed' }
