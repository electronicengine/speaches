param([string]$TorchRuntime = 'H:/workspace/tts-test/.venv')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$environment = Join-Path $projectRoot '.pocket-venv'
$packages = Join-Path $TorchRuntime 'Lib/site-packages'
if (-not (Test-Path -LiteralPath (Join-Path $packages 'torch'))) { throw 'TorchRuntime must contain CUDA-enabled PyTorch' }
if (-not (Test-Path -LiteralPath $environment)) {
    & python -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Pocket TTS environment' }
}
[IO.File]::WriteAllText((Join-Path $environment 'Lib/site-packages/tts-runtime.pth'), $packages.Replace('\', '/') + "`n")
& (Join-Path $environment 'Scripts/python.exe') -m pip install -r (Join-Path $PSScriptRoot 'requirements-pocket.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot install Pocket TTS dependencies' }
