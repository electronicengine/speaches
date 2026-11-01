param(
    [ValidateSet('start', 'stop', 'status')][string]$Action = 'status',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [ValidateSet('pocket', 'trendyol')][string]$Engine = 'pocket'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$runtime = Join-Path $projectRoot ('.' + $Engine + '-runtime')
$python = Join-Path $projectRoot '.pocket-venv/Scripts/python.exe'
$script = Join-Path $PSScriptRoot ($Engine + '_tts_server.py')
$port = if ($Engine -eq 'trendyol') { 8011 } else { 8010 }
$pidFile = Join-Path $runtime 'worker.pid'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$worker = $null
if (Test-Path -LiteralPath $pidFile) {
    $workerId = [int](Get-Content -LiteralPath $pidFile -Raw)
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$workerId"
    if ($candidate -and $candidate.CommandLine.Contains($script) -and $candidate.ExecutablePath -eq $python) {
        $worker = $candidate
    }
}
if ($Action -eq 'stop') {
    if ($worker) {
        # Windows venv launchers run the interpreter in a child process.
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$($worker.ProcessId)" |
            Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine.Contains($script) } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
        Stop-Process -Id $worker.ProcessId -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    Write-Output 'Pocket TTS stopped'
    return
}
if ($Action -eq 'start' -and -not $worker) {
    if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup-pocket.ps1 first' }
    $process = Start-Process -FilePath $python -ArgumentList @('"' + $script + '"', '--device', $Device) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'stdout.log') -RedirectStandardError (Join-Path $runtime 'stderr.log') -PassThru
    $process.Id | Set-Content -LiteralPath $pidFile
    $worker = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
}
if (-not $worker) { Write-Output 'Pocket TTS is not running'; return }
$keyLine = Get-Content (Join-Path $projectRoot '.env') | Where-Object { $_ -match '^API_KEY=.' } | Select-Object -First 1
$headers = @{ Authorization = 'Bearer ' + $keyLine.Substring(8) }
$attempts = if ($Action -eq 'start') { 180 } else { 1 }
for ($attempt = 0; $attempt -lt $attempts; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri ("http://127.0.0.1:$port/health") -Headers $headers -TimeoutSec 2
        $modelName = if ($health.model) { $health.model } elseif ($Engine -eq 'pocket') { 'wite-tech/pocket-tts-turkish' } else { 'Trendyol/Trendyol-TTS' }
        Write-Output ($modelName + ' ready: ' + $health.device + '; voices: ' + $health.voices.Count + '; PID ' + $worker.ProcessId)
        return
    } catch {
        if ($Action -eq 'status') { Write-Output 'Pocket TTS is starting or unavailable'; return }
        if ($process.HasExited) { throw 'Pocket TTS exited; inspect .pocket-runtime/stderr.log' }
        Start-Sleep -Seconds 1
    }
}
throw 'Pocket TTS did not become ready; inspect .pocket-runtime/stderr.log'
