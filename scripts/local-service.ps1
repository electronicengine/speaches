param(
    [ValidateSet('start', 'stop', 'status', 'logs')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $composeArgs = @('compose', '-f', 'compose.cuda.yaml', '-f', 'compose.local.yaml')
    switch ($Action) {
        'start' {
            if (Select-String -LiteralPath '.env' -Pattern '^POCKET_TTS_URL=.' -Quiet) {
                & (Join-Path $PSScriptRoot 'pocket-service.ps1') -Action start
            }
            if (Select-String -LiteralPath '.env' -Pattern '^TRENDYOL_TTS_URL=.' -Quiet) {
                & (Join-Path $PSScriptRoot 'pocket-service.ps1') -Action start -Engine trendyol
            }
            & docker @composeArgs up -d --no-build
        }
        'stop' {
            & docker @composeArgs down
            & (Join-Path $PSScriptRoot 'pocket-service.ps1') -Action stop
            & (Join-Path $PSScriptRoot 'pocket-service.ps1') -Action stop -Engine trendyol
        }
        'status' {
            & docker @composeArgs ps
            & (Join-Path $PSScriptRoot 'pocket-service.ps1') -Action status
            & (Join-Path $PSScriptRoot 'pocket-service.ps1') -Action status -Engine trendyol
            $tunnelLogs = & docker @composeArgs logs --no-color --tail 150 tunnel 2>&1
            $urls = [regex]::Matches(($tunnelLogs -join "`n"), 'https://[a-z0-9-]+\.trycloudflare\.com')
            if ($urls.Count -gt 0) {
                Write-Output ("API base URL: " + $urls[$urls.Count - 1].Value + '/v1')
            }
        }
        'logs' { & docker @composeArgs logs --tail 80 }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
