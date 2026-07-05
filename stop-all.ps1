<#
.SYNOPSIS
    Stop all ECG Anomaly Detection services running on the standard ports.

.DESCRIPTION
    Kills any process listening on ports 3000 (frontend), 8000 (backend),
    and 3003 (WebSocket service). Useful when service windows are closed
    but the underlying processes are still running.

.EXAMPLE
    .\stop-all.ps1
#>

$ports = @(3000, 3003, 8000)
$ErrorActionPreference = "SilentlyContinue"

Write-Host ""
Write-Host "Stopping ECG Anomaly Detection services..." -ForegroundColor Cyan

foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "  Stopping $($proc.ProcessName) (PID $($proc.Id)) on port $port" -ForegroundColor Yellow
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

# Also kill any uvicorn / bun / next processes spawned by our launcher
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    try { $_.CommandLine -match "uvicorn" } catch { $false }
} | ForEach-Object {
    Write-Host "  Stopping uvicorn process (PID $($_.Id))" -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Done. All services stopped." -ForegroundColor Green
Write-Host ""
