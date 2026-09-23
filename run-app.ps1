<#
.SYNOPSIS
    Single launcher for the ECG research application.

.DESCRIPTION
    Starts, stops, restarts, or reports status for the local ECG stack:
    - Next.js frontend
    - FastAPI backend
    - Socket.io WebSocket service

    The script keeps all runtime logs under runtime/logs and writes process
    metadata to runtime/pids.json. It never stops unrelated services unless
    their command line belongs to this project directory.

.EXAMPLE
    .\run-app.ps1

.EXAMPLE
    .\run-app.ps1 -Restart

.EXAMPLE
    .\run-app.ps1 -Stop
#>

param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status,
    [switch]$SkipSetup,
    [switch]$ForceSetup,
    [switch]$UsePostgres,
    [switch]$Open,
    [switch]$HiddenServices,
    [int]$FrontendPort = 3000,
    [int]$BackendPort = 8000,
    [int]$WsPort = 3003
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$WsServiceDir = Join-Path $ProjectRoot "mini-services\ws-service"
$VenvDir = Join-Path $BackendDir "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"
$BackendEnvFile = Join-Path $BackendDir ".env"
$RuntimeDir = Join-Path $ProjectRoot "runtime"
$LogDir = Join-Path $RuntimeDir "logs"
$PidFile = Join-Path $RuntimeDir "pids.json"

function Write-Header([string]$Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Write-Step([string]$Message) {
    Write-Host "-> $Message" -ForegroundColor Yellow
}

function Write-Ok([string]$Message) {
    Write-Host "OK $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "WARN $Message" -ForegroundColor DarkYellow
}

function Test-Command([string]$Command) {
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

function Test-PortListening([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Get-ProcessCommandLine([int]$ProcessId) {
    try {
        return (Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop).CommandLine
    } catch {
        return ""
    }
}

function Test-ProjectProcess([int]$ProcessId) {
    $cmd = Get-ProcessCommandLine $ProcessId
    return ($cmd -like "*$ProjectRoot*")
}

function Invoke-External {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $ProjectRoot,
        [string]$FailureMessage = "Command failed"
    )

    $old = Get-Location
    try {
        Set-Location $WorkingDirectory
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "$FailureMessage (exit code $LASTEXITCODE)"
        }
    } finally {
        Set-Location $old
    }
}

function ConvertTo-PowerShellSingleQuotedLiteral {
    param([AllowEmptyString()][string]$Value)
    return "'$(($Value -replace "'", "''"))'"
}

function Start-ServiceProcess {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory=$true)][string]$WorkingDirectory,
        [Parameter(Mandatory=$true)][string]$LogBaseName,
        [hashtable]$Environment = @{}
    )

    $logSuffix = if ($HiddenServices) {
        "out.log"
    } else {
        "live.$(Get-Date -Format 'yyyyMMdd-HHmmss-fff').log"
    }
    $outLog = Join-Path $LogDir "$LogBaseName.$logSuffix"
    $errLog = Join-Path $LogDir "$LogBaseName.err.log"

    if ($HiddenServices) {
        return Start-Process -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $WorkingDirectory `
            -WindowStyle Hidden `
            -RedirectStandardOutput $outLog `
            -RedirectStandardError $errLog `
            -PassThru
    }

    $titleLiteral = ConvertTo-PowerShellSingleQuotedLiteral "ECGdoc - $Name"
    $workingDirectoryLiteral = ConvertTo-PowerShellSingleQuotedLiteral $WorkingDirectory
    $filePathLiteral = ConvertTo-PowerShellSingleQuotedLiteral $FilePath
    $outLogLiteral = ConvertTo-PowerShellSingleQuotedLiteral $outLog
    $argumentLiterals = @($ArgumentList | ForEach-Object {
        ConvertTo-PowerShellSingleQuotedLiteral ([string]$_)
    })
    $argumentExpression = if ($argumentLiterals.Count -gt 0) {
        $argumentLiterals -join ', '
    } else {
        ''
    }
    $environmentLines = @($Environment.GetEnumerator() | ForEach-Object {
        $nameLiteral = ConvertTo-PowerShellSingleQuotedLiteral ([string]$_.Key)
        $valueLiteral = ConvertTo-PowerShellSingleQuotedLiteral ([string]$_.Value)
        "Set-Item -LiteralPath ('Env:' + $nameLiteral) -Value $valueLiteral"
    }) -join "`r`n"

    $serviceCommand = @"
try { `$Host.UI.RawUI.WindowTitle = $titleLiteral } catch {}
Set-Location -LiteralPath $workingDirectoryLiteral
$environmentLines
Write-Host 'ECGdoc $Name logs' -ForegroundColor Cyan
Write-Host ('Log file: ' + $outLogLiteral) -ForegroundColor DarkGray
& $filePathLiteral @($argumentExpression) 2>&1 | Tee-Object -FilePath $outLogLiteral
`$serviceExitCode = `$LASTEXITCODE
Write-Host ''
if (`$serviceExitCode -eq 0) {
    Write-Host 'Service stopped.' -ForegroundColor Yellow
} else {
    Write-Host ('Service exited with code ' + `$serviceExitCode + '.') -ForegroundColor Red
}
"@
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($serviceCommand))

    return Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCommand) `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Normal `
        -PassThru
}

function Get-EnvValue {
    param([string]$Path, [string]$Name, [string]$Default = "")
    if (-not (Test-Path $Path)) { return $Default }
    $line = Get-Content $Path | Where-Object { $_ -match "^\s*$Name\s*=" } | Select-Object -First 1
    if (-not $line) { return $Default }
    return (($line -replace "^\s*$Name\s*=\s*", "").Trim().Trim('"').Trim("'"))
}

function Set-EnvValue {
    param([string]$Path, [string]$Name, [string]$Value)
    $escaped = [regex]::Escape($Name)
    $replacement = "$Name=`"$Value`""
    if (Test-Path $Path) {
        $content = Get-Content $Path
        if ($content | Where-Object { $_ -match "^\s*$escaped\s*=" }) {
            $content = $content | ForEach-Object {
                if ($_ -match "^\s*$escaped\s*=") { $replacement } else { $_ }
            }
            $content | Set-Content $Path -Encoding ASCII
        } else {
            Add-Content -Path $Path -Value $replacement -Encoding ASCII
        }
    } else {
        $replacement | Set-Content $Path -Encoding ASCII
    }
}

function Initialize-EnvFiles {
    $databaseUrl = if ($UsePostgres) {
        "postgresql://ecg_user:ecg_password@localhost:5432/ecg_anomaly"
    } else {
        "file:./dev.db"
    }

    if (-not (Test-Path $EnvFile)) {
        Set-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Value $databaseUrl
    } elseif ($UsePostgres -or -not (Get-EnvValue -Path $EnvFile -Name "DATABASE_URL")) {
        Set-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Value $databaseUrl
    }

    $effectiveDatabaseUrl = Get-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Default $databaseUrl
    Set-EnvValue -Path $BackendEnvFile -Name "DATABASE_URL" -Value $effectiveDatabaseUrl

    $schemaPath = Join-Path $ProjectRoot "prisma\schema.prisma"
    $provider = if ($UsePostgres) { "postgresql" } else { "sqlite" }
    $schema = Get-Content $schemaPath
    $updated = $schema -replace '^(\s*)provider = "(sqlite|postgresql)"', "`${1}provider = `"$provider`""
    if (($updated -join "`n") -ne ($schema -join "`n")) {
        $updated | Set-Content $schemaPath -Encoding ASCII
    }
}

function Get-PythonLauncher {
    if (Test-Command "py") {
        try {
            & py -3.13 --version *> $null
            if ($LASTEXITCODE -eq 0) { return @("py", @("-3.13")) }
        } catch {}
    }
    if (Test-Command "python") { return @("python", @()) }
    throw "Python was not found. Install Python 3.10 or newer."
}

function Test-VenvUsable {
    if (-not (Test-Path $VenvPython)) { return $false }
    try {
        & $VenvPython -c "import sys; print(sys.version)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Ensure-Setup {
    New-Item -ItemType Directory -Force -Path $RuntimeDir, $LogDir | Out-Null
    Initialize-EnvFiles

    $needsSetup = $ForceSetup
    if (-not (Test-VenvUsable)) { $needsSetup = $true }
    if (-not (Test-Path (Join-Path $ProjectRoot "node_modules\next\package.json"))) { $needsSetup = $true }
    if (-not (Test-Path (Join-Path $WsServiceDir "node_modules\tsx\package.json"))) { $needsSetup = $true }
    if (-not (Test-Path (Join-Path $ProjectRoot "prisma\dev.db")) -and -not $UsePostgres) { $needsSetup = $true }

    if ($SkipSetup -and -not $ForceSetup) {
        Write-Ok "Setup skipped"
        return
    }
    if (-not $needsSetup) {
        Write-Ok "Dependencies and database already look ready"
        return
    }

    Write-Header "Setup"

    if (-not (Test-VenvUsable) -or $ForceSetup) {
        Write-Step "Creating backend virtual environment"
        $launcher = Get-PythonLauncher
        Invoke-External -FilePath $launcher[0] -ArgumentList ($launcher[1] + @("-m", "venv", "--clear", $VenvDir)) -FailureMessage "Virtual environment creation failed"
    }

    Write-Step "Installing backend dependencies"
    Invoke-External -FilePath $VenvPython -ArgumentList @("-m", "pip", "install", "-r", "backend\requirements.txt") -FailureMessage "Backend dependency installation failed"

    $npmCmd = if (Test-Command "npm.cmd") { "npm.cmd" } else { "npm" }
    $npxCmd = if (Test-Command "npx.cmd") { "npx.cmd" } else { "npx" }

    if ($ForceSetup -or -not (Test-Path (Join-Path $ProjectRoot "node_modules\next\package.json"))) {
        Write-Step "Installing frontend dependencies"
        if (Test-Path (Join-Path $ProjectRoot "package-lock.json")) {
            Invoke-External -FilePath $npmCmd -ArgumentList @("ci") -FailureMessage "Frontend dependency installation failed"
        } else {
            Invoke-External -FilePath $npmCmd -ArgumentList @("install") -FailureMessage "Frontend dependency installation failed"
        }
    }

    if ($ForceSetup -or -not (Test-Path (Join-Path $WsServiceDir "node_modules\tsx\package.json"))) {
        Write-Step "Installing WebSocket dependencies"
        if (Test-Path (Join-Path $WsServiceDir "package-lock.json")) {
            Invoke-External -FilePath $npmCmd -ArgumentList @("ci") -WorkingDirectory $WsServiceDir -FailureMessage "WebSocket dependency installation failed"
        } else {
            Invoke-External -FilePath $npmCmd -ArgumentList @("install") -WorkingDirectory $WsServiceDir -FailureMessage "WebSocket dependency installation failed"
        }
    }

    Write-Step "Applying database schema"
    Invoke-External -FilePath $npxCmd -ArgumentList @("prisma", "db", "push") -FailureMessage "Database initialization failed"
    Write-Ok "Setup complete"
}

function Stop-App {
    Write-Header "Stopping ECGdoc services"

    $ports = @($FrontendPort, $BackendPort, $WsPort)
    if (Test-Path $PidFile) {
        try {
            $pids = Get-Content $PidFile -Raw | ConvertFrom-Json
            foreach ($url in @($pids.frontendUrl, $pids.backendUrl, $pids.websocketUrl)) {
                if ($url -match ':(\d+)(/|$)') {
                    $ports += [int]$matches[1]
                }
            }
            foreach ($value in @($pids.frontendPid, $pids.backendPid, $pids.wsPid)) {
                if ($value) {
                    Stop-Process -Id ([int]$value) -Force -ErrorAction SilentlyContinue
                }
            }
        } catch {}
    }

    $ports = $ports | Sort-Object -Unique
    foreach ($port in $ports) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            $isKnownPid = $false
            if (Test-Path $PidFile) {
                try {
                    $pids = Get-Content $PidFile -Raw | ConvertFrom-Json
                    $isKnownPid = @($pids.frontendPid, $pids.backendPid, $pids.wsPid) -contains $conn.OwningProcess
                } catch {}
            }
            if ($isKnownPid -or (Test-ProjectProcess $conn.OwningProcess)) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*$ProjectRoot*" -and (
            $_.CommandLine -like "*uvicorn*" -or
            $_.CommandLine -like "*next\dist\server\lib\start-server.js*" -or
            $_.CommandLine -like "*node_modules\tsx*" -or
            $_.CommandLine -like "*npm.cmd*run dev*"
        )
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Ok "Stopped project-owned services"
}

function Get-FreeBackendPort {
    param([int]$Preferred)
    for ($port = $Preferred; $port -le ($Preferred + 50); $port++) {
        if (-not (Test-PortListening $port)) { return $port }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 1
            if ($health.ok -eq $true -and $health.version -eq "1.1.0") {
                $openapi = Invoke-WebRequest -Uri "http://127.0.0.1:$port/openapi.json" -UseBasicParsing -TimeoutSec 2
                if (
                    $openapi.Content -like "*/api/evaluation/run*" -and
                    $openapi.Content -like "*/api/evaluation/start*" -and
                    $openapi.Content -like "*/api/evaluation/progress*"
                ) {
                    return $port
                }
                Write-Warn "Port $port has an older ECG API instance; using the next free port"
            }
        } catch {}
    }
    throw "No free backend port found from $Preferred to $($Preferred + 50)."
}

function Wait-HttpOk {
    param([string]$Url, [int]$Seconds = 60, [switch]$ExpectJsonOk)
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            if ($ExpectJsonOk) {
                $resp = Invoke-RestMethod -Uri $Url -TimeoutSec 2
                if ($resp.ok -eq $true) { return $true }
            } else {
                $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
                if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { return $true }
            }
        } catch {}
        Start-Sleep -Milliseconds 700
    }
    return $false
}

function Ensure-DemoUser {
    param([int]$Port)
    $body = @{ email = "demo@ecg.local"; password = "demo1234" } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/auth/login" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 5 | Out-Null
        Write-Ok "Demo login is ready"
        return
    } catch {}

    $registerBody = @{ email = "demo@ecg.local"; name = "Demo Clinician"; password = "demo1234" } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/auth/register" -Method POST -ContentType "application/json" -Body $registerBody -TimeoutSec 5 | Out-Null
        Write-Ok "Demo user registered"
        return
    } catch {}

    Write-Step "Resetting demo user password"
    $python = @'
from sqlalchemy import text
from app.db.session import get_db
from app.core.auth import hash_password
import secrets

row = {
    "email": "demo@ecg.local",
    "name": "Demo Clinician",
    "pwh": hash_password("demo1234"),
}
with get_db() as db:
    existing = db.execute(
        text("SELECT id FROM User WHERE email=:email"),
        row,
    ).fetchone()
    if existing:
        db.execute(
            text(
                "UPDATE User SET name=:name, role='clinician', "
                "passwordHash=:pwh, updatedAt=datetime('now') "
                "WHERE email=:email"
            ),
            row,
        )
    else:
        row["id"] = secrets.token_hex(12)
        db.execute(
            text(
                "INSERT INTO User "
                "(id,email,name,role,passwordHash,createdAt,updatedAt) "
                "VALUES (:id,:email,:name,'clinician',:pwh,datetime('now'),datetime('now'))"
            ),
            row,
        )
    db.commit()
'@
    $old = Get-Location
    try {
        Set-Location $BackendDir
        $python | & $VenvPython -
    } finally {
        Set-Location $old
    }
    Write-Ok "Demo login is ready"
}

function Show-Status {
    Write-Header "Status"
    $ports = @($FrontendPort, $BackendPort, $WsPort)
    if (Test-Path $PidFile) {
        try {
            $saved = Get-Content $PidFile -Raw | ConvertFrom-Json
            foreach ($url in @($saved.frontendUrl, $saved.backendUrl, $saved.websocketUrl)) {
                if ($url -match ':(\d+)(/|$)') {
                    $ports += [int]$matches[1]
                }
            }
        } catch {}
    }
    foreach ($port in ($ports | Sort-Object -Unique)) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            foreach ($conn in $connections) {
                $owner = if (Test-ProjectProcess $conn.OwningProcess) { "project" } else { "external" }
                if ($owner -eq "external") {
                    try {
                        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 1
                        if ($health.ok -eq $true -and $health.version -eq "1.1.0") {
                            $owner = "project-api"
                        }
                    } catch {}
                }
                Write-Host "Port ${port}: listening, PID $($conn.OwningProcess), $owner"
            }
        } else {
            Write-Host "Port ${port}: free"
        }
    }
    if (Test-Path $PidFile) {
        Write-Host ""
        Get-Content $PidFile
    }
}

if ($Stop) {
    Stop-App
    return
}

if ($Status) {
    Show-Status
    return
}

if ($Restart) {
    Stop-App
}

New-Item -ItemType Directory -Force -Path $RuntimeDir, $LogDir | Out-Null
Ensure-Setup

$BackendPort = Get-FreeBackendPort -Preferred $BackendPort
$npmCmd = if (Test-Command "npm.cmd") { "npm.cmd" } else { "npm" }

Write-Header "Starting ECGdoc app"

$backendOwnerPid = $null
if (-not (Test-PortListening $BackendPort)) {
    Write-Step "Starting backend on port $BackendPort"
    $backend = Start-ServiceProcess `
        -Name "Backend" `
        -FilePath $VenvPython `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$BackendPort", "--reload", "--reload-dir", "app") `
        -WorkingDirectory $BackendDir `
        -LogBaseName "backend"
} else {
    $backend = $null
    $backendConn = Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($backendConn) { $backendOwnerPid = $backendConn.OwningProcess }
    Write-Ok "Reusing backend on port $BackendPort"
}

if (-not (Test-PortListening $WsPort)) {
    Write-Step "Starting WebSocket service on port $WsPort"
    $ws = Start-ServiceProcess `
        -Name "WebSocket" `
        -FilePath $npmCmd `
        -ArgumentList @("run", "dev:node") `
        -WorkingDirectory $WsServiceDir `
        -LogBaseName "websocket"
} else {
    $ws = $null
    Write-Ok "Reusing WebSocket service on port $WsPort"
}

if (Test-PortListening $FrontendPort) {
    $frontConn = Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($frontConn -and (Test-ProjectProcess $frontConn.OwningProcess)) {
        Stop-Process -Id $frontConn.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    } else {
        throw "Frontend port $FrontendPort is in use by another process."
    }
}

Write-Step "Starting frontend on port $FrontendPort"
$frontend = Start-ServiceProcess `
    -Name "Frontend" `
    -FilePath $npmCmd `
    -ArgumentList @("run", "dev") `
    -WorkingDirectory $ProjectRoot `
    -LogBaseName "frontend" `
    -Environment @{
        NEXT_PUBLIC_API_PORT = "$BackendPort"
        NEXT_PUBLIC_WS_PORT = "$WsPort"
    }

Write-Step "Waiting for services"
$backendReady = Wait-HttpOk -Url "http://127.0.0.1:$BackendPort/health" -Seconds 90 -ExpectJsonOk
$frontendReady = Wait-HttpOk -Url "http://127.0.0.1:$FrontendPort" -Seconds 90
$wsReady = Test-PortListening $WsPort

if (-not $backendReady) { throw "Backend did not become ready. See runtime/logs/backend.err.log." }
if (-not $frontendReady) { throw "Frontend did not become ready. See runtime/logs/frontend.err.log." }
if (-not $wsReady) { throw "WebSocket service did not bind to port $WsPort. See runtime/logs/websocket.err.log." }

Ensure-DemoUser -Port $BackendPort

[pscustomobject]@{
    frontendPid = $frontend.Id
    backendPid = if ($backend) { $backend.Id } else { $backendOwnerPid }
    wsPid = if ($ws) { $ws.Id } else { $null }
    frontendUrl = "http://localhost:$FrontendPort"
    backendUrl = "http://localhost:$BackendPort"
    websocketUrl = "http://localhost:$WsPort"
    startedAt = (Get-Date).ToString("s")
} | ConvertTo-Json | Set-Content $PidFile -Encoding ASCII

Write-Header "App is running"
Write-Host "Frontend:  http://localhost:$FrontendPort"
Write-Host "Backend:   http://localhost:$BackendPort"
Write-Host "API docs:  http://localhost:$BackendPort/docs"
Write-Host "WebSocket: http://localhost:$WsPort"
Write-Host ""
Write-Host "Login:    demo@ecg.local"
Write-Host "Password: demo1234"
Write-Host ""
if ($HiddenServices) {
    Write-Host "Logs:     runtime\logs (services started in hidden mode)"
} else {
    Write-Host "Logs:     live in separate service windows and runtime\logs"
}
Write-Host "Stop:     .\run-app.ps1 -Stop"

if ($Open) {
    Start-Process "http://localhost:$FrontendPort"
}
