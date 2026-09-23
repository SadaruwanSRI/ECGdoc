<#
.SYNOPSIS
    ECG Anomaly Detection - Full Stack Launcher for Windows

.DESCRIPTION
    Starts the entire ECG Anomaly Detection system on your local machine.

.PARAMETER UsePostgres
    Switch to use PostgreSQL instead of the default SQLite.

.PARAMETER SkipSetup
    Skip dependency checks and installation.

.PARAMETER ForceSetup
    Re-run dependency installation and database initialization even when an
    existing installation is detected.

.PARAMETER TrainDemo
    Train a demo model immediately after starting services.

.EXAMPLE
    .\start-all.ps1
    First-time setup with SQLite + start everything.

.EXAMPLE
    .\start-all.ps1 -UsePostgres
    Use PostgreSQL instead of SQLite.

.EXAMPLE
    .\start-all.ps1 -SkipSetup -TrainDemo
    Skip install steps, just start services + train a demo model.

.EXAMPLE
    .\start-all.ps1 -ForceSetup
    Reinstall dependencies, initialize the database, and start everything.
#>

param(
    [switch]$UsePostgres,
    [switch]$SkipSetup,
    [switch]$ForceSetup,
    [switch]$TrainDemo,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# ============================================================================
# Configuration
# ============================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot  = $PSScriptRoot
$BackendDir   = Join-Path $ProjectRoot "backend"
$WsServiceDir = Join-Path $ProjectRoot "mini-services\ws-service"
$EnvFile      = Join-Path $ProjectRoot ".env"
$BackendEnvFile = Join-Path $BackendDir ".env"
$EnvExample   = Join-Path $ProjectRoot ".env.example"
$VenvDir      = Join-Path $BackendDir "venv"

$BackendPort  = 8000
$WsPort       = 3003
$FrontendPort = 3000

# Pretty-print helpers (ASCII only)
function Write-Header($msg) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Write-Step($msg) {
    Write-Host "==> $msg" -ForegroundColor Yellow
}

function Write-Ok($msg) {
    Write-Host "    OK  $msg" -ForegroundColor Green
}

function Write-Err($msg) {
    Write-Host "    ERR $msg" -ForegroundColor Red
}

function Test-Command($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Test-PortListening([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Invoke-External {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $null,
        [switch]$Quiet,
        [string]$FailureMessage = "Command failed"
    )

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $oldLocation = Get-Location
    try {
        if ($WorkingDirectory) {
            Set-Location $WorkingDirectory
        }

        $output = & $FilePath @ArgumentList 2>&1
        $exitCode = $LASTEXITCODE

        if (-not $Quiet -and $output) {
            $output | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        }

        if ($exitCode -ne 0) {
            Write-Err "$FailureMessage (exit code $exitCode)"
            if ($Quiet -and $output) {
                Write-Host ""
                Write-Host "  ---- command output ----" -ForegroundColor DarkGray
                $output | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
                Write-Host "  ------------------------" -ForegroundColor DarkGray
                Write-Host ""
            }
            exit $exitCode
        }

    } finally {
        if ($WorkingDirectory) {
            Set-Location $oldLocation
        }
        $ErrorActionPreference = $prev
    }
}

function Get-EnvValue {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Name,
        [string]$Default = ""
    )

    if (-not (Test-Path $Path)) {
        return $Default
    }

    $line = Get-Content $Path | Where-Object { $_ -match "^\s*$Name\s*=" } | Select-Object -First 1
    if (-not $line) {
        return $Default
    }

    return (($line -replace "^\s*$Name\s*=\s*", "").Trim().Trim('"').Trim("'"))
}

function Set-EnvValue {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Value
    )

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

function Ensure-EnvFiles {
    param([bool]$Postgres)

    $databaseUrl = "file:./dev.db"
    if ($Postgres) {
        $databaseUrl = "postgresql://ecg_user:ecg_password@localhost:5432/ecg_anomaly"
    }

    if (-not (Test-Path $EnvFile)) {
        if (Test-Path $EnvExample) {
            Copy-Item $EnvExample $EnvFile
            Write-Ok "Created .env from .env.example"
        } else {
            @(
                "# ECG Anomaly Detection local configuration",
                "DATABASE_URL=`"$databaseUrl`""
            ) | Set-Content $EnvFile -Encoding ASCII
            Write-Ok "Created .env with default local settings"
        }
    } else {
        Write-Ok ".env already exists - preserving existing values"
    }

    if ($Postgres) {
        Set-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Value $databaseUrl
        Write-Ok "Configured root .env for PostgreSQL"
    } elseif (-not (Get-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Default "")) {
        Set-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Value $databaseUrl
        Write-Ok "Added SQLite DATABASE_URL to root .env"
    }

    $effectiveDatabaseUrl = Get-EnvValue -Path $EnvFile -Name "DATABASE_URL" -Default $databaseUrl
    Set-EnvValue -Path $BackendEnvFile -Name "DATABASE_URL" -Value $effectiveDatabaseUrl
    Write-Ok "Synced backend .env"
}

function Set-PrismaProvider {
    param([bool]$Postgres)

    $schemaPath = Join-Path $ProjectRoot "prisma\schema.prisma"
    $provider = if ($Postgres) { "postgresql" } else { "sqlite" }
    (Get-Content $schemaPath) -replace 'provider = "(sqlite|postgresql)"', "provider = `"$provider`"" | Set-Content $schemaPath -Encoding UTF8
    Write-Ok "Prisma provider: $provider"
}

# Pick the JavaScript runtime once so setup and launch agree, including -SkipSetup.
$useBun = $false
if (Test-Command "bun") {
    $useBun = $true
}
$npmCmd = if (Test-Command "npm.cmd") { "npm.cmd" } else { "npm" }
$npxCmd = if (Test-Command "npx.cmd") { "npx.cmd" } else { "npx" }

# Setup is expensive (especially pip/torch and the optional GitHub package), so
# only run it when the local installation is incomplete.  -SkipSetup remains a
# fast, unchecked override and -ForceSetup provides an explicit repair/update
# path.  Previously a plain .\start-all.ps1 repeated every install on every run.
$setupReasons = @()
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    $setupReasons += "Python virtual environment is missing"
}
if (-not (Test-Path (Join-Path $ProjectRoot "node_modules\next\package.json"))) {
    $setupReasons += "frontend dependencies are missing"
}
if (-not (Test-Path (Join-Path $WsServiceDir "node_modules\socket.io\package.json")) -or
    -not (Test-Path (Join-Path $WsServiceDir "node_modules\tsx\package.json"))) {
    $setupReasons += "WebSocket dependencies are missing"
}
if (-not (Test-Path $EnvFile) -or -not (Test-Path $BackendEnvFile)) {
    $setupReasons += "environment configuration is missing"
}
if (-not $UsePostgres -and -not (Test-Path (Join-Path $ProjectRoot "prisma\dev.db"))) {
    $setupReasons += "SQLite database is missing"
}

$runSetup = -not $SkipSetup -and ($ForceSetup -or $setupReasons.Count -gt 0)

# ============================================================================
# Step 1: Verify prerequisites
# ============================================================================
Write-Header "ECG Anomaly Detection - Full Stack Launcher"
Write-Host "  Project root: $ProjectRoot" -ForegroundColor DarkGray
if ($UsePostgres) {
    Write-Host "  Mode: PostgreSQL" -ForegroundColor DarkGray
} else {
    Write-Host "  Mode: SQLite (default)" -ForegroundColor DarkGray
}

if ($ForceSetup) {
    Write-Host "  Setup: forced" -ForegroundColor DarkGray
} elseif ($SkipSetup) {
    Write-Host "  Setup: skipped by -SkipSetup" -ForegroundColor DarkGray
} elseif ($runSetup) {
    Write-Host "  Setup: required ($($setupReasons -join '; '))" -ForegroundColor DarkGray
} else {
    Write-Ok "Existing installation detected - skipping setup"
    Write-Host "       Use -ForceSetup when dependencies or database schema change." -ForegroundColor DarkGray
}

if ($runSetup) {
    Write-Header "Step 1/5 - Verifying prerequisites"

    # Python
    $pythonLauncher = $null
    $pythonArgs = @()
    if (Test-Command "python") {
        $pythonLauncher = "python"
    } elseif (Test-Command "py") {
        $pythonLauncher = "py"
        $pythonArgs = @("-3")
    } else {
        Write-Err "Python not found. Install Python 3.10+ from https://www.python.org/downloads/"
        exit 1
    }

    $pyVer = (& $pythonLauncher @pythonArgs --version 2>&1) -replace "Python ", ""
    Write-Ok "Python $pyVer"

    # Bun (preferred) or Node
    if (Test-Command "bun") {
        $bunVer = (bun --version 2>&1)
        Write-Ok "Bun $bunVer"
        $useBun = $true
    } elseif (Test-Command "node") {
        $nodeVer = (node --version 2>&1)
        Write-Ok "Node.js $nodeVer (Bun not installed - using npm instead)"
    } else {
        Write-Err "Neither Bun nor Node.js found. Install one:"
        Write-Host "       Bun:  https://bun.sh/" -ForegroundColor DarkGray
        Write-Host "       Node: https://nodejs.org/" -ForegroundColor DarkGray
        exit 1
    }

    # PostgreSQL (only if -UsePostgres)
    if ($UsePostgres) {
        if (Test-Command "psql") {
            $pgVer = (psql --version 2>&1)
            Write-Ok "PostgreSQL: $pgVer"
        } else {
            Write-Err "PostgreSQL not found. Install from https://www.postgresql.org/download/windows/"
            Write-Host "       Then run: psql -U postgres -f setup-postgres.sql" -ForegroundColor DarkGray
            exit 1
        }
    } else {
        Write-Ok "Using SQLite (no setup required)"
    }
} else {
    $pythonLauncher = if (Test-Command "python") { "python" } elseif (Test-Command "py") { "py" } else { $null }
    $pythonArgs = if ($pythonLauncher -eq "py") { @("-3") } else { @() }
}

# ============================================================================
# Step 2: Set up .env file
# ============================================================================
if ($runSetup) {
    Write-Header "Step 2/5 - Setting up .env file"

    Ensure-EnvFiles -Postgres ([bool]$UsePostgres)
    Set-PrismaProvider -Postgres ([bool]$UsePostgres)
}

# ============================================================================
# Step 3: Set up Python backend
# ============================================================================
if ($runSetup) {
    Write-Header "Step 3/5 - Setting up Python backend"

    # Create venv if missing
    if (-not (Test-Path $VenvDir)) {
        Write-Step "Creating Python virtual environment..."
        Invoke-External -FilePath $pythonLauncher -ArgumentList ($pythonArgs + @("-m", "venv", "venv")) -WorkingDirectory $BackendDir -FailureMessage "Could not create Python virtual environment"
        Write-Ok "Created venv at $VenvDir"
    } else {
        Write-Ok "venv already exists"
    }

    $pipExe = Join-Path $VenvDir "Scripts\pip.exe"
    $pyExe  = Join-Path $VenvDir "Scripts\python.exe"

    # Upgrade pip
    Write-Step "Upgrading pip..."
    Invoke-External -FilePath $pyExe -ArgumentList @("-m", "pip", "install", "--upgrade", "pip", "--quiet") -Quiet -FailureMessage "pip upgrade failed"
    Write-Ok "pip upgraded"

    # Install requirements
    Write-Step "Installing Python dependencies (this may take 2-5 minutes on first run)..."
    $reqPath = Join-Path $BackendDir "requirements.txt"
    Invoke-External -FilePath $pipExe -ArgumentList @("install", "-r", $reqPath) -Quiet -FailureMessage "Python dependency installation failed"
    Write-Ok "Python dependencies installed"

    # Try to install ecg_qc separately (optional - system falls back to simple filter if this fails)
    # IMPORTANT: pip writes dependency-conflict warnings to stderr, which PowerShell's
    # $ErrorActionPreference="Stop" treats as a fatal NativeCommandError. We temporarily
    # relax the preference to "Continue" during the ecg_qc install so warnings don't
    # kill the script. The whole block is also wrapped in try/catch as a safety net.
    Write-Step "Installing ecg_qc (optional, for better ECG quality filtering)..."
    $ecgQcSuccess = $false
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Step 1: Install ecg-qc itself with --no-deps (avoids pulling pathtools automatically)
        & $pipExe install ecg-qc --no-deps 2>&1 | Out-Null

        # Step 2: Install the other ecg_qc dependencies (not pathtools yet)
        & $pipExe install biosppy py-ecg-detectors scikit-learn peakutils dill 2>&1 | Out-Null

        # Step 3: Create a stub `imp` module so pathtools can install on Python 3.12+
        $stubPath = Join-Path $VenvDir "Lib\site-packages\imp.py"
        $stubContent = @'
"""Stub for the deprecated `imp` module (removed in Python 3.12)."""
import importlib.util, importlib.machinery, types
PY_SOURCE = 1
def find_module(name, path=None):
    spec = importlib.machinery.PathFinder.find_spec(name, path)
    if spec is None: raise ImportError("No module named " + repr(name))
    return (None, spec.origin, ("", "", PY_SOURCE))
def load_module(name, file, pathname, description):
    return importlib.import_module(name)
def load_source(name, pathname, file=None):
    spec = importlib.util.spec_from_file_location(name, pathname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
def new_module(name): return types.ModuleType(name)
def get_suffixes(): return [(".py", "r", PY_SOURCE)]
'@
        $stubContent | Out-File -FilePath $stubPath -Encoding UTF8 -Force

        # Step 4: Install pathtools with --no-build-isolation (uses our stub)
        & $pipExe install pathtools --no-build-isolation 2>&1 | Out-Null

        # Step 5: Install ecg_qc source from GitHub (the PyPI wheel is empty)
        & $pipExe install --no-deps --force-reinstall "git+https://github.com/Aura-healthcare/ecg_qc.git@main" 2>&1 | Out-Null

        # Step 6: Verify the install works
        $verifyResult = & $pyExe -c "import warnings; warnings.filterwarnings('ignore'); from ecg_qc.sqi_computing.sqi_rr_intervals import qsqi; print('ECGQC_OK')" 2>&1
        if ($verifyResult -match "ECGQC_OK") {
            $ecgQcSuccess = $true
        }
    } catch {
        # Any error in the above is non-fatal - we just fall back to simple filter
        $ecgQcSuccess = $false
    } finally {
        # Restore the original error preference
        $ErrorActionPreference = $prevErrorAction
    }

    if ($ecgQcSuccess) {
        Write-Ok "ecg_qc installed - SQI-based quality filtering is enabled"
    } else {
        Write-Host "    WARN  ecg_qc install had issues - will use simple std-based filter" -ForegroundColor Yellow
        Write-Host "       The system works fine without it. To retry later:" -ForegroundColor DarkGray
        Write-Host "       .\install-ecg-qc.ps1" -ForegroundColor DarkGray
    }
}

# ============================================================================
# Step 4: Set up frontend + WebSocket service
# ============================================================================
if ($runSetup) {
    Write-Header "Step 4/5 - Setting up frontend dependencies"

    # Frontend (Next.js)
    Write-Step "Installing frontend dependencies..."
    if ($useBun) {
        Invoke-External -FilePath "bun" -ArgumentList @("install") -WorkingDirectory $ProjectRoot -FailureMessage "Frontend dependency installation failed"
    } elseif (Test-Path (Join-Path $ProjectRoot "package-lock.json")) {
        Invoke-External -FilePath $npmCmd -ArgumentList @("ci") -WorkingDirectory $ProjectRoot -FailureMessage "Frontend dependency installation failed"
    } else {
        Invoke-External -FilePath $npmCmd -ArgumentList @("install") -WorkingDirectory $ProjectRoot -FailureMessage "Frontend dependency installation failed"
    }
    Write-Ok "Frontend dependencies installed"

    # WebSocket service
    Write-Step "Installing WebSocket service dependencies..."
    if ($useBun) {
        Invoke-External -FilePath "bun" -ArgumentList @("install") -WorkingDirectory $WsServiceDir -FailureMessage "WebSocket dependency installation failed"
    } elseif (Test-Path (Join-Path $WsServiceDir "package-lock.json")) {
        Invoke-External -FilePath $npmCmd -ArgumentList @("ci") -WorkingDirectory $WsServiceDir -FailureMessage "WebSocket dependency installation failed"
    } else {
        Invoke-External -FilePath $npmCmd -ArgumentList @("install") -WorkingDirectory $WsServiceDir -FailureMessage "WebSocket dependency installation failed"
    }
    Write-Ok "WebSocket service dependencies installed"
}

# ============================================================================
# Step 5: Initialize database
# ============================================================================
if ($runSetup) {
    Write-Header "Step 5/5 - Initializing database"

    # Clean up any stale SQLite files from previous versions that might have
    # been created in the wrong location (backend/dev.db instead of prisma/dev.db)
    $staleDbPaths = @(
        (Join-Path $BackendDir "dev.db"),
        (Join-Path $ProjectRoot "dev.db")
    )
    foreach ($staleDb in $staleDbPaths) {
        if (Test-Path $staleDb) {
            Write-Step "Removing stale database file: $staleDb"
            Remove-Item $staleDb -Force
        }
    }

    # Run prisma db push to create/update the schema
    # Prisma creates the SQLite file relative to the prisma/ directory,
    # so "file:./dev.db" becomes prisma/dev.db
    if ($useBun) {
        Invoke-External -FilePath "bun" -ArgumentList @("run", "db:push") -WorkingDirectory $ProjectRoot -FailureMessage "Database initialization failed"
    } else {
        Invoke-External -FilePath $npxCmd -ArgumentList @("prisma", "db", "push") -WorkingDirectory $ProjectRoot -FailureMessage "Database initialization failed"
    }
    Write-Ok "Database schema applied"

    # Verify the database file exists where we expect it
    $expectedDbPath = Join-Path $ProjectRoot "prisma\dev.db"
    if (Test-Path $expectedDbPath) {
        Write-Ok "Database file: $expectedDbPath"
    } else {
        Write-Host "    WARN  Database file not found at expected location:" -ForegroundColor Yellow
        Write-Host "       $expectedDbPath" -ForegroundColor DarkGray
        Write-Host "       The backend may fail to connect. Check the .env DATABASE_URL setting." -ForegroundColor DarkGray
    }
}

# ============================================================================
# Launch services
# ============================================================================
Write-Header "Starting services"

# A previous launcher window or detached service may still be running. Starting
# a duplicate Next.js process causes EADDRINUSE / .next\dev\lock errors, so
# reuse listeners on the expected ports and validate them below.
$backendReady = $false
$frontendReady = $false
try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 1 -ErrorAction Stop
    $backendReady = [bool]$resp.ok
} catch {}
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort" -TimeoutSec 1 -ErrorAction Stop -UseBasicParsing
    $frontendReady = $resp.StatusCode -eq 200
} catch {}
$backendPortInUse = Test-PortListening $BackendPort
$frontendPortInUse = Test-PortListening $FrontendPort
$wsPortInUse = Test-PortListening $WsPort

# ---------- FastAPI backend (new window) ----------
if ($backendPortInUse) {
    Write-Ok "Backend port $BackendPort already has a listener - reusing it"
} else {
    Write-Step "Starting FastAPI backend on port $BackendPort..."
    $backendBat = Join-Path $env:TEMP "ecg-backend.bat"
    $backendLines = @(
        "@echo off",
        "title ECG-Backend (FastAPI port $BackendPort)",
        "cd /d `"$BackendDir`"",
        "call venv\Scripts\activate.bat",
        "echo Starting FastAPI backend on port $BackendPort ...",
        "echo.",
        # Limit reload watching to source code. Watching all of backend also scans
        # the large venv, trained models, datasets, and generated reports.
        "python -m uvicorn app.main:app --host 0.0.0.0 --port $BackendPort --reload --reload-dir app",
        "echo.",
        "echo Backend has stopped. Press any key to close this window.",
        "pause >nul"
    )
    $backendLines | Set-Content -Path $backendBat -Encoding ASCII
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "`"$backendBat`""
    Write-Ok "FastAPI backend started (new window)"
}

# ---------- WebSocket service (new window) ----------
if ($wsPortInUse) {
    Write-Ok "WebSocket port $WsPort already has a listener - reusing it"
} else {
    Write-Step "Starting WebSocket service on port $WsPort..."
    $wsBat = Join-Path $env:TEMP "ecg-ws.bat"
    $wsLines = @(
        "@echo off",
        "title ECG-WebSocket (port $WsPort)",
        "cd /d `"$WsServiceDir`"",
        "echo Starting WebSocket service on port $WsPort ...",
        "echo."
    )
    if ($useBun) {
        $wsLines += "bun run index.ts"
    } else {
        $wsLines += "npm run dev:node"
    }
    $wsLines += "echo."
    $wsLines += "echo WebSocket service has stopped. Press any key to close this window."
    $wsLines += "pause >nul"
    $wsLines | Set-Content -Path $wsBat -Encoding ASCII
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "`"$wsBat`""
    Write-Ok "WebSocket service started (new window)"
}

# ---------- Next.js frontend (new window) ----------
if ($frontendPortInUse) {
    Write-Ok "Frontend port $FrontendPort already has a listener - reusing it"
} else {
    Write-Step "Starting Next.js frontend on port $FrontendPort..."
    $frontendBat = Join-Path $env:TEMP "ecg-frontend.bat"
    $frontendLines = @(
        "@echo off",
        "title ECG-Frontend (Next.js port $FrontendPort)",
        "cd /d `"$ProjectRoot`"",
        "echo Starting Next.js frontend on port $FrontendPort ...",
        "echo."
    )
    if ($useBun) {
        $frontendLines += "bun run dev"
    } else {
        $frontendLines += "npm run dev"
    }
    $frontendLines += "echo."
    $frontendLines += "echo Frontend has stopped. Press any key to close this window."
    $frontendLines += "pause >nul"
    $frontendLines | Set-Content -Path $frontendBat -Encoding ASCII
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "`"$frontendBat`""
    Write-Ok "Next.js frontend started (new window)"
}

# Both applications are loading now. Poll them together instead of delaying the
# frontend until after the scientific Python stack has finished importing.
Write-Step "Waiting for backend and frontend to be ready..."
$readinessDeadline = [DateTime]::UtcNow.AddSeconds(45)
while ([DateTime]::UtcNow -lt $readinessDeadline -and -not ($backendReady -and $frontendReady)) {
    if (-not $backendReady) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 1 -ErrorAction Stop
            $backendReady = [bool]$resp.ok
        } catch {}
    }

    if (-not $frontendReady) {
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort" -TimeoutSec 1 -ErrorAction Stop -UseBasicParsing
            $frontendReady = $resp.StatusCode -eq 200
        } catch {}
    }

    if (-not ($backendReady -and $frontendReady)) {
        Start-Sleep -Milliseconds 500
    }
}

if ($backendReady) {
    Write-Ok "Backend is ready"
} else {
    Write-Err "Backend did not become ready in 45s - check the backend window for errors"
}
if ($frontendReady) {
    Write-Ok "Frontend is ready"
} else {
    Write-Err "Frontend did not become ready in 45s - check the frontend window for errors"
}

# ============================================================================
# First-time setup: create demo user + train demo model
# ============================================================================
$firstRunMarker = Join-Path $ProjectRoot ".first-run-complete"
if (-not (Test-Path $firstRunMarker)) {
    Write-Header "First-run setup: demo user"

    # Register demo user
    Write-Step "Creating demo user (demo@ecg.local / demo1234)..."
    $token = $null
    try {
        $body = @{ email = "demo@ecg.local"; name = "Demo Clinician"; password = "demo1234" } | ConvertTo-Json
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/auth/register" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
        $token = $resp.data.token
        Write-Ok "Demo user registered"
    } catch {
        # Maybe already exists - try logging in
        try {
            $body = @{ email = "demo@ecg.local"; password = "demo1234" } | ConvertTo-Json
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/auth/login" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
            $token = $resp.data.token
            Write-Ok "Demo user already exists - logged in"
        } catch {
            Write-Err "Could not create/login demo user: $($_.Exception.Message)"
        }
    }

    if ($token -and $TrainDemo) {
        Write-Step "Training a demo model (3 epochs on synthetic data, ~30s)..."
        $trainBody = @{
            model_name    = "demo-model"
            description   = "Auto-trained by start-all.ps1"
            epochs        = 3
            batch_size    = 32
            learning_rate = 0.001
            dataset_name  = "synthetic"
        } | ConvertTo-Json
        $headers = @{ Authorization = "Bearer $token" }
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/training/start" -Method POST -Body $trainBody -ContentType "application/json" -Headers $headers -TimeoutSec 10
            $runId = $resp.data.run_id
            Write-Ok "Training started (run_id: $runId)"

            # Poll for completion
            Write-Step "Waiting for training to complete..."
            for ($i = 0; $i -lt 60; $i++) {
                Start-Sleep -Seconds 5
                try {
                    $status = (Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/training/$runId" -Headers $headers -TimeoutSec 5).data.status
                    if ($status -eq "completed") {
                        Write-Ok "Demo model trained successfully!"
                        break
                    } elseif ($status -eq "failed") {
                        Write-Err "Training failed - check the backend window"
                        break
                    }
                    Write-Host "    ... status: $status" -ForegroundColor DarkGray
                } catch {}
            }
        } catch {
            Write-Err "Training start failed: $($_.Exception.Message)"
        }
    }

    New-Item -Path $firstRunMarker -ItemType File -Force | Out-Null
    Write-Ok "First-run setup complete"
}

# ============================================================================
# Done - open browser
# ============================================================================
Write-Header "All services are running!"

Write-Host "  Frontend (Next.js):    http://localhost:$FrontendPort" -ForegroundColor White
Write-Host "  Backend  (FastAPI):    http://localhost:$BackendPort/health" -ForegroundColor White
Write-Host "  Backend  (API docs):   http://localhost:$BackendPort/docs" -ForegroundColor White
Write-Host "  WebSocket service:    ws://localhost:$WsPort" -ForegroundColor White
Write-Host ""
Write-Host "  Login:    demo@ecg.local" -ForegroundColor Yellow
Write-Host "  Password: demo1234" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To stop everything:  close the 3 service windows that just opened." -ForegroundColor DarkGray
Write-Host "  To start again later: .\start-all.ps1" -ForegroundColor DarkGray
Write-Host "  To reinstall/update setup: .\start-all.ps1 -ForceSetup" -ForegroundColor DarkGray
Write-Host ""

$openBrowser = Read-Host "Open browser now? [Y/n]"
if ($openBrowser -ne "n") {
    Start-Process "http://localhost:$FrontendPort"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
