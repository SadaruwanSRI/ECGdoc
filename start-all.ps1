<#
.SYNOPSIS
    ECG Anomaly Detection - Full Stack Launcher for Windows

.DESCRIPTION
    Starts the entire ECG Anomaly Detection system on your local machine.

.PARAMETER UsePostgres
    Switch to use PostgreSQL instead of the default SQLite.

.PARAMETER SkipSetup
    Skip dependency installation (use after first run).

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
#>

param(
    [switch]$UsePostgres,
    [switch]$SkipSetup,
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

# Run a native command (npm, npx, pip, bun, prisma, etc.) without letting
# stderr output kill the script. Many tools write informational messages
# ("Environment variables loaded from .env", dependency-conflict warnings,
# etc.) to stderr, which PowerShell's $ErrorActionPreference="Stop" would
# otherwise treat as a fatal NativeCommandError.
function Invoke-NativeSafe {
    param([scriptblock]$Block)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Block 2>&1 | Out-Null
    } catch {
        # Ignore - the command may have written to stderr but still succeeded
    } finally {
        $ErrorActionPreference = $prev
    }
}

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

if (-not $SkipSetup) {
    Write-Header "Step 1/5 - Verifying prerequisites"

    # Python
    if (-not (Test-Command "python")) {
        Write-Err "Python not found. Install Python 3.10+ from https://www.python.org/downloads/"
        exit 1
    }
    $pyVer = (python --version 2>&1) -replace "Python ", ""
    Write-Ok "Python $pyVer"

    # Bun (preferred) or Node
    $useBun = $false
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
}

# ============================================================================
# Step 2: Set up .env file
# ============================================================================
if (-not $SkipSetup) {
    Write-Header "Step 2/5 - Setting up .env file"

    if (-not (Test-Path $EnvFile)) {
        Copy-Item $EnvExample $EnvFile
        Write-Ok "Created .env from .env.example"

        if ($UsePostgres) {
            # Switch DATABASE_URL to PostgreSQL
            (Get-Content $EnvFile) -replace 'DATABASE_URL="file:./dev.db"', 'DATABASE_URL="postgresql://ecg_user:ecg_password@localhost:5432/ecg_anomaly"' | Set-Content $EnvFile
            Write-Ok "Switched DATABASE_URL to PostgreSQL"

            # Update Prisma schema to use postgresql provider
            $schemaPath = Join-Path $ProjectRoot "prisma\schema.prisma"
            (Get-Content $schemaPath) -replace 'provider = "sqlite"', 'provider = "postgresql"' | Set-Content $schemaPath
            Write-Ok "Updated prisma/schema.prisma to use PostgreSQL"
        }
    } else {
        Write-Ok ".env already exists - leaving as-is"
    }
}

# ============================================================================
# Step 3: Set up Python backend
# ============================================================================
if (-not $SkipSetup) {
    Write-Header "Step 3/5 - Setting up Python backend"

    # Create venv if missing
    if (-not (Test-Path $VenvDir)) {
        Write-Step "Creating Python virtual environment..."
        Push-Location $BackendDir
        python -m venv venv
        Pop-Location
        Write-Ok "Created venv at $VenvDir"
    } else {
        Write-Ok "venv already exists"
    }

    $pipExe = Join-Path $VenvDir "Scripts\pip.exe"
    $pyExe  = Join-Path $VenvDir "Scripts\python.exe"

    # Upgrade pip
    Write-Step "Upgrading pip..."
    Invoke-NativeSafe { & $pyExe -m pip install --upgrade pip --quiet }
    Write-Ok "pip upgraded"

    # Install requirements
    Write-Step "Installing Python dependencies (this may take 2-5 minutes on first run)..."
    # Use cmd.exe to run pip so stderr is not treated as a PowerShell error stream
    $reqPath = Join-Path $BackendDir "requirements.txt"
    $pipCmd = "`"$pipExe`" install -r `"$reqPath`" 2>&1"
    $installOutput = cmd /c $pipCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Python dependencies installed"
    } else {
        Write-Err "Python dependency installation failed (exit code $LASTEXITCODE):"
        Write-Host ""
        Write-Host "  ---- pip output ----" -ForegroundColor DarkGray
        $installOutput | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
        Write-Host "  ---------------------" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Common fixes:" -ForegroundColor Yellow
        Write-Host "    1. Make sure you have Python 3.10 - 3.13 (you have $pyVer)" -ForegroundColor Yellow
        Write-Host "    2. Try installing the failing package manually:" -ForegroundColor Yellow
        Write-Host "         .\backend\venv\Scripts\pip.exe install <package-name>" -ForegroundColor Yellow
        Write-Host "    3. For torch GPU support, see https://pytorch.org/get-started/locally/" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }

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
# Invoke-NativeSafe helper is defined near the top of the script.
if (-not $SkipSetup) {
    Write-Header "Step 4/5 - Setting up frontend dependencies"

    # Frontend (Next.js)
    Write-Step "Installing frontend dependencies..."
    Push-Location $ProjectRoot
    if ($useBun) {
        Invoke-NativeSafe { bun install }
    } else {
        Invoke-NativeSafe { npm install --silent }
    }
    Pop-Location
    Write-Ok "Frontend dependencies installed"

    # WebSocket service
    Write-Step "Installing WebSocket service dependencies..."
    Push-Location $WsServiceDir
    if ($useBun) {
        Invoke-NativeSafe { bun install }
    } else {
        if (-not (Test-Path "node_modules")) {
            Invoke-NativeSafe { npm init -y }
            Invoke-NativeSafe { npm install socket.io --silent }
        }
    }
    Pop-Location
    Write-Ok "WebSocket service dependencies installed"
}

# ============================================================================
# Step 5: Initialize database
# ============================================================================
if (-not $SkipSetup) {
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
    Push-Location $ProjectRoot
    if ($useBun) {
        Invoke-NativeSafe { bun run db:push }
    } else {
        Invoke-NativeSafe { npx prisma db push }
    }
    Pop-Location
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

# ---------- FastAPI backend (new window) ----------
Write-Step "Starting FastAPI backend on port $BackendPort..."
$backendBat = Join-Path $env:TEMP "ecg-backend.bat"
$backendLines = @(
    "@echo off",
    "title ECG-Backend (FastAPI port $BackendPort)",
    "cd /d `"$BackendDir`"",
    "call venv\Scripts\activate.bat",
    "echo Starting FastAPI backend on port $BackendPort ...",
    "echo.",
    "python -m uvicorn app.main:app --host 0.0.0.0 --port $BackendPort --reload",
    "echo.",
    "echo Backend has stopped. Press any key to close this window.",
    "pause >nul"
)
$backendLines | Set-Content -Path $backendBat -Encoding ASCII
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "`"$backendBat`""
Write-Ok "FastAPI backend started (new window)"

# ---------- WebSocket service (new window) ----------
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

# Wait a moment for backend to come up
Write-Step "Waiting for backend to be ready..."
$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:$BackendPort/health" -TimeoutSec 2 -ErrorAction Stop
        if ($resp.ok) {
            $backendReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if ($backendReady) {
    Write-Ok "Backend is ready"
} else {
    Write-Err "Backend did not become ready in 30s - check the backend window for errors"
}

# ---------- Next.js frontend (new window) ----------
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

# Wait for frontend
Write-Step "Waiting for frontend to be ready..."
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$FrontendPort" -TimeoutSec 2 -ErrorAction Stop -UseBasicParsing
        if ($resp.StatusCode -eq 200) {
            Write-Ok "Frontend is ready"
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
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
        $resp = Invoke-RestMethod -Uri "http://localhost:$BackendPort/api/auth/register" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
        $token = $resp.data.token
        Write-Ok "Demo user registered"
    } catch {
        # Maybe already exists - try logging in
        try {
            $body = @{ email = "demo@ecg.local"; password = "demo1234" } | ConvertTo-Json
            $resp = Invoke-RestMethod -Uri "http://localhost:$BackendPort/api/auth/login" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
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
            $resp = Invoke-RestMethod -Uri "http://localhost:$BackendPort/api/training/start" -Method POST -Body $trainBody -ContentType "application/json" -Headers $headers -TimeoutSec 10
            $runId = $resp.data.run_id
            Write-Ok "Training started (run_id: $runId)"

            # Poll for completion
            Write-Step "Waiting for training to complete..."
            for ($i = 0; $i -lt 60; $i++) {
                Start-Sleep -Seconds 5
                try {
                    $status = (Invoke-RestMethod -Uri "http://localhost:$BackendPort/api/training/$runId" -Headers $headers -TimeoutSec 5).data.status
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
Write-Host "  To start again later: .\start-all.ps1 -SkipSetup" -ForegroundColor DarkGray
Write-Host ""

$openBrowser = Read-Host "Open browser now? [Y/n]"
if ($openBrowser -ne "n") {
    Start-Process "http://localhost:$FrontendPort"
}

Write-Host ""
Write-Host "Done. Press any key to exit this launcher..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
