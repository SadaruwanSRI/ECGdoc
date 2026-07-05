<#
.SYNOPSIS
    Install the ecg_qc library for advanced ECG quality classification.

.DESCRIPTION
    ecg_qc is an optional library that classifies ECG signal quality using
    SQI (Signal Quality Indicator) features. The system works without it
    (falls back to a simple std-based filter), but installing it gives
    better quality filtering.

    This script handles two install issues on Python 3.12+:
      1. pathtools (a dependency) uses the removed `imp` module
      2. The pre-trained model pickle is incompatible with modern sklearn
         (we work around this in code by using SQI features directly)

.EXAMPLE
    .\install-ecg-qc.ps1
#>

# Use "Continue" because pip writes dependency-conflict warnings to stderr
# which PowerShell would otherwise treat as fatal NativeCommandError.
$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $BackendDir "venv\Scripts\python.exe"
$VenvPip = Join-Path $BackendDir "venv\Scripts\pip.exe"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Installing ecg_qc (optional ECG quality classifier)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verify venv exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "Python venv not found. Run .\start-all.ps1 first." -ForegroundColor Red
    exit 1
}

# Step 1: Create a stub `imp` module so pathtools can install on Python 3.12+
Write-Host "==> Step 1: Creating stub 'imp' module (workaround for pathtools)..." -ForegroundColor Yellow
$stubDir = Join-Path $BackendDir "venv\Lib\site-packages"
$stubPath = Join-Path $stubDir "imp.py"

$stubContent = @'
"""Stub for the deprecated `imp` module (removed in Python 3.12).

Allows legacy setup.py scripts (e.g. pathtools) to run on Python 3.12+.
Only the functions actually used by pathtools' setup.py are stubbed.
"""
import importlib.util
import importlib.machinery
import types

PY_SOURCE = 1
PY_COMPILED = 2
C_EXTENSION = 3
PKG_DIRECTORY = 5


def find_module(name, path=None):
    spec = importlib.machinery.PathFinder.find_spec(name, path)
    if spec is None:
        raise ImportError("No module named " + repr(name))
    return (None, spec.origin, ("", "", PY_SOURCE))


def load_module(name, file, pathname, description):
    return importlib.import_module(name)


def load_source(name, pathname, file=None):
    spec = importlib.util.spec_from_file_location(name, pathname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_module(name):
    return types.ModuleType(name)


def get_suffixes():
    return [(".py", "r", PY_SOURCE)]
'@

# Create the stub file
$stubContent | Out-File -FilePath $stubPath -Encoding UTF8 -Force
Write-Host "    OK  Stub imp.py created at $stubPath" -ForegroundColor Green

# Step 2: Install pathtools with --no-build-isolation so it uses our stub
Write-Host ""
Write-Host "==> Step 2: Installing pathtools (with imp stub)..." -ForegroundColor Yellow
$pathtoolsResult = & cmd /c "`"$VenvPip`" install pathtools --no-build-isolation 2>&1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK  pathtools installed" -ForegroundColor Green
} else {
    Write-Host "    WARN pathtools install had issues (continuing anyway)" -ForegroundColor Yellow
    $pathtoolsResult | ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
}

# Step 3: Install ecg_qc and its other dependencies with --no-deps
# (we install deps manually to skip pathtools which is already installed)
Write-Host ""
Write-Host "==> Step 3: Installing ecg_qc dependencies..." -ForegroundColor Yellow
$depsResult = & cmd /c "`"$VenvPip`" install biosppy py-ecg-detectors scikit-learn peakutils dill 2>&1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK  ecg_qc dependencies installed" -ForegroundColor Green
} else {
    Write-Host "    ERR  ecg_qc dependency install failed:" -ForegroundColor Red
    $depsResult | ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
    exit 1
}

# Step 4: Install ecg_qc itself with --no-deps (we already installed deps)
Write-Host ""
Write-Host "==> Step 4: Installing ecg_qc (with --no-deps)..." -ForegroundColor Yellow
$ecgqcResult = & cmd /c "`"$VenvPip`" install ecg-qc --no-deps 2>&1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK  ecg_qc installed" -ForegroundColor Green
} else {
    Write-Host "    ERR  ecg_qc install failed:" -ForegroundColor Red
    $ecgqcResult | ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
    exit 1
}

# Step 5: Also install the ecg_qc source code
# The PyPI wheel for ecg-qc 1.0b6 is empty (only metadata) - we need to
# install from GitHub source to get the actual Python code.
Write-Host ""
Write-Host "==> Step 5: Installing ecg_qc source from GitHub..." -ForegroundColor Yellow
$srcResult = & cmd /c "`"$VenvPip`" install --no-deps --force-reinstall git+https://github.com/Aura-healthcare/ecg_qc.git@main 2>&1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK  ecg_qc source installed from GitHub" -ForegroundColor Green
} else {
    Write-Host "    WARN  GitHub install failed (will try to use PyPI version)" -ForegroundColor Yellow
    $srcResult | ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
}

# Step 6: Verify the install
Write-Host ""
Write-Host "==> Step 6: Verifying installation..." -ForegroundColor Yellow
$verifyResult = & $VenvPython -c "
import warnings
warnings.filterwarnings('ignore')
try:
    from ecg_qc.sqi_computing.sqi_rr_intervals import csqi, qsqi
    from ecg_qc.sqi_computing.sqi_frequency_distribution import ssqi, ksqi
    from ecg_qc.sqi_computing.sqi_power_spectrum import bassqi, psqi
    print('OK - all SQI functions imported successfully')
    print('  qsqi, csqi, ssqi, ksqi, psqi, bassqi are all available')
except ImportError as e:
    print('FAIL - ' + str(e))
    exit(1)
" 2>&1

if ($verifyResult -match "OK") {
    Write-Host "    $verifyResult" -ForegroundColor Green
} else {
    Write-Host "    $verifyResult" -ForegroundColor Red
    Write-Host ""
    Write-Host "  ecg_qc installation did not succeed. The system will fall back to" -ForegroundColor Yellow
    Write-Host "  the simple std-based quality filter. You can still use the system" -ForegroundColor Yellow
    Write-Host "  normally - the only difference is less sophisticated quality filtering." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  ecg_qc installation complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  The system will now use the ecg_qc SQI feature extractors for" -ForegroundColor White
Write-Host "  quality filtering during training. You'll see 'ecg_qc' as the" -ForegroundColor White
Write-Host "  QC method in the Training tab." -ForegroundColor White
Write-Host ""
Write-Host "  Restart the FastAPI backend (close its window and re-run" -ForegroundColor Yellow
Write-Host "  .\start-all.ps1 -SkipSetup) to pick up the change." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
