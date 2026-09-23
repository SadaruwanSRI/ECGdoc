<#
.SYNOPSIS
    Train an ECG autoencoder model from the command line (no UI needed).

.DESCRIPTION
    Useful for the final MIT-BIH normal-sinus autoencoder run where
    you don't want to keep the browser open. The trained model is saved to
    the database and immediately available in the web UI.

.PARAMETER ModelName
    Name to give the new model version.

.PARAMETER Epochs
    Number of training epochs (default: 2).

.PARAMETER Dataset
    "mitbih-nsrdb" (default, final PhysioNet source) or "synthetic".

.PARAMETER MaxRecords
    When using mitbih-nsrdb: how many healthy subjects to use (default: 18).

.PARAMETER BatchSize
    Batch size (default: 256).

.PARAMETER LearningRate
    Adam learning rate (default: 0.001).

.PARAMETER ThresholdK
    Threshold multiplier k (default: 2.0).

.EXAMPLE
    .\train-model.ps1 -ModelName "quick-test" -Epochs 5
    Quick 5-epoch smoke test on synthetic data.

.EXAMPLE
    .\train-model.ps1 -ModelName "nsrdb-final" -Epochs 2 -Dataset mitbih-nsrdb -MaxRecords 18
    Final normal-only training using all 18 complete MIT-BIH NSRDB recordings.
#>

param(
    [string]$ModelName = "ecg-ae-$(Get-Date -Format 'yyyy-MM-dd')",
    [int]$Epochs = 2,
    [ValidateSet("synthetic", "mitbih-nsrdb")]
    [string]$Dataset = "mitbih-nsrdb",
    [int]$MaxRecords = 18,
    [int]$BatchSize = 256,
    [double]$LearningRate = 0.001,
    [double]$ThresholdK = 2.0,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $BackendDir "venv\Scripts\python.exe"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  ECG Autoencoder Training" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Model name:    $ModelName"
Write-Host "  Dataset:       $Dataset"
if ($Dataset -eq "mitbih-nsrdb") {
    Write-Host "  Records:       $MaxRecords"
}
Write-Host "  Epochs:        $Epochs"
Write-Host "  Batch size:    $BatchSize"
Write-Host "  Learning rate: $LearningRate"
Write-Host "  Threshold k:   $ThresholdK"
Write-Host "================================================================"
Write-Host ""

# Check venv exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "Python venv not found. Run .\start-all.ps1 first to set up the environment." -ForegroundColor Red
    exit 1
}

# Write the inline Python training script to a temp file
$tempScript = Join-Path $env:TEMP "ecg-train-model.py"
$pythonCode = @"
import sys, os, json, time, secrets, threading
sys.path.insert(0, r'$BackendDir')
os.chdir(r'$BackendDir')

# Load .env
from pathlib import Path
env_file = Path(r'$ProjectRoot') / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

from app.core.config import settings
from app.db.session import get_db
from app.ml.training import TrainConfig, train
from sqlalchemy import text

print(f'Backend dir: {settings.BASE_DIR}')
print(f'Database:    {settings.SQLALCHEMY_URL}')
print(f'Storage:     {settings.MODEL_DIR}')
print()

model_id = secrets.token_hex(12)
version = f'v{int(time.time())}'
config = {
    'model_name': '$ModelName',
    'epochs': $Epochs,
    'batch_size': $BatchSize,
    'learning_rate': $LearningRate,
    'dataset_name': '$Dataset',
    'max_records': $MaxRecords,
    'threshold_k': $ThresholdK,
}

with get_db() as db:
    db.execute(text('''
        INSERT INTO ModelVersion
            (id, name, version, description, architecture, parameters, latentDim,
             compressionRatio, status, modelPath, threshold, thresholdK,
             configJson, createdAt, updatedAt)
        VALUES
            (:id, :name, :version, :desc, '1D-Conv-Autoencoder', 9758209, 1024,
             32.0, 'training', NULL, NULL, :k, :cfg,
             datetime('now'), datetime('now'))
    '''), {
        'id': model_id, 'name': '$ModelName', 'version': version,
        'desc': 'Trained via train-model.ps1', 'k': $ThresholdK,
        'cfg': json.dumps(config),
    })
    db.commit()

print(f'Created ModelVersion row: {model_id}')
print()
print('Starting training...')
print()

cfg = TrainConfig(
    epochs=$Epochs,
    batch_size=$BatchSize,
    learning_rate=$LearningRate,
    dataset_name='$Dataset',
    max_records=$MaxRecords,
    threshold_k=$ThresholdK,
    use_ecg_qc=True,
    min_quality=2,
)

def on_progress(ev):
    et = ev.get('type')
    if et == 'qc':
        if 'distribution' in ev:
            print(f'  [qc]   {ev["n_kept"]}/{ev["n_in"]} windows kept ({ev.get("retention_pct",0):.1f}%)')
            print(f'         quality distribution: {ev["distribution"]}')
        else:
            print(f'  [qc]   {ev["n_kept"]} windows kept ({ev.get("lead", ev.get("qc_method", "verified"))})')
    elif et == 'init':
        print(f'  [init] Train windows: {ev["n_train"]}, Val windows: {ev["n_val"]}')
        print(f'  [init] Architecture: {ev["architecture"]["name"]} ({ev["architecture"]["total_parameters"]:,} params)')
        print(f'  [init] Device: {ev["device"]}')
        print()
    elif et == 'epoch':
        print(f'  [epoch {ev["epoch"]}/{ev["total_epochs"]}] train_loss={ev["train_loss"]:.6f}  val_loss={ev["val_loss"]:.6f}  ({ev["elapsed_s"]:.1f}s)')
    elif et == 'batch':
        if ev['batch'] % 10 == 0 or ev['batch'] == ev['total_batches']:
            print(f'    batch {ev["batch"]}/{ev["total_batches"]}  loss={ev["batch_loss"]:.6f}')

stop_flag = threading.Event()
result = train(cfg, on_progress=on_progress, should_stop=lambda: stop_flag.is_set())

print()
print('=' * 60)
print('  TRAINING COMPLETE')
print('=' * 60)
print(f'  Final train loss: {result.final_loss:.6f}')
print(f'  Best val loss:    {result.val_loss:.6f}')
print(f'  Threshold (tau):  {result.threshold:.6f}')
print(f'  Epochs run:       {result.epochs_run}')
print(f'  Model file:       {result.model_path}')
print()

with get_db() as db:
    db.execute(text('''
        UPDATE ModelVersion
        SET status = 'ready', modelPath = :mp, threshold = :thr,
            thresholdK = :k, updatedAt = datetime('now')
        WHERE id = :id
    '''), {
        'mp': result.model_path, 'thr': result.threshold,
        'k': result.threshold_k, 'id': model_id,
    })
    for h in result.history:
        for metric_name, value in (('train_loss', h['train_loss']), ('val_loss', h['val_loss'])):
            db.execute(text('''
                INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                VALUES (:id, :mid, :mn, :mv, :ep, datetime('now'))
            '''), {
                'id': secrets.token_hex(12), 'mid': model_id, 'mn': metric_name,
                'mv': value, 'ep': h['epoch'],
            })
    db.execute(text('''
        INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
        VALUES (:id, :mid, 'threshold', :v, NULL, datetime('now'))
    '''), {'id': secrets.token_hex(12), 'mid': model_id, 'v': result.threshold})
    db.execute(text('''
        INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
        VALUES (:id, :mid, 'final_val_loss', :v, NULL, datetime('now'))
    '''), {'id': secrets.token_hex(12), 'mid': model_id, 'v': result.val_loss})
    db.commit()

print(f'Model ID: {model_id}')
print(f'Version:  {version}')
print()
print('The model is now available in the web UI - start the stack with:')
print(f'  .\start-all.ps1 -SkipSetup')
"@

$pythonCode | Out-File -FilePath $tempScript -Encoding UTF8

Write-Host "Launching training..." -ForegroundColor Yellow
Write-Host ""
& $VenvPython $tempScript

# Cleanup
Remove-Item $tempScript -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
