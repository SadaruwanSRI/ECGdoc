# Actuarial Dev - Daily Startup Script
# Run from: D:\SANDARUWAN\actuarial_dev

 $ErrorActionPreference = "Continue"
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Actuarial Dev - Daily Startup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. PostgreSQL
Write-Host "[1/5] PostgreSQL..." -ForegroundColor Yellow
 $pg = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq "Running" }
if ($pg) {
    Write-Host "      OK - PostgreSQL is running" -ForegroundColor Green
} else {
    Write-Host "      WARN - PostgreSQL not detected as a Windows service." -ForegroundColor DarkYellow
    Write-Host "             Start it manually if needed." -ForegroundColor DarkYellow
}

# 2. MinIO
Write-Host "[2/5] MinIO..." -ForegroundColor Yellow
 $minioProc = Get-Process -Name "minio" -ErrorAction SilentlyContinue
if ($minioProc) {
    Write-Host "      OK - MinIO already running (PID $($minioProc.Id))" -ForegroundColor Green
} else {
    Write-Host "      Starting MinIO..." -ForegroundColor Gray
    $env:MINIO_ROOT_USER = "actuarialminio"
    $env:MINIO_ROOT_PASSWORD = "change-me-strong-minio-password"
    $minioExe = "C:\Users\hp\AppData\Local\Microsoft\WinGet\Packages\MinIO.Server_Microsoft.Winget.Source_8wekyb3d8bbwe\minio.exe"
    Start-Process -FilePath $minioExe -ArgumentList "server","D:\SANDARUWAN\actuarial_dev\minio_data","--console-address",":9001"
    Start-Sleep -Seconds 4

    # Ensure buckets exist using a helper python script
    $pyExe = "D:\SANDARUWAN\actuarial_dev\.venv\Scripts\python.exe"
    $bucketScript = "D:\SANDARUWAN\actuarial_dev\minio_data\ensure_buckets.py"
    Set-Content -Path $bucketScript -Value "from minio import Minio"
    Add-Content -Path $bucketScript -Value "c = Minio('localhost:9000', access_key='actuarialminio', secret_key='change-me-strong-minio-password', secure=False)"
    Add-Content -Path $bucketScript -Value "for b in ['cashflow-artifacts', 'models']:"
    Add-Content -Path $bucketScript -Value "    if not c.bucket_exists(b):"
    Add-Content -Path $bucketScript -Value "        c.make_bucket(b)"
    Add-Content -Path $bucketScript -Value "        print(f'  Created bucket: {b}')"
    Add-Content -Path $bucketScript -Value "    else:"
    Add-Content -Path $bucketScript -Value "        print(f'  Bucket exists: {b}')"
    & $pyExe $bucketScript
    Write-Host "      OK - MinIO started" -ForegroundColor Green
}

# 3. Kafka + Zookeeper (Docker)
Write-Host "[3/5] Kafka + Zookeeper (Docker)..." -ForegroundColor Yellow
 $kafkaRunning = docker compose -f "D:\SANDARUWAN\actuarial_dev\actuarial_backend\docker-compose.kafka.yml" ps --status running 2>$null | Select-String "kafka"
if ($kafkaRunning) {
    Write-Host "      OK - Kafka already running" -ForegroundColor Green
} else {
    Write-Host "      Starting Kafka + Zookeeper..." -ForegroundColor Gray
    docker compose -f "D:\SANDARUWAN\actuarial_dev\actuarial_backend\docker-compose.kafka.yml" up -d
    Write-Host "      OK - Kafka started" -ForegroundColor Green
}

# 4. Backend (FastAPI)
Write-Host "[4/5] Backend (FastAPI on port 8000)..." -ForegroundColor Yellow
 $backendProc = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($backendProc) {
    Write-Host "      OK - Backend already running on port 8000" -ForegroundColor Green
} else {
    Write-Host "      Starting backend..." -ForegroundColor Gray
    $backendDir = "D:\SANDARUWAN\actuarial_dev\actuarial_backend"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command","cd '$backendDir'; & 'D:\SANDARUWAN\actuarial_dev\.venv\Scripts\Activate.ps1'; & 'D:\SANDARUWAN\actuarial_dev\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
    Write-Host "      OK - Backend starting in new terminal" -ForegroundColor Green
}

# 5. Frontend (Next.js)
Write-Host "[5/5] Frontend (Next.js on port 8005)..." -ForegroundColor Yellow
 $frontendProc = Get-NetTCPConnection -LocalPort 8005 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($frontendProc) {
    Write-Host "      OK - Frontend already running on port 8005" -ForegroundColor Green
} else {
    Write-Host "      Starting frontend..." -ForegroundColor Gray
    $frontendDir = "D:\SANDARUWAN\actuarial_dev\actuarial-frontend"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command","cd '$frontendDir'; .\node_modules\.bin\next dev -H 0.0.0.0 -p 8005"
    Write-Host "      OK - Frontend starting in new terminal" -ForegroundColor Green
}

# Done
Start-Sleep -Seconds 2
Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  All services started!" -ForegroundColor Green
Write-Host "  Frontend:  http://localhost:8005" -ForegroundColor White
Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "  MinIO:     http://localhost:9001 (console)" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""