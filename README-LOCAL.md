# ECG Anomaly Detection System - Local Guide and Architecture

This project is a local ECG research platform. It includes the original
normal-only autoencoder and the final beat-aligned hierarchical system. The
final system combines morphology, spectrum and RR timing for abnormal-beat
detection and supported-subtype classification using one ECG lead. It
streams ECG sessions, raises per-beat alerts, and generates reviewable PDF
reports. It is not a clinical diagnostic device.

The system has three main services:

1. Next.js frontend on `http://localhost:3000`
2. FastAPI backend on `http://localhost:8000` by default, or the next free port if `8000` is occupied
3. Socket.io WebSocket service on `http://localhost:3003`

Use `run-app.ps1` on Windows to install dependencies, initialize the database, and start the full stack.

---

## 1. Main Objective

The objective is to detect abnormal ECG patterns in near real time and support review by clinicians or researchers.

The recommended final-system workflow is:

1. Start the stack; the shipped `Final MLII Temporal-Holdout ECG System` is registered
   automatically when its model artifact is present.
2. Select an autoencoder as the legacy waveform model and select the final
   hierarchy as the detection and rhythm model.
3. Start a session from an eligible MIT-BIH MLII replay, an MLII-like
   synthetic stream, or an Arduino connected with the MLII torso placement.
4. Buffer the stream, detect R peaks, and wait until the next RR interval and
   complete 512-sample R-centred window are available.
5. Calculate morphology, spectrum and RR features and compare abnormal
   probability with the saved threshold `0.620833`.
6. Use the same 242-value MLII feature vector for abnormality detection and
   supported-subtype classification.
7. Store true beat counts, per-beat results, alerts, waveforms, RR context and
   session metadata.
8. Generate a PDF report that clearly labels probability-based and legacy
   reconstruction-based results.

---

## 2. Prerequisites

Install these tools before the first run:

| Tool | Required version | Purpose |
| --- | --- | --- |
| Python | 3.10+ | FastAPI backend and PyTorch training |
| Bun | 1.0+ preferred | Frontend and WebSocket dependency install |
| Node.js | 18+ fallback | Used if Bun is unavailable |
| Git | Any recent version | Source control |
| PostgreSQL | Optional | Production or multi-user database |

Check your tools:

```powershell
python --version
bun --version
node --version
git --version
```

If PowerShell blocks scripts, run this in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

---

## 3. Start the System with `run-app.ps1`

Open PowerShell in the project root:

```powershell
cd D:\SANDARUWAN\Research\ECGdoc
```

First-time startup:

```powershell
.\run-app.ps1
```

This single script performs the full setup and launch:

1. Checks required tools.
2. Creates `backend\venv\`.
3. Installs Python dependencies from `backend\requirements.txt`.
4. Installs frontend dependencies.
5. Initializes the database.
6. Starts the backend, frontend, and WebSocket service.
7. Writes logs to `runtime\logs`.
8. Starts the backend, frontend, and WebSocket service.

For later runs, skip setup:

```powershell
.\run-app.ps1 -SkipSetup
```

By default, the launcher opens a separate PowerShell window for the frontend,
backend, and WebSocket service so their logs are visible live. The same output
is also saved under `runtime\logs`. To run all services silently in the
background instead, add `-HiddenServices`:

```powershell
.\run-app.ps1 -SkipSetup -HiddenServices
```

Restart the full stack:

```powershell
.\run-app.ps1 -Restart -SkipSetup
```

To use PostgreSQL instead of SQLite:

```powershell
psql -U postgres -f setup-postgres.sql
.\run-app.ps1 -UsePostgres
```

Stop all services:

```powershell
.\run-app.ps1 -Stop
```

Default login:

```text
Email: demo@ecg.local
Password: demo1234
```

---

## 4. Important URLs

| URL | Purpose |
| --- | --- |
| `http://localhost:3000` | Main web application |
| `http://localhost:8000/docs` | FastAPI Swagger API docs, unless the runner selects a fallback backend port |
| `http://localhost:8000/health` | Backend health check, unless the runner selects a fallback backend port |
| `http://localhost:3003` | Socket.io WebSocket service |

---

## 5. Project Structure

```text
ECGdoc/
|-- run-app.ps1                    # Single Windows launcher/status/stop script
|-- package.json                   # Frontend scripts and dependencies
|-- prisma/
|   |-- schema.prisma              # Database schema
|   `-- dev.db                     # SQLite database when using default setup
|-- src/
|   |-- app/                       # Next.js app router
|   |-- components/                # UI tabs and shared components
|   |-- hooks/                     # React hooks
|   `-- lib/                       # API client and auth context
|-- backend/
|   |-- requirements.txt           # Python dependencies
|   |-- app/
|   |   |-- main.py                # FastAPI app entry point
|   |   |-- api/                   # REST endpoints
|   |   |-- core/                  # Config and auth
|   |   |-- db/                    # SQLAlchemy session
|   |   |-- ml/                    # Data, models, training, inference
|   |   |-- services/              # PDF report generation
|   |   `-- storage/
|   |       |-- datasets/          # Local datasets and downloaded PhysioNet cache
|   |       |-- models/            # Saved `.pt` model checkpoints
|   |       `-- reports/           # Generated plots and PDF reports
|-- mini-services/
|   `-- ws-service/                # Socket.io bridge for live updates
|-- db/                            # Optional local database files
|-- docs/
|   |-- app-screenshots/           # Screenshots and generated example reports
|   `-- uploads/                   # Manually uploaded research artifacts
|-- runtime/
|   `-- logs/                      # Local launcher and service logs
|-- scripts/
|   `-- legacy/                    # Old launch/helper scripts kept for reference
`-- README-LOCAL.md                # This document
```

---

## 6. Backend ML Files

The main ML code is in `backend/app/ml/`.

| File | Purpose |
| --- | --- |
| `data.py` | ECG loading, preprocessing, windowing, synthetic data, MIT-BIH download/cache, quality control |
| `model.py` | Model 1: dilated U-Net autoencoder for anomaly detection |
| `training.py` | Autoencoder training loop and threshold calibration |
| `classifier.py` | Model 2: arrhythmia classifier architecture |
| `train_classifier.py` | Classifier training loop using MIT-BIH annotations |
| `inference.py` | Runtime model loading, reconstruction scoring, and classification |
| `ecg_metrics.py` | ECG vital metrics such as BPM and intervals |
| `health_info.py` | Health interpretation helpers |

---

## 7. ECG Data Pipeline

All ECG data is normalized into the same format before training or inference.

Pipeline:

```text
Raw ECG
  -> resample to 128 Hz
  -> bandpass filter 0.5-50 Hz
  -> z-score normalization
  -> split into 4 second windows
  -> each window has 512 samples
```

Key constants are in `backend/app/core/config.py`:

```python
SAMPLING_RATE_HZ = 128
WINDOW_SECONDS = 4.0
WINDOW_SAMPLES = 512
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 50.0
DEFAULT_THRESHOLD_K = 2.0
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
```

Supported training data sources:

1. Synthetic normal ECG.
2. MIT-BIH Normal Sinus Rhythm Database (all 18 complete records, first stored ECG channel).
3. Uploaded local WFDB or CSV datasets.

Supported live/testing sources:

1. MIT-BIH Arrhythmia Database replay.
2. Synthetic arrhythmia ECG.
3. Simulated Arduino ECG stream.

Local datasets are stored under:

```text
backend/app/storage/datasets/
```

This folder is intentionally ignored by Git because datasets can become large.

---

## 8. Model 1: Anomaly Detection Autoencoder

Model file:

```text
backend/app/ml/model.py
```

Training file:

```text
backend/app/ml/training.py
```

Model 1 is an unsupervised reconstruction model. It learns how normal ECG should look. During inference, abnormal ECG should reconstruct poorly, producing a higher reconstruction error.

Architecture summary:

```text
Input:  (batch, 1, 512)
Encoder:
  1 -> 32 -> 64 -> 128 -> 256 channels
  residual Conv1D blocks
  stride-2 downsampling
Bottleneck:
  dilated Conv1D blocks with dilation 1, 2, 4, 8
Decoder:
  U-Net style upsampling with skip connections
Output:
  (batch, 1, 512)
```

Important design choices:

1. GroupNorm is used instead of BatchNorm for stability with small batches.
2. LeakyReLU preserves negative ECG values.
3. The output layer is linear, because inputs are z-score normalized and can be negative.
4. Skip connections are scaled for new anomaly models to reduce direct copying of abnormal morphology.
5. Training uses optional denoising noise so the model becomes more robust to sensor noise.

The anomaly score is mean absolute reconstruction error:

```text
score = mean(abs(input_window - reconstructed_window))
```

An anomaly is detected when:

```text
score > threshold
```

The threshold is calculated from validation normal windows using robust statistics:

```text
threshold = max(median_error + threshold_k * MAD_sigma, 95th_percentile_error)
```

This is more stable than a simple mean-plus-standard-deviation threshold when the validation set contains outliers.

---

## 9. How to Modify the Anomaly Detection Model

Edit:

```text
backend/app/ml/model.py
backend/app/ml/training.py
backend/app/ml/inference.py
```

Common changes:

### Change encoder/decoder size

In `model.py`, modify:

```python
ENCODER_CHANNELS = [1, 32, 64, 128, 256]
DECODER_CHANNELS = [128, 64, 32, 32]
```

Rules:

1. `ENCODER_CHANNELS[0]` must stay `1` for single-lead ECG.
2. Decoder channel sizes must match the upsampling path.
3. If you change the number of encoder stages, update the `enc*`, `up*`, `encode`, and `decode` sections together.
4. Old `.pt` checkpoints may not load if the layer shapes change.

### Change convolution kernel size

In `model.py`:

```python
KERNEL_SIZE = 7
```

Larger kernels capture wider morphology but increase compute. Smaller kernels train faster but may miss wider QRS/T-wave shape.

### Change bottleneck rhythm context

In `model.py`:

```python
DILATIONS = [1, 2, 4, 8]
```

Larger dilation values give more temporal context. This can help rhythm abnormalities but may make training slower.

### Change skip-copy behavior

In `model.py`:

```python
DEFAULT_SKIP_SCALE = 1.0
DENOISING_SKIP_SCALE = 0.5
```

For anomaly detection, too much skip copying can allow the model to reconstruct abnormal beats too well. Lower `DENOISING_SKIP_SCALE` makes reconstruction stricter, which can improve anomaly separation, but too low may hurt normal reconstruction.

Recommended experiments:

```text
0.25, 0.5, 0.75, 1.0
```

### Change training noise

In `training.py`:

```python
input_noise_std = 0.03
```

Higher noise improves robustness but can blur fine ECG morphology. Try:

```text
0.00, 0.01, 0.03, 0.05
```

### Change threshold sensitivity

From the UI, adjust `Threshold k`.

Lower values detect more anomalies but may increase false positives.

Higher values reduce false positives but may miss subtle arrhythmias.

Recommended values:

```text
1.5 = sensitive
2.0 = balanced
2.5 = conservative
3.0 = very conservative
```

After any architecture change, train a new model. Do not expect old model checkpoints to work if layer shapes changed.

---

## 10. Model 2: Arrhythmia Classification Model

Model file:

```text
backend/app/ml/classifier.py
```

Training file:

```text
backend/app/ml/train_classifier.py
```

Model 2 classifies anomalous or selected ECG windows into arrhythmia categories.

Current classes:

```python
ARRHYTHMIA_CLASSES = ["N", "PVC", "PAC", "LBBB", "RBBB", "AFib"]
```

Human-readable labels:

```text
N    = Normal Sinus Rhythm
PVC  = Premature Ventricular Contraction
PAC  = Premature Atrial Contraction
LBBB = Left Bundle Branch Block
RBBB = Right Bundle Branch Block
AFib = Atrial Fibrillation
```

Classifier architecture:

```text
Input ECG window
  -> frozen autoencoder encoder
  -> latent representation
  -> average temporal pooling
  -> max temporal pooling
  -> dense layer
  -> BatchNorm
  -> GELU
  -> Dropout
  -> dense output logits
```

Training data comes from MIT-BIH Arrhythmia Database annotations:

1. Beat annotations map symbols to classes.
2. AFib is identified from rhythm annotations.
3. Data is split by complete patient, not randomly by beat. MIT-BIH records 201 and 202 are kept in the same patient group.
4. The checkpoint stores the exact train, validation, and held-out test patient lists, and the UI displays the test patients beside the metrics.
5. Class weighting and weighted sampling help handle class imbalance.

---

## 11. How to Modify the Classification Model

Edit:

```text
backend/app/ml/classifier.py
backend/app/ml/train_classifier.py
backend/app/ml/inference.py
```

### Add or remove classes

In `classifier.py`, update:

```python
ARRHYTHMIA_CLASSES = ["N", "PVC", "PAC", "LBBB", "RBBB", "AFib"]
ARRHYTHMIA_NAMES = {...}
MITBIH_TO_CLASS = {...}
```

Then check:

1. The new class exists in `ARRHYTHMIA_NAMES`.
2. MIT-BIH annotation symbols map correctly in `MITBIH_TO_CLASS`.
3. `num_classes` matches `len(ARRHYTHMIA_CLASSES)`.
4. Any saved old classifier checkpoint will not match if the class count changes.

### Change the classification head

In `classifier.py`, modify `ClassificationHead`.

Useful experiments:

1. Increase `hidden_dim` from `128` to `256`.
2. Increase dropout from `0.3` to `0.4` if overfitting.
3. Add a second dense layer if training accuracy is low.
4. Use fine-tuning instead of a fully frozen encoder.

Example for fine-tuning the encoder:

```python
for param in self.autoencoder.parameters():
    param.requires_grad = True
```

If you fine-tune the encoder, reduce the learning rate:

```text
1e-4 or 5e-5
```

### Modify beat extraction

In `train_classifier.py`, edit:

```python
beat_window_samples = 512
r_peak_before = 200
```

`r_peak_before` controls where the R peak appears in the window. A value of `200` means the model sees context before and after the beat.

### Improve classifier accuracy

Recommended training changes:

1. Use all 46 MIT-BIH records that contain MLII; never mix V5/V2 records.
2. Train for 30-100 epochs.
3. Check per-class F1, not only overall accuracy.
4. Inspect the confusion matrix for PVC/PAC confusion.
5. Do not use random beat-level splits for final reporting; use record-wise splits.

Recommended UI settings:

```text
Epochs: 30-100
Batch size: 64
Learning rate: 0.001 for head-only training
Max MLII records: 46
```

---

## 12. Training Workflow

### Train the anomaly detection model from the UI

1. Start the stack:

   ```powershell
   .\run-app.ps1 -SkipSetup
   ```

2. Open:

   ```text
   http://localhost:3000
   ```

3. Log in.
4. Go to `Train Model`.
5. Choose the dataset:
   - `synthetic` for quick testing.
   - `mitbih-nsrdb` for all 18 real MIT-BIH normal-sinus subjects.
   - `uploaded` for your own dataset.
6. Recommended real training settings:

   ```text
   Dataset: MIT-BIH Normal Sinus Rhythm Database
   Max records: 18
   Duration per record: 20 seconds
   Epochs: 8
   Batch size: 256
   Learning rate: 0.001
   Threshold k: 2.0
   Use ECG QC: true
   Min quality: 2
   ```

7. Click `Start Training`.
8. Wait until the model status becomes `ready`.

### Train the anomaly detection model from CLI

```powershell
.\scripts\legacy\train-model.ps1 -ModelName "ecg-nsrdb" -Epochs 2 -Dataset mitbih-nsrdb -MaxRecords 18
```

### Train the classification model

1. First train or select a ready anomaly model.
2. In the Training tab, choose that anomaly model as the encoder.
3. Start classifier training.

Recommended classifier settings:

```text
Epochs: 30
Batch size: 64
Learning rate: 0.001
Max MLII records: 46
```

Classifier training saves a separate `.pt` file in:

```text
backend/app/storage/models/
```

---

## 13. Live Analysis Workflow

1. Start the system:

   ```powershell
   .\run-app.ps1 -SkipSetup
   ```

2. Open `http://localhost:3000`.
3. Go to `Live Analysis`.
4. Select a ready anomaly model.
5. Optionally select a ready classifier model.
6. Select a source:
   - MIT-BIH Arrhythmia replay.
   - Synthetic arrhythmia.
   - Arduino simulated stream.
7. Click `Start Live Analysis`.

For every ECG chunk:

```text
source chunk
  -> preprocessing
  -> sliding 512-sample windows
  -> autoencoder reconstruction
  -> anomaly score
  -> threshold comparison
  -> optional classifier
  -> alert creation
  -> database persistence
  -> WebSocket update to UI
```

The UI displays:

1. ECG waveform.
2. Reconstructed waveform.
3. Anomaly score.
4. Alerts.
5. Classification label and confidence when classifier is loaded.
6. ECG metrics such as heart rate.

---

## 14. Report Generation Workflow

1. Run a live session.
2. Stop the session.
3. Go to `Sessions`.
4. Open the session details.
5. Click `Generate Report`.
6. Add physician name and notes.
7. Download the generated PDF.

Reports are stored under:

```text
backend/app/storage/reports/
```

The report includes:

1. Session metadata.
2. The number of unique alerted beats after duplicate R-peak alerts are removed.
3. A model-output summary with abnormal probability, decision threshold,
   subtype confidence, RR interval and single-lead mode.
4. One representative ECG waveform for each distinct predicted anomaly type;
   repeated occurrences are counted but the same type is not plotted again.
5. A red R-peak marker and subtype-probability bars on each anomaly example.
6. Clinical summary.
7. Physician notes and sign-off section.

---

## 15. Database Structure

Database schema:

```text
prisma/schema.prisma
```

Main tables:

| Table | Purpose |
| --- | --- |
| `User` | Login users |
| `ModelVersion` | Saved anomaly and classifier model metadata |
| `ModelMetric` | Training metrics and evaluation metrics |
| `TrainingRun` | Training execution status and logs |
| `EcgSession` | Live or recorded ECG monitoring sessions |
| `Alert` | Anomaly alerts for sessions |
| `EcgDataPoint` | Downsampled ECG points for replay/reporting |

SQLite is the default. The database URL is controlled by `.env`:

```text
DATABASE_URL="file:./dev.db"
```

For PostgreSQL:

```text
DATABASE_URL="postgresql://ecg_user:ecg_password@localhost:5432/ecg_anomaly"
```

---

## 16. API Structure

Backend API files:

```text
backend/app/api/
```

Important route files:

| File | Purpose |
| --- | --- |
| `routes_auth.py` | Login, registration, auth |
| `routes_datasets.py` | Dataset listing and upload |
| `routes_models.py` | Model listing, comparison, deletion |
| `routes_training.py` | Start/stop/list training runs |
| `routes_sessions.py` | Start/stop/live chunks/session data |
| `routes_reports.py` | PDF report generation |

Interactive API docs:

```text
http://localhost:8000/docs
```

---

## 17. Frontend Structure

Main frontend files:

```text
src/app/page.tsx
src/components/dashboard-tab.tsx
src/components/training-tab.tsx
src/components/live-tab.tsx
src/components/sessions-tab.tsx
src/components/ecg-canvas.tsx
src/lib/api.ts
src/lib/auth-context.tsx
```

Tabs:

1. Dashboard: model comparison and summary.
2. Train Model: anomaly/classifier training.
3. Live Analysis: live ECG streaming and alerts.
4. Sessions: historical sessions and PDF reports.

---

## 18. How to Add a Real Arduino ECG Source

The current Arduino source is simulated in:

```text
backend/app/api/routes_sessions.py
```

Find:

```python
elif req.source_type == "arduino":
```

Replace the simulated generator with serial reads.

Example:

```python
import serial

ser = serial.Serial("COM3", 115200)

def arduino_stream():
    while True:
        chunk = []
        for _ in range(512):
            line = ser.readline().strip()
            if line:
                chunk.append(float(line))
        yield np.array(chunk, dtype=np.float32)

source_iter = arduino_stream()
```

Arduino sketch example:

```cpp
void setup() {
  Serial.begin(115200);
}

void loop() {
  int ecg = analogRead(A0);
  Serial.println(ecg);
  delay(8); // about 125 Hz
}
```

If your hardware samples at a different rate, pass the correct source sampling rate or resample before yielding chunks.

---

## 19. Validation and Accuracy Checklist

Use this checklist before reporting model accuracy:

1. Train anomaly detection on normal ECG only.
2. Validate anomaly detection on records not used for training.
3. Tune `threshold_k` using validation data.
4. Report false positives and false negatives, not only loss.
5. Train and test the classifier using disjoint patient groups; keep MIT-BIH records 201 and 202 together.
6. Report per-class precision, recall, and F1.
7. Inspect confusion matrix.
8. Keep dataset versions and model checkpoints.
9. Do not evaluate on the same records used for training.
10. Test live inference with both synthetic arrhythmia and MIT-BIH arrhythmia replay.

Useful metrics:

```text
Anomaly model:
- validation reconstruction loss
- threshold
- normal false positive rate
- abnormal detection rate

Classification model:
- test accuracy
- per-class precision
- per-class recall
- per-class F1
- confusion matrix
```

---

## 20. Common Commands

Install/start everything:

```powershell
.\run-app.ps1
```

Start after setup:

```powershell
.\run-app.ps1 -SkipSetup
```

Restart services:

```powershell
.\run-app.ps1 -Restart -SkipSetup
```

Stop services:

```powershell
.\run-app.ps1 -Stop
```

Train anomaly model from CLI:

```powershell
.\scripts\legacy\train-model.ps1 -ModelName "ecg-nsrdb" -Epochs 2 -Dataset mitbih-nsrdb -MaxRecords 18
```

Run frontend lint:

```powershell
npm.cmd run lint
```

Run frontend production build:

```powershell
npm.cmd run build
```

Compile backend Python files:

```powershell
python -m compileall backend\app
```

---

## 21. Troubleshooting

### PowerShell script execution is blocked

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then retry:

```powershell
.\run-app.ps1
```

### Port already in use

Stop existing services:

```powershell
.\run-app.ps1 -Stop
```

Then start again:

```powershell
.\run-app.ps1 -SkipSetup
```

### Model does not appear in Live Analysis

Only models with status `ready` appear. Check:

1. Training tab status.
2. Backend logs.
3. `ModelVersion` row in the database.

### Training fails with not enough windows

Try:

1. Increase `Max records`.
2. Increase `Duration per record`.
3. Lower `Min quality` from `2` to `1`.
4. Use synthetic data to confirm the pipeline works.

### PhysioNet download fails

The system first checks local cache in:

```text
backend/app/storage/datasets/
```

You can manually place WFDB `.dat` and `.hea` files there.

Example:

```text
backend/app/storage/datasets/mitdb/100.dat
backend/app/storage/datasets/mitdb/100.hea
backend/app/storage/datasets/nsrdb-primary-full/16265_primary_full.npy
```

### `ecg_qc` is unavailable

This is acceptable. The system falls back to simple standard-deviation quality control.

To retry optional ECG QC install:

```powershell
.\scripts\legacy\install-ecg-qc.ps1
```

### Frontend build tries to download fonts

The app now uses local system font stacks, so `npm.cmd run build` should work without fetching Google Fonts.

### PyTorch is missing

Install backend dependencies through the launcher:

```powershell
.\run-app.ps1
```

Or manually:

```powershell
backend\venv\Scripts\activate
pip install -r backend\requirements.txt
```

---

## 22. Development Notes

Generated files should not be committed:

```text
backend/app/**/__pycache__/
backend/app/storage/datasets/
backend/app/storage/models/
backend/app/storage/reports/
backend/venv/
.next/
node_modules/
```

Large datasets and model checkpoints should stay local unless you intentionally version them with an artifact system.

When changing model architecture:

1. Update the model code.
2. Train a new checkpoint.
3. Verify inference can load the checkpoint.
4. Run a live session.
5. Compare metrics against the previous version.
6. Document the experiment settings.

---

## 23. Recommended Research Experiment Plan

For anomaly detection:

1. Train baseline with synthetic data.
2. Train the real model on complete MIT-BIH normal-sinus recordings.
3. Compare skip scales: `0.25`, `0.5`, `0.75`, `1.0`.
4. Compare threshold values: `1.5`, `2.0`, `2.5`, `3.0`.
5. Test on MIT-BIH Arrhythmia records.
6. Record false positives and true detections.

For classification:

1. Train with all 46 eligible MIT-BIH MLII records.
2. Use record-wise train/val/test split.
3. Report confusion matrix.
4. Focus on per-class F1 because classes are imbalanced.
5. Tune hidden dimension, dropout, and learning rate.

Recommended final report metrics:

```text
Anomaly detection:
- threshold
- validation loss
- detection rate on arrhythmia sessions
- false positive rate on normal sessions

Classification:
- test accuracy
- macro F1
- weighted F1
- per-class F1
- confusion matrix
```
