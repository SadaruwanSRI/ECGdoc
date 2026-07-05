"""Training loop with live progress callbacks.

Designed to be called from the API layer so that progress events can be streamed
back to the frontend via WebSocket (training:progress events).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import settings
from app.ml.model import DENOISING_SKIP_SCALE, build_model
from app.ml.data import (
    synthetic_normal_ecg,
    load_mitbih_nsr,
    load_uploaded_dataset,
    windows_to_tensor,
    windowize,
    ecg_qc_quality_control,
    quality_control,
)


ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class TrainConfig:
    epochs: int = settings.DEFAULT_EPOCHS
    batch_size: int = settings.DEFAULT_BATCH_SIZE
    learning_rate: float = settings.DEFAULT_LEARNING_RATE
    latent_channels: int = 1024
    kernel_size: int = 7
    dropout: float = 0.2
    dataset_name: str = "synthetic"  # "synthetic" | "mit-bih-nsr" | "uploaded"
    uploaded_dataset_id: str = ""    # name of uploaded dataset folder (when dataset_name="uploaded")
    max_records: int = 4              # used for mit-bih-nsr and uploaded
    duration_per_record: float = 60.0
    train_split: float = 0.85
    seed: int = 42
    device: str = "cpu"
    threshold_k: float = settings.DEFAULT_THRESHOLD_K
    # Quality control settings
    use_ecg_qc: bool = True           # use ecg_qc SQI-based filter (recommended)
    min_quality: int = 2              # minimum quality class to keep (0-3; 2 = medium-high or better)
    skip_scale: float = DENOISING_SKIP_SCALE
    input_noise_std: float = 0.03


@dataclass
class TrainResult:
    final_loss: float
    val_loss: float
    epochs_run: int
    threshold: float
    threshold_k: float
    model_path: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    architecture: Dict[str, Any] = field(default_factory=dict)


def _gather_training_windows(cfg: TrainConfig,
                              on_progress: Optional[ProgressCallback] = None
                              ) -> np.ndarray:
    """Collect (N, 512) normal ECG windows from the chosen source.

    Applies the ecg_qc SQI-based quality classifier when `cfg.use_ecg_qc`
    is True (default), falling back to the simple std-based filter if
    `ecg_qc` is not installed or `cfg.use_ecg_qc` is False.

    Emits a 'qc' progress event with the quality distribution so the UI
    can show how many windows were kept/dropped.
    """
    qc_stats = None
    if cfg.dataset_name == "mit-bih-nsr":
        try:
            windows = load_mitbih_nsr(
                max_records=cfg.max_records,
                duration_s_per_record=cfg.duration_per_record,
                use_ecg_qc=cfg.use_ecg_qc,
                min_quality=cfg.min_quality,
            )
            if len(windows) > 0:
                print(f"[train] Loaded {len(windows)} clean windows from MIT-BIH NSR DB")
                if on_progress:
                    on_progress({
                        "type": "qc",
                        "dataset": "mit-bih-nsr",
                        "n_kept": len(windows),
                        "qc_method": "ecg_qc" if cfg.use_ecg_qc else "simple_std",
                        "min_quality": cfg.min_quality,
                    })
                return windows
            print("[train] MIT-BIH NSR returned 0 windows - falling back to synthetic")
        except Exception as e:
            print(f"[train] MIT-BIH NSR load failed ({e}) - falling back to synthetic")

    elif cfg.dataset_name == "uploaded" and cfg.uploaded_dataset_id:
        # User-uploaded dataset (local .dat/.hea or .csv files)
        upload_dir = settings.DATASET_DIR / "uploads" / cfg.uploaded_dataset_id
        print(f"[train] Loading uploaded dataset from {upload_dir}")
        try:
            windows = load_uploaded_dataset(
                upload_dir,
                max_records=cfg.max_records,
                duration_s_per_record=cfg.duration_per_record,
                use_ecg_qc=cfg.use_ecg_qc,
                min_quality=cfg.min_quality,
            )
            if len(windows) > 0:
                print(f"[train] Loaded {len(windows)} clean windows from uploaded dataset")
                if on_progress:
                    on_progress({
                        "type": "qc",
                        "dataset": f"uploaded:{cfg.uploaded_dataset_id}",
                        "n_kept": len(windows),
                        "qc_method": "ecg_qc" if cfg.use_ecg_qc else "simple_std",
                        "min_quality": cfg.min_quality,
                    })
                return windows
            print("[train] Uploaded dataset returned 0 windows - falling back to synthetic")
        except Exception as e:
            print(f"[train] Uploaded dataset load failed ({e}) - falling back to synthetic")

    # Synthetic fallback (always works)
    n_minutes = 30
    print(f"[train] Using {n_minutes} min of synthetic normal ECG")
    sig = synthetic_normal_ecg(duration_s=60 * n_minutes, seed=cfg.seed)
    # Synthetic signal is already preprocessed - just windowize
    windows = windowize(sig)

    if cfg.use_ecg_qc:
        windows, qc_stats = ecg_qc_quality_control(
            windows, fs=settings.SAMPLING_RATE_HZ,
            min_quality=cfg.min_quality,
        )
        print(f"[qc] Synthetic: {qc_stats['n_kept']}/{qc_stats['n_in']} windows kept "
              f"({qc_stats['retention_pct']:.1f}%) - dist: {qc_stats['distribution']}")
    else:
        n_before = len(windows)
        windows = quality_control(windows)
        qc_stats = {
            'n_in': n_before, 'n_kept': len(windows),
            'retention_pct': 100.0 * len(windows) / max(1, n_before),
            'distribution': {'0': n_before - len(windows), '1': 0, '2': len(windows), '3': 0},
        }
        print(f"[qc] Synthetic (simple QC): {len(windows)}/{n_before} windows kept")

    if on_progress:
        on_progress({
            "type": "qc",
            "dataset": "synthetic",
            "n_in": qc_stats['n_in'],
            "n_kept": qc_stats['n_kept'],
            "retention_pct": qc_stats['retention_pct'],
            "distribution": qc_stats['distribution'],
            "qc_method": "ecg_qc" if cfg.use_ecg_qc else "simple_std",
            "min_quality": cfg.min_quality,
        })

    return windows


def compute_threshold(model: nn.Module,
                      normal_loader: DataLoader,
                      device: torch.device,
                      k: float = settings.DEFAULT_THRESHOLD_K) -> float:
    """Robust threshold on reconstruction MAE of normal validation windows."""
    model.eval()
    maes: List[float] = []
    with torch.no_grad():
        for batch in normal_loader:
            # TensorDataset yields tuples — unpack
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x_hat = model(x)
            mae = (x - x_hat).abs().mean(dim=(1, 2))  # per-window MAE
            maes.extend(mae.cpu().numpy().tolist())
    arr = np.array(maes)
    if arr.size == 0:
        raise RuntimeError("Cannot compute anomaly threshold from an empty validation set")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma < 1e-8:
        robust_sigma = float(arr.std())
    robust_threshold = median + k * robust_sigma
    percentile_floor = float(np.percentile(arr, 95.0))
    return max(robust_threshold, percentile_floor)


def train(cfg: TrainConfig,
          on_progress: Optional[ProgressCallback] = None,
          should_stop: Optional[Callable[[], bool]] = None) -> TrainResult:
    """Train the autoencoder. Streams progress via `on_progress` callback.

    Returns a TrainResult containing the final model path and metrics.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------- Data ----------
    windows = _gather_training_windows(cfg, on_progress=on_progress)
    if len(windows) < 8:
        raise RuntimeError(
            f"Not enough training windows ({len(windows)}). "
            "Need at least 8. Try increasing max_records or duration_per_record."
        )

    # Train / val split (subject-wise simulation: last 15% as val)
    n = len(windows)
    n_train = int(n * cfg.train_split)
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_w = windows[train_idx]
    val_w   = windows[val_idx]

    train_ds = TensorDataset(windows_to_tensor(train_w))
    val_ds   = TensorDataset(windows_to_tensor(val_w))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    # ---------- Model ----------
    model = build_model(input_length=settings.WINDOW_SAMPLES,
                        skip_scale=cfg.skip_scale).to(device)
    arch_summary = model.architecture_summary()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()

    if on_progress:
        on_progress({
            "type": "init",
            "n_train": len(train_w),
            "n_val": len(val_w),
            "architecture": arch_summary,
            "device": str(device),
        })

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            on_progress and on_progress({"type": "stopped", "epoch": epoch})
            break

        # ---- Train ----
        model.train()
        running = 0.0
        n_batches = 0
        for i, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            if cfg.input_noise_std > 0:
                noisy_x = x + torch.randn_like(x) * cfg.input_noise_std
            else:
                noisy_x = x
            x_hat = model(noisy_x)
            loss = mse(x_hat, x)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

            # Per-batch progress (throttled)
            if on_progress and (i % 5 == 0 or i == len(train_loader) - 1):
                on_progress({
                    "type": "batch",
                    "epoch": epoch,
                    "total_epochs": cfg.epochs,
                    "batch": i + 1,
                    "total_batches": len(train_loader),
                    "batch_loss": loss.item(),
                    "elapsed_s": round(time.time() - start, 1),
                })

        train_loss = running / max(1, n_batches)

        # ---- Validate ----
        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_hat = model(x)
                val_running += mse(x_hat, x).item()
                val_batches += 1
        val_loss = val_running / max(1, val_batches)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_s": round(time.time() - start, 1),
        })

        if on_progress:
            on_progress({
                "type": "epoch",
                "epoch": epoch,
                "total_epochs": cfg.epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": round(time.time() - start, 1),
            })

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- Save best model + compute threshold ----
    if best_state is not None:
        model.load_state_dict(best_state)
    threshold = compute_threshold(model, val_loader, device, k=cfg.threshold_k)

    model_path = settings.MODEL_DIR / f"model_{int(time.time())}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": arch_summary,
        "config": asdict(cfg),
        "threshold": threshold,
        "threshold_k": cfg.threshold_k,
        "val_loss": best_val,
        "history": history,
    }, model_path)

    return TrainResult(
        final_loss=history[-1]["train_loss"] if history else float("nan"),
        val_loss=best_val,
        epochs_run=len(history),
        threshold=threshold,
        threshold_k=cfg.threshold_k,
        model_path=str(model_path),
        history=history,
        architecture=arch_summary,
    )
