"""Train the final normal-only autoencoder on the MIT-BIH NSRDB.

All 18 real long-term recordings are used.  Only the first stored ECG channel
is accepted (the WFDB headers call it ECG1).  The whole recording is cleaned
and divided into non-overlapping four-second windows.  Fifteen subjects fit the
network and three different subjects provide validation and the normal-error
threshold.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import wfdb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = REPORT_DIR.parent
BACKEND_DIR = REPO_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.data import bandpass_filter, windowize, windows_to_tensor  # noqa: E402
from app.ml.model import DENOISING_SKIP_SCALE, build_model  # noqa: E402


SEED = 20260802
RECORDS = [
    "16265", "16272", "16273", "16420", "16483", "16539",
    "16773", "16786", "16795", "17052", "17453", "18177",
    "18184", "19088", "19090", "19093", "19140", "19830",
]
DATABASE = "nsrdb"
CHANNEL_INDEX = 0
CHANNEL_HEADER_NAME = "ECG1"
EPOCHS = 2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VALIDATION_SUBJECTS = 3
INPUT_NOISE_STD = 0.03
CACHE_DIR = BACKEND_DIR / "app/storage/datasets/nsrdb-primary-full"
SOURCE_CACHE_DIR = BACKEND_DIR / "app/storage/datasets/nsrdb"
MODEL_PATH = BACKEND_DIR / "app/storage/models/model_nsrdb_primary_healthy.pt"
SUMMARY_PATH = REPORT_DIR / "experiments/nsrdb_autoencoder_training.json"
FIGURE_PATH = REPORT_DIR / "pic/generated/nsrdb_autoencoder_learning_curve.pdf"


def _source_signal(record_name: str) -> tuple[np.ndarray, float, str]:
    """Return the complete first-channel waveform, sampling rate, and name."""
    cached = SOURCE_CACHE_DIR / f"{record_name}.npy"
    cached_fs = SOURCE_CACHE_DIR / f"{record_name}.fs"
    if cached.exists() and cached_fs.exists():
        header = wfdb.rdheader(str(SOURCE_CACHE_DIR / record_name))
        channel_name = str(header.sig_name[CHANNEL_INDEX])
        if channel_name.upper() != CHANNEL_HEADER_NAME:
            raise ValueError(
                f"Record {record_name} first channel is {channel_name}, "
                f"not {CHANNEL_HEADER_NAME}"
            )
        signal = np.load(cached).astype(np.float32, copy=False)
        return signal, float(cached_fs.read_text().strip()), channel_name

    record = wfdb.rdrecord(record_name, pn_dir=DATABASE)
    if record.p_signal.ndim != 2 or record.p_signal.shape[1] <= CHANNEL_INDEX:
        raise ValueError(f"Record {record_name} has no first ECG channel")
    channel_name = str(record.sig_name[CHANNEL_INDEX])
    if channel_name.upper() != CHANNEL_HEADER_NAME:
        raise ValueError(
            f"Record {record_name} first channel is {channel_name}, not {CHANNEL_HEADER_NAME}"
        )
    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    signal = record.p_signal[:, CHANNEL_INDEX].astype(np.float32)
    np.save(cached, signal)
    cached_fs.write_text(str(record.fs), encoding="utf-8")
    return signal, float(record.fs), channel_name


def prepare_record(record_name: str) -> dict[str, object]:
    """Clean the complete recording and cache every accepted 4-second window."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{record_name}_primary_full.npy"
    meta_path = CACHE_DIR / f"{record_name}_primary_full.json"
    if cache_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["cache_path"] = str(cache_path)
        return metadata

    raw, source_rate, channel_name = _source_signal(record_name)
    finite = np.isfinite(raw)
    if finite.sum() < len(raw) * 0.99:
        raise ValueError(f"Record {record_name} has too many missing samples")
    if not finite.all():
        positions = np.arange(len(raw))
        raw = np.interp(positions, positions[finite], raw[finite]).astype(np.float32)

    if source_rate != 128:
        raise ValueError(
            f"NSRDB record {record_name} has unexpected sampling rate {source_rate}"
        )
    # Filtering a 24-hour vector in one call creates large temporary arrays.
    # Process aligned 30-minute blocks with a four-second margin on each side.
    # The retained central pieces cover every complete four-second window once.
    window_samples = 512
    block_samples = 30 * 60 * int(source_rate)
    usable_samples = (len(raw) // window_samples) * window_samples
    candidate_parts: list[np.ndarray] = []
    for central_start in range(0, usable_samples, block_samples):
        central_end = min(central_start + block_samples, usable_samples)
        extended_start = max(0, central_start - window_samples)
        extended_end = min(len(raw), central_end + window_samples)
        filtered = bandpass_filter(raw[extended_start:extended_end], source_rate)
        kept_start = central_start - extended_start
        kept_end = kept_start + (central_end - central_start)
        candidate_parts.append(windowize(filtered[kept_start:kept_end]))
    candidates = np.concatenate(candidate_parts).astype(np.float32, copy=False)
    # Standardise each model example independently. This prevents a long-term
    # gain shift from dominating the reconstruction objective.
    row_mean = candidates.mean(axis=1, keepdims=True)
    row_std = candidates.std(axis=1, keepdims=True)
    candidates = ((candidates - row_mean) / (row_std + 1e-8)).astype(np.float32)
    standard_deviation = candidates.std(axis=1)
    accepted = (
        np.all(np.isfinite(candidates), axis=1)
        & (np.ptp(candidates, axis=1) >= 0.5)
        & (standard_deviation >= 0.05)
        & (standard_deviation <= 5.0)
    )
    windows = candidates[accepted].astype(np.float32)
    np.save(cache_path, windows)
    metadata = {
        "record": record_name,
        "channel_index": CHANNEL_INDEX,
        "channel_header_name": channel_name,
        "source_rate_hz": source_rate,
        "source_samples": int(len(raw)),
        "source_hours": float(len(raw) / source_rate / 3600.0),
        "candidate_windows": int(len(candidates)),
        "accepted_windows": int(len(windows)),
        "rejected_windows": int(len(candidates) - len(windows)),
        "cache_path": str(cache_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def validation_loss(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    squared_error = 0.0
    examples = 0
    with torch.no_grad():
        for (clean,) in loader:
            batch_loss = nn.functional.mse_loss(model(clean), clean, reduction="sum")
            squared_error += float(batch_loss.item())
            examples += int(clean.numel())
    return squared_error / max(1, examples)


def create_learning_curve(history: list[dict[str, float | int]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    training = [float(row["train_loss"]) for row in history]
    validation = [float(row["validation_loss"]) for row in history]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(epochs, training, marker="o", color="#2563eb", label="fitting loss")
    ax.plot(epochs, validation, marker="o", color="#0f766e", label="subject-disjoint validation loss")
    ax.set(
        xlabel="epoch",
        ylabel="mean squared reconstruction error",
        title="MIT-BIH normal-sinus autoencoder learning curve",
    )
    ax.set_xticks(epochs)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.annotate(
        f"best validation MSE = {min(validation):.5f}",
        xy=(epochs[int(np.argmin(validation))], min(validation)),
        xytext=(epochs[0], max(validation)),
        arrowprops={"arrowstyle": "->", "color": "#0f766e"},
        fontsize=9,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    metadata = []
    for index, name in enumerate(RECORDS, start=1):
        row = prepare_record(name)
        metadata.append(row)
        print(
            f"[nsrdb] prepared {index}/{len(RECORDS)} {name}: "
            f"{row['accepted_windows']}/{row['candidate_windows']} windows",
            flush=True,
        )

    rng = np.random.default_rng(SEED)
    shuffled = np.asarray(RECORDS)[rng.permutation(len(RECORDS))]
    validation_names = shuffled[:VALIDATION_SUBJECTS].tolist()
    training_names = shuffled[VALIDATION_SUBJECTS:].tolist()
    path_by_record = {str(row["record"]): Path(str(row["cache_path"])) for row in metadata}
    training_windows = np.concatenate(
        [np.load(path_by_record[name]) for name in training_names]
    ).astype(np.float32, copy=False)
    validation_windows = np.concatenate(
        [np.load(path_by_record[name]) for name in validation_names]
    ).astype(np.float32, copy=False)
    print(
        f"[nsrdb] fitting subjects={len(training_names)} windows={len(training_windows):,}; "
        f"validation subjects={len(validation_names)} windows={len(validation_windows):,}",
        flush=True,
    )

    train_loader = DataLoader(
        TensorDataset(windows_to_tensor(training_windows)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(windows_to_tensor(validation_windows)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    model = build_model(input_length=512, skip_scale=DENOISING_SKIP_SCALE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        squared_error = 0.0
        examples = 0
        for batch_index, (clean,) in enumerate(train_loader, start=1):
            noisy = clean + torch.randn_like(clean) * INPUT_NOISE_STD
            optimizer.zero_grad()
            reconstruction = model(noisy)
            loss = nn.functional.mse_loss(reconstruction, clean)
            loss.backward()
            optimizer.step()
            squared_error += float(loss.item()) * int(clean.numel())
            examples += int(clean.numel())
            if batch_index % 250 == 0:
                print(
                    f"[nsrdb] epoch {epoch}/{EPOCHS}: batch "
                    f"{batch_index}/{len(train_loader)}",
                    flush=True,
                )
        train_loss = squared_error / max(1, examples)
        val_loss = validation_loss(model, validation_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(
            f"[nsrdb] epoch {epoch}/{EPOCHS}: train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Autoencoder training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    residual_rows: list[np.ndarray] = []
    with torch.no_grad():
        for (clean,) in validation_loader:
            residual_rows.append(
                (clean - model(clean)).abs().mean(dim=(1, 2)).numpy()
            )
    residuals = np.concatenate(residual_rows)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    threshold = max(
        median + 3.0 * 1.4826 * mad,
        float(np.percentile(residuals, 95)),
    )

    total_source_hours = float(sum(float(row["source_hours"]) for row in metadata))
    total_candidates = int(sum(int(row["candidate_windows"]) for row in metadata))
    total_accepted = int(sum(int(row["accepted_windows"]) for row in metadata))
    config = {
        "dataset": "MIT-BIH Normal Sinus Rhythm Database",
        "database_slug": DATABASE,
        "input": "first stored NSRDB ECG channel",
        "wfdb_channel_name": CHANNEL_HEADER_NAME,
        "channel_index": CHANNEL_INDEX,
        "healthy_subjects_total": len(RECORDS),
        "training_subjects": len(training_names),
        "validation_subjects": len(validation_names),
        "training_record_names": training_names,
        "validation_record_names": validation_names,
        "whole_recordings_processed": True,
        "total_source_hours": total_source_hours,
        "candidate_windows": total_candidates,
        "accepted_clean_windows": total_accepted,
        "rejected_windows": total_candidates - total_accepted,
        "training_windows": int(len(training_windows)),
        "validation_windows": int(len(validation_windows)),
        "subject_disjoint_validation": True,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "skip_scale": DENOISING_SKIP_SCALE,
        "input_noise_std": INPUT_NOISE_STD,
        "record_preparation": metadata,
    }
    checkpoint = {
        "state_dict": best_state,
        "architecture": model.architecture_summary(),
        "config": config,
        "threshold": threshold,
        "threshold_k": 3.0,
        "val_loss": best_loss,
        "history": history,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MODEL_PATH)
    summary = {
        **config,
        "best_validation_loss": best_loss,
        "normal_residual_threshold": threshold,
        "history": history,
        "model_path": str(MODEL_PATH),
        "elapsed_seconds": time.time() - started,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_learning_curve(history)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
