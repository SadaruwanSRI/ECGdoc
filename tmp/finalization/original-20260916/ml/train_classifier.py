"""Training loop for the arrhythmia classifier.

Loads MIT-BIH Arrhythmia Database records with beat annotations,
extracts individual beats, and trains the classification head
(transfer learning from the frozen autoencoder encoder).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from app.core.config import settings
from app.ml.classifier import (
    ECGClassifier, ARRHYTHMIA_CLASSES, ARRHYTHMIA_NAMES,
    build_classifier, mitbih_symbol_to_class,
)
from app.ml.data import (
    preprocess_signal, windowize, _download_record,
    MITDB_RECORDS,
)


ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class ClassifierTrainConfig:
    encoder_model_path: str = ""          # path to Model 1's .pt file
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_records: int = 46                 # eligible MIT-BIH MLII records (1-46)
    duration_per_record: float = 1800.0   # 30 minutes per record
    train_split: float = 0.8
    val_split: float = 0.1                # test = 1 - train - val
    seed: int = 42
    device: str = "cpu"
    hidden_dim: int = 256
    dropout: float = 0.35
    freeze_encoder: bool = False
    encoder_learning_rate: float = 1e-4
    focal_gamma: float = 2.0
    augment: bool = True
    # Beat extraction
    beat_window_samples: int = 512        # 4 seconds at 128 Hz
    r_peak_before: int = 200              # samples before R-peak


@dataclass
class ClassifierTrainResult:
    final_loss: float
    final_accuracy: float
    val_accuracy: float
    val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    epochs_run: int
    model_path: str
    confusion_matrix: List[List[int]]
    class_names: List[str]
    per_class_metrics: Dict[str, Dict[str, float]]
    history: List[Dict[str, Any]] = field(default_factory=list)
    n_beats: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)
    split_protocol: str = "patient-level stratified search"
    train_records: List[str] = field(default_factory=list)
    validation_records: List[str] = field(default_factory=list)
    test_records: List[str] = field(default_factory=list)
    train_subjects: List[str] = field(default_factory=list)
    validation_subjects: List[str] = field(default_factory=list)
    test_subjects: List[str] = field(default_factory=list)


def _subject_id(record_id: str) -> str:
    """Return the MITDB subject identifier used for leakage-safe splitting.

    MITDB records 201 and 202 are two recordings from the same person and must
    never be assigned to different partitions.
    """
    return "201-202" if record_id in {"201", "202"} else record_id


def _patient_level_split(
    record_ids: np.ndarray,
    labels: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    candidates: int = 4000,
) -> Dict[str, List[str]]:
    """Choose disjoint patient partitions with similar class distributions.

    A deterministic random search operates only on subject-level class counts.
    Test labels are used only to construct a representative research split;
    model fitting, early stopping, and parameter selection never use test
    beats. The final report uses the stronger all-subject out-of-fold study.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_split and val_split must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_split + val_split must be less than one")

    subjects = np.asarray([_subject_id(str(record)) for record in record_ids])
    unique_subjects = np.asarray(sorted(np.unique(subjects).tolist()))
    n_subjects = len(unique_subjects)
    if n_subjects < 3:
        raise RuntimeError(
            f"Need at least 3 patients for train/validation/test splitting; got {n_subjects}. "
            "Increase max_records."
        )

    n_train = max(1, int(round(n_subjects * train_fraction)))
    n_validation = max(1, int(round(n_subjects * validation_fraction)))
    if n_train + n_validation >= n_subjects:
        n_train = max(1, n_subjects - 2)
        n_validation = 1

    counts = np.zeros((n_subjects, len(ARRHYTHMIA_CLASSES)), dtype=np.float64)
    subject_index = {subject: i for i, subject in enumerate(unique_subjects)}
    for subject, label in zip(subjects, labels):
        counts[subject_index[str(subject)], int(label)] += 1
    overall = counts.sum(axis=0)
    overall_distribution = overall / max(1.0, overall.sum())
    class_subject_support = np.sum(counts > 0, axis=0)
    feasible_everywhere = class_subject_support >= 3

    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_partition: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(max(1, candidates)):
        order = rng.permutation(n_subjects)
        train = order[:n_train]
        validation = order[n_train:n_train + n_validation]
        test = order[n_train + n_validation:]
        score = 0.0
        for partition in (train, validation, test):
            partition_counts = counts[partition].sum(axis=0)
            distribution = partition_counts / max(1.0, partition_counts.sum())
            score += float(np.mean(np.abs(distribution - overall_distribution)))
            score += 2.0 * float(np.sum((partition_counts == 0) & feasible_everywhere))
        if score < best_score:
            best_score = score
            best_partition = (train, validation, test)

    assert best_partition is not None
    split: Dict[str, List[str]] = {}
    for name, indices in zip(("train", "validation", "test"), best_partition):
        selected_subjects = set(unique_subjects[indices].tolist())
        split[f"{name}_subjects"] = sorted(selected_subjects)
        split[f"{name}_records"] = sorted(
            np.unique(record_ids[np.isin(subjects, list(selected_subjects))]).tolist()
        )

    assert set(split["train_subjects"]).isdisjoint(split["validation_subjects"])
    assert set(split["train_subjects"]).isdisjoint(split["test_subjects"])
    assert set(split["validation_subjects"]).isdisjoint(split["test_subjects"])
    return split


def _extract_beats_from_record(record_name: str,
                                 beat_window: int = 512,
                                 r_before: int = 200
                                 ) -> Tuple[List[np.ndarray], List[str]]:
    """Extract individual beats from a MIT-BIH Arrhythmia record with annotations.

    Returns:
        (beats, labels) where beats is a list of (512,) arrays and
        labels is a list of class codes ("N", "PVC", "PAC", "LBBB", "RBBB").
    """
    try:
        signal, src_fs = _download_record(
            record_name, "mitdb", required_lead="MLII"
        )
    except Exception as e:
        print(f"[warn] Could not load MITDB record {record_name}: {e}")
        return [], []

    # Load annotations
    try:
        import wfdb
        cache_dir = settings.DATASET_DIR / "mitdb"
        if (cache_dir / f"{record_name}.hea").exists() and (cache_dir / f"{record_name}.atr").exists():
            ann = wfdb.rdann(str(cache_dir / record_name), "atr")
        else:
            try:
                ann = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
            except TypeError:
                ann = wfdb.rdann(record_name, "atr", pb_dir="mitdb")
    except Exception as e:
        print(f"[warn] Could not load annotations for {record_name}: {e}")
        return [], []

    # Preprocess the signal
    signal = preprocess_signal(signal, src_fs)

    # Resample annotations to match the preprocessed signal
    # The annotation sample indices are in the original sampling rate
    # After resampling to 128 Hz, we need to scale the indices
    scale = settings.SAMPLING_RATE_HZ / src_fs
    r_peaks = (ann.sample * scale).astype(int)
    rhythm_by_sample = []
    current_rhythm = None
    for sample, note in zip(r_peaks, getattr(ann, "aux_note", [])):
        note = (note or "").strip()
        if note.startswith("("):
            if "AFIB" in note or "AFL" in note:
                current_rhythm = "AFib"
            elif note:
                current_rhythm = None
            rhythm_by_sample.append((int(sample), current_rhythm))

    def rhythm_at(sample: int) -> Optional[str]:
        rhythm = None
        for change_sample, change_rhythm in rhythm_by_sample:
            if change_sample > sample:
                break
            rhythm = change_rhythm
        return rhythm

    # Map annotation symbols to our classes
    beats = []
    labels = []
    for i, (peak, symbol) in enumerate(zip(r_peaks, ann.symbol)):
        cls = mitbih_symbol_to_class(symbol, rhythm=rhythm_at(int(peak)))
        if cls is None:
            continue

        # Extract beat window centered on R-peak
        start = peak - r_before
        end = start + beat_window
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end].astype(np.float32)
        if len(beat) != beat_window:
            continue

        beats.append(beat)
        labels.append(cls)

    return beats, labels


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Small ECG-safe augmentations for classifier robustness."""
    if x.numel() == 0:
        return x
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.85, 1.15)
    noise = torch.randn_like(x) * 0.025
    shift = int(torch.randint(-24, 25, (1,), device=x.device).item())
    out = x * scale + noise
    if shift:
        out = torch.roll(out, shifts=shift, dims=-1)
    return out


def _evaluate_classifier(model: nn.Module,
                         loader: DataLoader,
                         criterion: nn.Module,
                         device: torch.device) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    confusion = np.zeros((len(ARRHYTHMIA_CLASSES), len(ARRHYTHMIA_CLASSES)), dtype=int)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += criterion(logits, yb).item()
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            for true, pred in zip(yb.cpu(), preds.cpu()):
                confusion[int(true), int(pred)] += 1

    f1s = []
    for i in range(len(ARRHYTHMIA_CLASSES)):
        support = confusion[i, :].sum()
        if support == 0:
            continue
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-8, precision + recall))

    return (
        loss_sum / max(1, len(loader)),
        correct / max(1, total),
        float(np.mean(f1s)) if f1s else 0.0,
        confusion,
    )


def gather_training_data(cfg: ClassifierTrainConfig,
                          on_progress: Optional[ProgressCallback] = None
                          ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    """Gather labeled beats from MIT-BIH Arrhythmia Database.

    Returns:
        (beats, labels, class_distribution, record_ids)
        beats: (N, 512) float32 array
        labels: (N,) int array (0-5 for our 6 classes)
        class_distribution: {"N": 75000, "PVC": 7000, ...}
        record_ids: (N,) source MIT-BIH record id for leakage-free splitting
    """
    records = MITDB_RECORDS[:cfg.max_records]
    all_beats: List[np.ndarray] = []
    all_labels: List[str] = []
    all_record_ids: List[str] = []
    class_dist = {c: 0 for c in ARRHYTHMIA_CLASSES}

    for i, rec in enumerate(records):
        beats, labels = _extract_beats_from_record(
            rec, beat_window=cfg.beat_window_samples, r_before=cfg.r_peak_before
        )
        all_beats.extend(beats)
        all_labels.extend(labels)
        all_record_ids.extend([rec] * len(beats))
        for l in labels:
            class_dist[l] = class_dist.get(l, 0) + 1
        print(f"[ok] MITDB {rec}: {len(beats)} beats extracted")

        if on_progress and (i + 1) % 5 == 0:
            on_progress({
                "type": "extraction_progress",
                "records_processed": i + 1,
                "total_records": len(records),
                "total_beats": len(all_beats),
            })

    if not all_beats:
        raise RuntimeError("No beats extracted from MIT-BIH Arrhythmia DB")

    # Convert to arrays
    beats_array = np.array(all_beats, dtype=np.float32)
    label_to_idx = {c: i for i, c in enumerate(ARRHYTHMIA_CLASSES)}
    labels_array = np.array([label_to_idx[l] for l in all_labels], dtype=np.int64)
    record_ids = np.array(all_record_ids)

    print(f"[classifier] Total beats: {len(beats_array)}")
    print(f"[classifier] Class distribution: {class_dist}")

    if on_progress:
        on_progress({
            "type": "extraction_complete",
            "total_beats": len(beats_array),
            "class_distribution": class_dist,
        })

    return beats_array, labels_array, class_dist, record_ids


def train_classifier(cfg: ClassifierTrainConfig,
                      on_progress: Optional[ProgressCallback] = None,
                      should_stop: Optional[Callable[[], bool]] = None
                      ) -> ClassifierTrainResult:
    """Train the arrhythmia classifier.

    1. Loads the encoder from Model 1 (frozen)
    2. Gathers labeled beats from MIT-BIH Arrhythmia DB
    3. Trains the classification head (CrossEntropyLoss with class weights)
    4. Evaluates on test set (confusion matrix + per-class metrics)
    5. Saves the classifier to disk
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------- Step 1: Load encoder + build classifier ----------
    if not cfg.encoder_model_path:
        raise ValueError("encoder_model_path is required (path to Model 1's .pt file)")

    if on_progress:
        on_progress({"type": "loading_encoder", "path": cfg.encoder_model_path})

    classifier = build_classifier(
        cfg.encoder_model_path,
        input_length=cfg.beat_window_samples,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        freeze_encoder=cfg.freeze_encoder,
        use_raw_branch=True,
    )
    classifier = classifier.to(device)

    if on_progress:
        on_progress({
            "type": "model_built",
            "architecture": classifier.architecture_summary(),
        })

    # ---------- Step 2: Gather training data ----------
    if on_progress:
        on_progress({"type": "gathering_data"})

    beats, labels, class_dist, record_ids = gather_training_data(cfg, on_progress=on_progress)

    # ---------- Step 3: Patient-wise train/val/test split ----------
    # Records 201 and 202 belong to the same person. Splitting only by record
    # would leak that person's morphology across partitions.
    split = _patient_level_split(
        record_ids,
        labels,
        cfg.train_split,
        cfg.val_split,
        cfg.seed,
    )
    train_records = set(split["train_records"])
    val_records = set(split["validation_records"])
    test_records = set(split["test_records"])

    train_idx = np.where(np.isin(record_ids, list(train_records)))[0]
    val_idx = np.where(np.isin(record_ids, list(val_records)))[0]
    test_idx = np.where(np.isin(record_ids, list(test_records)))[0]

    X_train = torch.from_numpy(beats[train_idx]).float().unsqueeze(1)  # (N, 1, 512)
    y_train = torch.from_numpy(labels[train_idx])
    X_val = torch.from_numpy(beats[val_idx]).float().unsqueeze(1)
    y_val = torch.from_numpy(labels[val_idx])
    X_test = torch.from_numpy(beats[test_idx]).float().unsqueeze(1)
    y_test = torch.from_numpy(labels[test_idx])

    # ---------- Step 4: Class-weighted sampling ----------
    # Compute class weights (inverse frequency) to handle class imbalance
    class_counts = np.bincount(labels[train_idx], minlength=len(ARRHYTHMIA_CLASSES))
    class_weights = np.zeros(len(ARRHYTHMIA_CLASSES), dtype=np.float32)
    present = class_counts > 0
    class_weights[present] = class_counts[present].sum() / (
        present.sum() * class_counts[present]
    )
    sample_weights = class_weights[labels[train_idx]]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_idx), replacement=True
    )

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # ---------- Step 5: Train ----------
    head_params = list(classifier.head.parameters())
    if classifier.raw_branch is not None:
        head_params += list(classifier.raw_branch.parameters())
    encoder_params = [p for p in classifier.autoencoder.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": cfg.learning_rate}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.encoder_learning_rate})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    # Class-weighted cross-entropy loss
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(weight=weight_tensor, gamma=cfg.focal_gamma)

    if on_progress:
        on_progress({
            "type": "init",
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "class_weights": class_weights.tolist(),
            "split_protocol": "patient-level stratified search",
            **split,
        })

    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = 0.0
    best_state = None
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            on_progress and on_progress({"type": "stopped", "epoch": epoch})
            break

        # ---- Train ----
        classifier.train()
        running_loss = 0.0
        n_correct = 0
        n_total = 0

        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            if cfg.augment:
                xb = _augment_batch(xb)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            n_correct += (preds == yb).sum().item()
            n_total += yb.size(0)

            if on_progress and (i % 20 == 0 or i == len(train_loader) - 1):
                on_progress({
                    "type": "batch",
                    "epoch": epoch,
                    "total_epochs": cfg.epochs,
                    "batch": i + 1,
                    "total_batches": len(train_loader),
                    "batch_loss": loss.item(),
                    "elapsed_s": round(time.time() - start, 1),
                })

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = n_correct / max(1, n_total)

        val_loss, val_acc, val_macro_f1, _ = _evaluate_classifier(
            classifier, val_loader, criterion, device
        )
        scheduler.step(val_macro_f1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "elapsed_s": round(time.time() - start, 1),
        })

        if on_progress:
            on_progress({
                "type": "epoch",
                "epoch": epoch,
                "total_epochs": cfg.epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "elapsed_s": round(time.time() - start, 1),
            })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in classifier.state_dict().items()}

    # ---------- Step 6: Evaluate on test set ----------
    if best_state is not None:
        classifier.load_state_dict(best_state)

    _, test_acc, test_macro_f1, confusion = _evaluate_classifier(
        classifier, test_loader, criterion, device
    )

    # Per-class metrics
    per_class = {}
    for i, cls in enumerate(ARRHYTHMIA_CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[i, :].sum()),
        }

    if on_progress:
        on_progress({
            "type": "evaluation",
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "confusion_matrix": confusion.tolist(),
            "per_class_metrics": per_class,
        })

    # ---------- Step 7: Save ----------
    model_path = settings.MODEL_DIR / f"classifier_{int(time.time())}.pt"
    torch.save({
        "state_dict": classifier.state_dict(),
        "architecture": classifier.architecture_summary(),
        "config": asdict(cfg),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "confusion_matrix": confusion.tolist(),
        "per_class_metrics": per_class,
        "class_names": ARRHYTHMIA_CLASSES,
        "history": history,
        "split_protocol": "patient-level stratified search",
        **split,
    }, model_path)

    return ClassifierTrainResult(
        final_loss=history[-1]["train_loss"] if history else float("nan"),
        final_accuracy=history[-1]["train_acc"] if history else 0.0,
        val_accuracy=history[-1]["val_acc"] if history else 0.0,
        val_macro_f1=best_val_macro_f1,
        test_accuracy=test_acc,
        test_macro_f1=test_macro_f1,
        epochs_run=len(history),
        model_path=str(model_path),
        confusion_matrix=confusion.tolist(),
        class_names=ARRHYTHMIA_CLASSES,
        per_class_metrics=per_class,
        history=history,
        n_beats=len(beats),
        class_distribution=class_dist,
        architecture=classifier.architecture_summary(),
        split_protocol="patient-level stratified search",
        train_records=split["train_records"],
        validation_records=split["validation_records"],
        test_records=split["test_records"],
        train_subjects=split["train_subjects"],
        validation_subjects=split["validation_subjects"],
        test_subjects=split["test_subjects"],
    )
