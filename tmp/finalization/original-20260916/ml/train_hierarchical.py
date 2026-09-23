"""Leakage-safe training and evaluation for the hierarchical ECG model."""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import settings
from app.ml.classifier import (
    ARRHYTHMIA_CLASSES,
    mitbih_symbol_is_anomaly,
    mitbih_symbol_to_class,
)
from app.ml.data import MITDB_RECORDS, _download_record, preprocess_signal
from app.ml.hierarchical import build_hierarchical_model


# The original six records remain untouched for direct comparison. Record 201
# adds AFib support to the expanded final test. Validation records collectively
# contain all six project classes.
ORIGINAL_TEST_RECORDS = ["122", "107", "228", "115", "207", "220"]
EXPANDED_TEST_RECORDS = ORIGINAL_TEST_RECORDS + ["201"]
VALIDATION_RECORDS = ["111", "124", "119", "234", "202"]
TRAIN_RECORDS = [
    record
    for record in MITDB_RECORDS
    if record not in set(EXPANDED_TEST_RECORDS + VALIDATION_RECORDS)
]


@dataclass
class HierarchicalTrainConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 2e-4
    dropout: float = 0.20
    class_loss_weight: float = 0.70
    class_balance_beta: float = 0.9995
    label_smoothing: float = 0.03
    patience: int = 7
    seed: int = 42
    device: str = "cpu"
    r_peak_before: int = 200
    window_samples: int = 512
    augment: bool = True
    num_workers: int = 0
    cache_path: str = str(
        settings.DATASET_DIR / "mitdb" / "hierarchical_beat_cache_512_mlii_v1.npz"
    )


def _load_annotations(record_name: str):
    import wfdb

    return wfdb.rdann(str(settings.DATASET_DIR / "mitdb" / record_name), "atr")


def _rhythm_lookup(samples: np.ndarray, notes: Iterable[str]):
    changes: list[tuple[int, Optional[str]]] = []
    current: Optional[str] = None
    for sample, note in zip(samples, notes):
        note = (note or "").strip()
        if note.startswith("("):
            current = "AFib" if ("AFIB" in note or "AFL" in note) else None
            changes.append((int(sample), current))

    def rhythm_at(sample: int) -> Optional[str]:
        rhythm: Optional[str] = None
        for change_sample, change_rhythm in changes:
            if change_sample > sample:
                break
            rhythm = change_rhythm
        return rhythm

    return rhythm_at


def _extract_record(
    record_name: str,
    window_samples: int,
    r_peak_before: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_signal, source_rate = _download_record(
        record_name, "mitdb", required_lead="MLII"
    )
    annotation = _load_annotations(record_name)
    signal = preprocess_signal(raw_signal, source_rate)
    scale = settings.SAMPLING_RATE_HZ / source_rate
    peaks = (annotation.sample * scale).astype(int)
    rhythm_at = _rhythm_lookup(
        peaks, list(getattr(annotation, "aux_note", []))
    )
    class_to_index = {name: index for index, name in enumerate(ARRHYTHMIA_CLASSES)}

    accepted: list[tuple[int, int, int]] = []
    for peak, symbol in zip(peaks, annotation.symbol):
        rhythm = rhythm_at(int(peak))
        binary_label = mitbih_symbol_is_anomaly(symbol, rhythm=rhythm)
        if binary_label is None:
            continue
        start = int(peak) - r_peak_before
        end = start + window_samples
        if start < 0 or end > len(signal):
            continue
        window = signal[start:end]
        if window.shape != (window_samples,):
            continue
        project_class = mitbih_symbol_to_class(symbol, rhythm=rhythm)
        accepted.append(
            (
                int(peak),
                int(binary_label),
                class_to_index[project_class] if project_class is not None else -1,
            )
        )

    accepted_peaks = np.asarray([item[0] for item in accepted], dtype=np.int64)
    intervals = (
        np.diff(accepted_peaks).astype(np.float32) / settings.SAMPLING_RATE_HZ
    )
    typical_interval = float(np.median(intervals)) if len(intervals) else 1.0
    windows: list[np.ndarray] = []
    binary_labels: list[int] = []
    class_labels: list[int] = []
    rr_features: list[np.ndarray] = []
    for index, (peak, binary_label, class_label) in enumerate(accepted):
        start = peak - r_peak_before
        end = start + window_samples
        window = signal[start:end]
        windows.append(window.astype(np.float32))
        binary_labels.append(int(binary_label))
        class_labels.append(class_label)

        previous_rr = float(intervals[index - 1]) if index > 0 else typical_interval
        next_rr = (
            float(intervals[index])
            if index < len(intervals)
            else previous_rr
        )
        history = intervals[max(0, index - 10):index]
        if len(history) == 0:
            history = np.asarray([typical_interval], dtype=np.float32)
        local_mean = float(np.mean(history))
        local_std = float(np.std(history))
        safe_mean = max(local_mean, 1e-3)
        rr_features.append(
            np.asarray(
                [
                    previous_rr,
                    next_rr,
                    local_mean,
                    previous_rr / safe_mean,
                    next_rr / safe_mean,
                    local_std / safe_mean,
                    60.0 / max(previous_rr, 1e-3),
                ],
                dtype=np.float32,
            )
        )

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(binary_labels, dtype=np.int64),
        np.asarray(class_labels, dtype=np.int64),
        np.asarray(rr_features, dtype=np.float32),
    )


def load_or_build_dataset(
    config: HierarchicalTrainConfig,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    cache_path = Path(config.cache_path)
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path)
        required = ("windows", "binary", "classes", "rr_features", "records")
        return {key: cached[key] for key in required}

    windows: list[np.ndarray] = []
    binary: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    records: list[np.ndarray] = []
    rr_features: list[np.ndarray] = []
    for record in MITDB_RECORDS:
        (
            record_windows,
            record_binary,
            record_classes,
            record_rr,
        ) = _extract_record(
            record,
            window_samples=config.window_samples,
            r_peak_before=config.r_peak_before,
        )
        windows.append(record_windows)
        binary.append(record_binary)
        classes.append(record_classes)
        rr_features.append(record_rr)
        records.append(np.asarray([record] * len(record_windows), dtype="<U3"))
        print(
            f"[hierarchical:data] {record}: {len(record_windows):,} binary beats, "
            f"{int(np.sum(record_classes >= 0)):,} class-labelled"
        )

    dataset = {
        "windows": np.concatenate(windows),
        "binary": np.concatenate(binary),
        "classes": np.concatenate(classes),
        "rr_features": np.concatenate(rr_features),
        "records": np.concatenate(records),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **dataset)
    return dataset


def _indices_for_records(records: np.ndarray, selected: Iterable[str]) -> np.ndarray:
    return np.where(np.isin(records, list(selected)))[0]


def _effective_number_weights(
    labels: np.ndarray,
    beta: float,
    n_classes: int,
) -> np.ndarray:
    counts = np.bincount(labels[labels >= 0], minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    effective_number = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_number, 1e-12)
    weights[present] /= np.mean(weights[present])
    return weights.astype(np.float32)


def _augment(x: torch.Tensor) -> torch.Tensor:
    batch, _, length = x.shape
    device = x.device
    scale = torch.empty(batch, 1, 1, device=device).uniform_(0.90, 1.10)
    noise_scale = torch.empty(batch, 1, 1, device=device).uniform_(0.0, 0.025)
    augmented = x * scale + torch.randn_like(x) * noise_scale

    # Independent small timing shifts preserve morphology while reducing
    # dependence on one exact annotation alignment.
    shifts = torch.randint(-12, 13, (batch, 1, 1), device=device)
    positions = torch.arange(length, device=device).view(1, 1, length)
    gather_index = (positions - shifts) % length
    augmented = torch.gather(augmented, 2, gather_index.expand(batch, 1, length))

    # Small low-frequency baseline variation approximates electrode drift.
    t = torch.linspace(0.0, 1.0, length, device=device).view(1, 1, length)
    frequency = torch.empty(batch, 1, 1, device=device).uniform_(0.20, 0.65)
    phase = torch.empty(batch, 1, 1, device=device).uniform_(0.0, 2.0 * math.pi)
    amplitude = torch.empty(batch, 1, 1, device=device).uniform_(0.0, 0.035)
    return augmented + amplitude * torch.sin(2.0 * math.pi * frequency * t + phase)


def _div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def choose_binary_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    thresholds = np.unique(np.concatenate([[0.0, 0.5, 1.0], probabilities]))
    order = np.argsort(probabilities)
    ordered_probabilities = probabilities[order]
    ordered_labels = labels[order]
    positive_prefix = np.concatenate([[0], np.cumsum(ordered_labels == 1)])
    negative_prefix = np.concatenate([[0], np.cumsum(ordered_labels == 0)])
    cut = np.searchsorted(ordered_probabilities, thresholds, side="right")
    total_positive = int(np.sum(labels == 1))
    total_negative = int(np.sum(labels == 0))
    tp = total_positive - positive_prefix[cut]
    fp = total_negative - negative_prefix[cut]
    fn = total_positive - tp
    tn = total_negative - fp
    sensitivity = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
    )
    specificity = np.divide(
        tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0
    )
    balanced_accuracy = (sensitivity + specificity) / 2.0
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0
    )
    f1 = np.divide(
        2.0 * precision * sensitivity,
        precision + sensitivity,
        out=np.zeros_like(precision),
        where=(precision + sensitivity) > 0,
    )
    best = max(
        range(len(thresholds)),
        key=lambda index: (
            balanced_accuracy[index],
            f1[index],
            sensitivity[index],
            specificity[index],
            -abs(float(thresholds[index]) - 0.5),
        ),
    )
    return float(thresholds[best]), {
        "threshold": thresholds,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = probabilities > threshold
    tp = int(np.sum((labels == 1) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    sensitivity = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    precision = _div(tp, tp + fp)
    return {
        "threshold": float(threshold),
        "support": int(len(labels)),
        "normal_support": int(np.sum(labels == 0)),
        "abnormal_support": int(np.sum(labels == 1)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _div(tp + tn, len(labels)),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": _div(2.0 * precision * sensitivity, precision + sensitivity),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _class_metrics(
    true_classes: np.ndarray,
    class_logits: np.ndarray,
    binary_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    supported = true_classes >= 0
    true = true_classes[supported]
    logits = class_logits[supported]
    gate = binary_probabilities[supported] > threshold
    predictions = np.zeros(len(true), dtype=np.int64)
    if np.any(gate):
        # Once the binary gate declares an anomaly, choose only an abnormal
        # class.  This makes the hierarchy mathematically explicit.
        predictions[gate] = 1 + np.argmax(logits[gate, 1:], axis=1)

    matrix = confusion_matrix(
        true, predictions, labels=np.arange(len(ARRHYTHMIA_CLASSES))
    )
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, class_name in enumerate(ARRHYTHMIA_CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        f1 = _div(2.0 * precision * recall, precision + recall)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support:
            f1_values.append(f1)
    abnormal_mask = true != 0
    return {
        "support": int(len(true)),
        "accuracy": float(np.mean(predictions == true)) if len(true) else 0.0,
        "macro_f1_supported": float(np.mean(f1_values)) if f1_values else 0.0,
        "abnormal_identification_rate": (
            float(np.mean(predictions[abnormal_mask] == true[abnormal_mask]))
            if np.any(abnormal_mask)
            else 0.0
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    binary_probabilities: list[np.ndarray] = []
    class_logits: list[np.ndarray] = []
    binary_labels: list[np.ndarray] = []
    class_labels: list[np.ndarray] = []
    for x, binary, classes in loader:
        binary_logit, logits = model(x.to(device))
        binary_probabilities.append(torch.sigmoid(binary_logit).cpu().numpy())
        class_logits.append(logits.cpu().numpy())
        binary_labels.append(binary.numpy())
        class_labels.append(classes.numpy())
    return (
        np.concatenate(binary_probabilities),
        np.concatenate(class_logits),
        np.concatenate(binary_labels),
        np.concatenate(class_labels),
    )


def _make_loader(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset["windows"][indices]).float().unsqueeze(1),
        torch.from_numpy(dataset["binary"][indices]).float(),
        torch.from_numpy(dataset["classes"][indices]).long(),
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def train_hierarchical(
    config: HierarchicalTrainConfig,
    dataset: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available() and config.device.startswith("cuda"):
        device = torch.device(config.device)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    dataset = dataset or load_or_build_dataset(config)
    records = dataset["records"]
    train_indices = _indices_for_records(records, TRAIN_RECORDS)
    validation_indices = _indices_for_records(records, VALIDATION_RECORDS)
    original_test_indices = _indices_for_records(records, ORIGINAL_TEST_RECORDS)
    expanded_test_indices = _indices_for_records(records, EXPANDED_TEST_RECORDS)

    train_loader = _make_loader(
        dataset,
        train_indices,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = _make_loader(
        dataset,
        validation_indices,
        config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )
    original_test_loader = _make_loader(
        dataset,
        original_test_indices,
        config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )
    expanded_test_loader = _make_loader(
        dataset,
        expanded_test_indices,
        config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_hierarchical_model(dropout=config.dropout).to(device)
    binary_train_labels = dataset["binary"][train_indices]
    positive = float(np.sum(binary_train_labels == 1))
    negative = float(np.sum(binary_train_labels == 0))
    binary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / max(positive, 1.0)], device=device)
    )
    class_weights = _effective_number_weights(
        dataset["classes"][train_indices],
        beta=config.class_balance_beta,
        n_classes=len(ARRHYTHMIA_CLASSES),
    )
    class_criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device),
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=2e-5
    )

    best_state: Optional[dict[str, torch.Tensor]] = None
    best_selection = -float("inf")
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_binary = 0.0
        running_class = 0.0
        for x, binary, classes in train_loader:
            x = x.to(device)
            binary = binary.to(device)
            classes = classes.to(device)
            if config.augment:
                x = _augment(x)
            optimizer.zero_grad(set_to_none=True)
            binary_logit, class_logits = model(x)
            binary_loss = binary_criterion(binary_logit, binary)
            supported = classes >= 0
            class_loss = (
                class_criterion(class_logits[supported], classes[supported])
                if torch.any(supported)
                else torch.zeros((), device=device)
            )
            loss = binary_loss + config.class_loss_weight * class_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            running_loss += float(loss.item())
            running_binary += float(binary_loss.item())
            running_class += float(class_loss.item())

        (
            validation_probabilities,
            validation_logits,
            validation_binary,
            validation_classes,
        ) = _predict(model, validation_loader, device)
        threshold, _ = choose_binary_threshold(
            validation_binary, validation_probabilities
        )
        validation_binary_metrics = binary_metrics(
            validation_binary, validation_probabilities, threshold
        )
        validation_class_metrics = _class_metrics(
            validation_classes,
            validation_logits,
            validation_probabilities,
            threshold,
        )
        selection = (
            0.55 * validation_binary_metrics["balanced_accuracy"]
            + 0.45 * validation_class_metrics["macro_f1_supported"]
        )
        scheduler.step(selection)
        epoch_result = {
            "epoch": epoch,
            "loss": running_loss / max(1, len(train_loader)),
            "binary_loss": running_binary / max(1, len(train_loader)),
            "class_loss": running_class / max(1, len(train_loader)),
            "validation_threshold": threshold,
            "validation_balanced_accuracy": validation_binary_metrics[
                "balanced_accuracy"
            ],
            "validation_macro_f1": validation_class_metrics[
                "macro_f1_supported"
            ],
            "selection_score": selection,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_result)
        print(
            f"[hierarchical] epoch {epoch:02d} "
            f"loss={epoch_result['loss']:.4f} "
            f"val_BA={epoch_result['validation_balanced_accuracy']:.4f} "
            f"val_macroF1={epoch_result['validation_macro_f1']:.4f} "
            f"threshold={threshold:.4f}"
        )

        if selection > best_selection + 1e-5:
            best_selection = selection
            best_threshold = threshold
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"[hierarchical] early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No hierarchical model state was selected")
    model.load_state_dict(best_state)
    model.to(device)

    def evaluate(loader: DataLoader) -> dict[str, Any]:
        probabilities, logits, binary_labels, class_labels = _predict(
            model, loader, device
        )
        return {
            "binary": binary_metrics(
                binary_labels, probabilities, best_threshold
            ),
            "hierarchical_classification": _class_metrics(
                class_labels, logits, probabilities, best_threshold
            ),
            "binary_probabilities": probabilities,
            "binary_labels": binary_labels,
            "class_logits": logits,
            "class_labels": class_labels,
        }

    validation_result = evaluate(validation_loader)
    original_test_result = evaluate(original_test_loader)
    expanded_test_result = evaluate(expanded_test_loader)
    model_path = settings.MODEL_DIR / f"hierarchical_{int(time.time())}.pt"
    checkpoint = {
        "state_dict": best_state,
        "architecture": model.architecture_summary(),
        "config": asdict(config),
        "class_names": ARRHYTHMIA_CLASSES,
        "binary_threshold": best_threshold,
        "best_epoch": best_epoch,
        "best_validation_selection": best_selection,
        "history": history,
        "records": {
            "train": TRAIN_RECORDS,
            "validation": VALIDATION_RECORDS,
            "original_test": ORIGINAL_TEST_RECORDS,
            "expanded_test": EXPANDED_TEST_RECORDS,
        },
        "metrics": {
            "validation": {
                "binary": validation_result["binary"],
                "hierarchical_classification": validation_result[
                    "hierarchical_classification"
                ],
            },
            "original_test": {
                "binary": original_test_result["binary"],
                "hierarchical_classification": original_test_result[
                    "hierarchical_classification"
                ],
            },
            "expanded_test": {
                "binary": expanded_test_result["binary"],
                "hierarchical_classification": expanded_test_result[
                    "hierarchical_classification"
                ],
            },
        },
    }
    torch.save(checkpoint, model_path)

    # Arrays are returned for reproducible plotting but are excluded from the
    # checkpoint's JSON-friendly metric section.
    return {
        "model_path": str(model_path),
        "checkpoint": checkpoint,
        "validation_arrays": validation_result,
        "original_test_arrays": original_test_result,
        "expanded_test_arrays": expanded_test_result,
        "dataset_counts": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "original_test": int(len(original_test_indices)),
            "expanded_test": int(len(expanded_test_indices)),
        },
        "class_weights": class_weights.tolist(),
    }


def result_to_json(result: dict[str, Any]) -> str:
    compact = {
        "model_path": result["model_path"],
        "checkpoint": result["checkpoint"],
        "dataset_counts": result["dataset_counts"],
        "class_weights": result["class_weights"],
    }
    return json.dumps(compact, indent=2)
