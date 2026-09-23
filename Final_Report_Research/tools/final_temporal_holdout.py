"""Final experiment: fixed within-record temporal holdout on MLII records.

Protocol
--------
* The normal-only autoencoder checkpoint is frozen. It was developed from the
  complete first stored ECG channel of all 18 MIT-BIH normal-sinus records and
  contributes reconstruction-error features.
* For every eligible MIT-BIH MLII record, the first 80% of accepted beats is the
  development region and the final 20% is the fixed temporal test region.
* Candidate selection uses only an earlier 64% region and a 64--80%
  validation region.  Selected models are refitted on the full first 80%.
* In this rerun, the final 20% does not select candidates or thresholds. The
  segment was observed in earlier system development, so this is not a pristine
  first-use external test. It is later signal from known patients.

The script deliberately reports record/subject-clustered uncertainty.  A
beat-level confidence interval would be falsely narrow because neighbouring
beats from the same recording are correlated.
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import binomtest, wilcoxon
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPORT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = REPORT_DIR.parent
BACKEND_DIR = REPO_DIR / "backend"
sys.path[:0] = [str(BACKEND_DIR), str(Path(__file__).resolve().parent)]

from app.ml.classifier import ARRHYTHMIA_CLASSES  # noqa: E402
from app.ml.feature_system import build_features  # noqa: E402
from app.ml.model import build_model  # noqa: E402
from app.ml.beat_preparation import PREPROCESSING_VERSION  # noqa: E402
from corrected_dataset import load_dataset  # noqa: E402
from app.ml.train_hierarchical import (  # noqa: E402
    HierarchicalTrainConfig,
    choose_binary_threshold,
    load_or_build_dataset,
)


SEED = 20260801
N_BOOTSTRAP = 10_000
TRAIN_FRACTION = 0.80
SELECTION_FRACTION = 0.64
# Roughly 15 seconds at a typical heart rate. The final 20% remains intact;
# these beats are removed only from the end of the training region.
BOUNDARY_GAP_BEATS = 16
RESULT_PATH = REPORT_DIR / "experiments" / "corrected_temporal_holdout.json"
MODEL_PATH = REPORT_DIR / "experiments" / "corrected_temporal_holdout.joblib"
FIGURE_DIR = REPORT_DIR / "pic" / "generated"
FIRST_FEATURE_PATH = BACKEND_DIR / "app/storage/datasets/mitdb/corrected_local_features_v1.npz"
AUTOENCODER_PATH = BACKEND_DIR / "app/storage/models/model_nsrdb_primary_healthy.pt"
AE_FEATURE_PATH = BACKEND_DIR / "app/storage/datasets/mitdb/corrected_local_ae_features_v1.npz"
HEALTHY_WINDOW_DIR = BACKEND_DIR / "app/storage/datasets/nsrdb-primary-full"


def load_first_features(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Load or rebuild the transparent features for the selected MLII beats."""
    if FIRST_FEATURE_PATH.exists():
        cached = np.load(FIRST_FEATURE_PATH)
        if np.array_equal(cached["records"], dataset["records"]):
            if str(cached.get("input_hash", "")) == hashlib.sha256(dataset["windows"].tobytes()).hexdigest():
                return cached["features"]
    features = build_features(dataset["windows"], dataset["rr_features"])
    FIRST_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIRST_FEATURE_PATH, features=features, records=dataset["records"],
        input_hash=hashlib.sha256(dataset["windows"].tobytes()).hexdigest(),
    )
    return features


def causal_rr_context(
    records: np.ndarray,
    rr_features: np.ndarray,
    history_size: int = 20,
) -> np.ndarray:
    """Calculate eight causal rhythm features from current and earlier RR values."""
    result = np.zeros((len(records), 8), dtype=np.float32)
    for record in np.unique(records):
        indices = np.where(records == record)[0]
        intervals = rr_features[indices, 0].astype(np.float64)
        for local_index, global_index in enumerate(indices):
            start = max(0, local_index - history_size + 1)
            history = intervals[start:local_index + 1]
            median = float(np.median(history))
            q25, q75 = np.percentile(history, [25, 75])
            mad = float(np.median(np.abs(history - median)))
            differences = np.diff(history)
            rmssd = float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0
            pnn50 = float(np.mean(np.abs(differences) > 0.05)) if len(differences) else 0.0
            safe_median = max(median, 1e-3)
            range_ratio = float((np.max(history) - np.min(history)) / safe_median)
            trend = (
                float(np.polyfit(np.arange(len(history)), history, 1)[0] / safe_median)
                if len(history) >= 3 else 0.0
            )
            previous = float(intervals[local_index - 1]) if local_index else float(intervals[local_index])
            result[global_index] = np.asarray([
                median, float(q75 - q25), mad, rmssd, pnn50, range_ratio,
                trend, float((intervals[local_index] - previous) / safe_median),
            ], dtype=np.float32)
    return result


def record_class_weights(
    labels: np.ndarray,
    records: np.ndarray,
    record_balance_power: float,
    clip: float = 20.0,
) -> np.ndarray:
    """Balance classes while limiting domination by long patient/class cells."""
    weights = np.ones(len(labels), dtype=np.float64)
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        class_records = records[class_indices]
        unique_records = np.unique(class_records)
        class_count = len(class_indices)
        for record in unique_records:
            cell = class_indices[class_records == record]
            factor = (
                class_count / (len(unique_records) * max(1, len(cell)))
            ) ** record_balance_power
            weights[cell] = factor / max(1, class_count)
    weights /= max(np.mean(weights), 1e-12)
    return np.clip(weights, 1.0 / clip, clip).astype(np.float64)


def subject_groups(records: np.ndarray) -> np.ndarray:
    """Map MITDB records to subjects; records 201 and 202 share one subject."""
    groups = records.astype("<U7", copy=True)
    groups[records == "201"] = "201-202"
    groups[records == "202"] = "201-202"
    return groups


def multiclass_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    labels = np.arange(len(ARRHYTHMIA_CLASSES))
    matrix = confusion_matrix(true, predicted, labels=labels)
    per_f1 = f1_score(true, predicted, labels=labels, average=None, zero_division=0)
    per_precision = precision_score(true, predicted, labels=labels, average=None, zero_division=0)
    per_recall = recall_score(true, predicted, labels=labels, average=None, zero_division=0)
    return {
        "support": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "support": int(np.sum(true == index)),
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
            }
            for index, name in enumerate(ARRHYTHMIA_CLASSES)
        },
    }


def binary_metrics(
    true: np.ndarray,
    probability: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "support": int(len(true)),
        "normal_support": int(np.sum(true == 0)),
        "abnormal_support": int(np.sum(true == 1)),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "sensitivity": float(recall_score(true, predicted, pos_label=1)),
        "specificity": float(recall_score(true, predicted, pos_label=0)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(true, probability)),
        "auprc": float(average_precision_score(true, probability)),
        "brier": float(brier_score_loss(true, probability)),
    }


def load_autoencoder_features(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Load or build the four residual features required by the final model."""
    if AE_FEATURE_PATH.exists():
        cached = np.load(AE_FEATURE_PATH)
        if np.array_equal(cached["records"], dataset["records"]):
            if str(cached.get("input_hash", "")) == hashlib.sha256(dataset["windows"].tobytes()).hexdigest():
                return cached["features"]
    checkpoint = torch.load(AUTOENCODER_PATH, map_location="cpu", weights_only=False)
    model = build_model(
        input_length=512,
        skip_scale=float(checkpoint.get("config", {}).get("skip_scale", 0.5)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(dataset["windows"]).float().unsqueeze(1)),
        batch_size=256,
        shuffle=False,
    )
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, (x,) in enumerate(loader):
            reconstruction = model(x)
            residual = x - reconstruction
            absolute = residual.abs()
            rows.append(torch.stack([
                absolute.mean(dim=(1, 2)),
                residual.square().mean(dim=(1, 2)),
                absolute[:, :, 104:344].mean(dim=(1, 2)),
                torch.diff(residual, dim=2).abs().mean(dim=(1, 2)),
            ], dim=1).cpu().numpy().astype(np.float32))
            if batch_index % 25 == 0:
                print(f"[residuals] {min((batch_index + 1) * 256, len(dataset['windows']))}/{len(dataset['windows'])}", flush=True)
    features = np.concatenate(rows)
    np.savez_compressed(AE_FEATURE_PATH, features=features, records=dataset["records"],
                        input_hash=hashlib.sha256(dataset["windows"].tobytes()).hexdigest())
    return features


def create_autoencoder_example_figure(dataset: dict[str, np.ndarray]) -> None:
    """Show the frozen NSRDB autoencoder on real healthy and MLII beats."""
    healthy_files = sorted(HEALTHY_WINDOW_DIR.glob("*_primary_full.npy"))
    if not healthy_files:
        raise RuntimeError("No cached healthy NSRDB windows for the report figure")
    healthy = np.load(healthy_files[0])[0].astype(np.float32)
    abnormal_indices = np.where(dataset["classes"] == 1)[0]
    if not len(abnormal_indices):
        raise RuntimeError("No PVC example is available for the report figure")
    abnormal = dataset["windows"][abnormal_indices[len(abnormal_indices) // 2]].astype(np.float32)
    checkpoint = torch.load(AUTOENCODER_PATH, map_location="cpu", weights_only=False)
    model = build_model(
        input_length=512,
        skip_scale=float(checkpoint.get("config", {}).get("skip_scale", 0.5)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    examples = np.stack([healthy, abnormal])
    with torch.no_grad():
        reconstructed = model(torch.from_numpy(examples).float().unsqueeze(1)).squeeze(1).numpy()
    time_axis = np.arange(512) / 128.0
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8), sharex=True)
    for row, (title, colour) in enumerate([
        ("Healthy MIT-BIH NSRDB first ECG channel", "#0f766e"),
        ("Abnormal MITDB MLII example: PVC", "#dc2626"),
    ]):
        axes[row, 0].plot(time_axis, examples[row], color=colour, linewidth=1.0, label="input")
        axes[row, 0].plot(time_axis, reconstructed[row], color="#2563eb", linewidth=0.9, alpha=.85, label="reconstruction")
        axes[row, 0].set_title(title, loc="left", fontsize=9, weight="bold")
        axes[row, 0].set_ylabel("standardised amplitude")
        axes[row, 0].grid(alpha=.18)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
        residual = np.abs(examples[row] - reconstructed[row])
        axes[row, 1].fill_between(time_axis, residual, color="#d97706", alpha=.55)
        axes[row, 1].plot(time_axis, residual, color="#b45309", linewidth=.75)
        axes[row, 1].set_title(f"absolute residual; mean = {residual.mean():.3f}", loc="left", fontsize=9, weight="bold")
        axes[row, 1].set_ylabel("absolute error")
        axes[row, 1].grid(alpha=.18)
    for axis in axes[-1]:
        axis.set_xlabel("time (seconds)")
    fig.suptitle("Real one-channel examples through the frozen normal autoencoder", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "real_autoencoder_processing.pdf", bbox_inches="tight")
    plt.close(fig)


def temporal_split(records: np.ndarray) -> dict[str, np.ndarray]:
    development: list[int] = []
    selection_train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    audit: dict[str, Any] = {}
    for record in np.unique(records):
        indices = np.where(records == record)[0]
        cut_64 = int(np.floor(SELECTION_FRACTION * len(indices)))
        cut_80 = int(np.floor(TRAIN_FRACTION * len(indices)))
        development_end = max(0, cut_80 - BOUNDARY_GAP_BEATS)
        selection_end = max(0, cut_64 - BOUNDARY_GAP_BEATS)
        selection_train.extend(indices[:selection_end])
        validation.extend(indices[cut_64:development_end])
        development.extend(indices[:development_end])
        test.extend(indices[cut_80:])
        audit[str(record)] = {
            "accepted_beats": int(len(indices)),
            "development_beats_after_gap": int(development_end),
            "boundary_gap_beats": int(cut_80 - development_end),
            "temporal_test_beats": int(len(indices) - cut_80),
        }
    return {
        "development": np.asarray(development, dtype=np.int64),
        "selection_train": np.asarray(selection_train, dtype=np.int64),
        "validation": np.asarray(validation, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
        "audit": audit,
    }


def fit_tree(
    features: np.ndarray,
    labels: np.ndarray,
    records: np.ndarray,
    indices: np.ndarray,
    candidate: dict[str, Any],
    seed: int,
) -> ExtraTreesClassifier:
    weights = record_class_weights(
        labels[indices], subject_groups(records[indices]), candidate["record_power"]
    )
    model = ExtraTreesClassifier(
        n_estimators=int(candidate["trees"]),
        max_depth=candidate["max_depth"],
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        max_features=candidate["max_features"],
        criterion="entropy",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features[indices], labels[indices], sample_weight=weights)
    return model


def subtype_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    labels = np.arange(1, len(ARRHYTHMIA_CLASSES))
    return {
        "support": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
    }


def select_models(
    dataset: dict[str, np.ndarray],
    records: np.ndarray,
    binary_features: np.ndarray,
    class_features: np.ndarray,
    split: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], float, list[dict[str, Any]], list[dict[str, Any]]]:
    train = split["selection_train"]
    validation = split["validation"]
    binary_candidates = [
        {"name": "depth14", "trees": 160, "max_depth": 14, "min_samples_leaf": 2, "max_features": "sqrt", "record_power": .5},
        {"name": "depth22", "trees": 200, "max_depth": 22, "min_samples_leaf": 2, "max_features": "sqrt", "record_power": .5},
        {"name": "deep", "trees": 240, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "record_power": .35},
    ]
    class_candidates = [
        {"name": "depth22", "trees": 200, "max_depth": 22, "min_samples_leaf": 2, "max_features": "sqrt", "record_power": .35},
        {"name": "deep_sqrt", "trees": 260, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "record_power": .25},
        {"name": "deep_third", "trees": 260, "max_depth": None, "min_samples_leaf": 1, "max_features": .33, "record_power": .25},
    ]

    binary_rows: list[dict[str, Any]] = []
    best_binary: tuple[tuple[float, float], dict[str, Any], float] | None = None
    for number, candidate in enumerate(binary_candidates):
        model = fit_tree(binary_features, dataset["binary"], records, train, candidate, SEED + number)
        probability = model.predict_proba(binary_features[validation])[:, 1]
        threshold, _ = choose_binary_threshold(dataset["binary"][validation], probability)
        metric = binary_metrics(dataset["binary"][validation], probability, probability > threshold)
        rank = (metric["balanced_accuracy"], metric["f1"])
        binary_rows.append({"candidate": candidate, "threshold": threshold, "validation": metric})
        if best_binary is None or rank > best_binary[0]:
            best_binary = (rank, candidate, threshold)
        print(f"[binary] {candidate['name']} validation BA={metric['balanced_accuracy']:.4f}", flush=True)

    abnormal_train = train[dataset["classes"][train] > 0]
    abnormal_validation = validation[dataset["classes"][validation] > 0]
    class_rows: list[dict[str, Any]] = []
    best_class: tuple[tuple[float, float], dict[str, Any]] | None = None
    for number, candidate in enumerate(class_candidates):
        model = fit_tree(class_features, dataset["classes"], records, abnormal_train, candidate, SEED + 20 + number)
        prediction = model.predict(class_features[abnormal_validation])
        metric = subtype_metrics(dataset["classes"][abnormal_validation], prediction)
        rank = (metric["macro_f1"], metric["accuracy"])
        class_rows.append({"candidate": candidate, "validation": metric})
        if best_class is None or rank > best_class[0]:
            best_class = (rank, candidate)
        print(f"[class] {candidate['name']} validation macro-F1={metric['macro_f1']:.4f}", flush=True)
    assert best_binary is not None and best_class is not None
    return best_binary[1], best_class[1], best_binary[2], binary_rows, class_rows


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def clustered_intervals(
    records: np.ndarray,
    truth_binary: np.ndarray,
    pred_binary: np.ndarray,
    truth_class: np.ndarray,
    pred_subtype: np.ndarray,
    pred_system: np.ndarray,
) -> dict[str, Any]:
    subjects = subject_groups(records)
    unique_subjects = np.unique(subjects)
    rows = []
    for subject in unique_subjects:
        mask = subjects == subject
        supported = mask & (truth_class >= 0)
        abnormal = mask & (truth_class > 0)
        subtype_value = (
            float(np.mean(pred_subtype[abnormal] == truth_class[abnormal]))
            if np.any(abnormal) else None
        )
        rows.append({
            "subject": str(subject),
            "binary": float(np.mean(pred_binary[mask] == truth_binary[mask])),
            "subtype": subtype_value,
            "whole": float(np.mean(pred_system[supported] == truth_class[supported])) if np.any(supported) else np.nan,
        })
    rng = np.random.default_rng(SEED)
    samples = {"binary": [], "subtype": [], "whole": []}
    for _ in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(rows), len(rows))
        for metric in samples:
            values = np.asarray([
                np.nan if rows[index][metric] is None else rows[index][metric]
                for index in chosen
            ], dtype=float)
            samples[metric].append(float(np.nanmean(values)))
    point = {
        metric: float(np.nanmean([
            np.nan if row[metric] is None else row[metric] for row in rows
        ]))
        for metric in samples
    }
    whole_values = np.asarray([row["whole"] for row in rows], dtype=float)
    non_ties = whole_values[whole_values != .90]
    sign_result = binomtest(
        int(np.sum(non_ties > .90)), len(non_ties), .5, alternative="greater"
    )
    signed_rank = wilcoxon(
        whole_values - .90, alternative="greater", zero_method="wilcox", method="approx"
    )
    return {
        "method": "percentile bootstrap of complete subjects; each subject has equal weight",
        "replicates": N_BOOTSTRAP,
        "seed": SEED,
        "independent_subjects": int(len(rows)),
        "point_estimate_subject_macro": point,
        "ci_95": {metric: percentile_ci(np.asarray(values)) for metric, values in samples.items()},
        "ci_width": {metric: float(np.diff(percentile_ci(np.asarray(values)))[0]) for metric, values in samples.items()},
        "whole_system_tests_against_90_percent": {
            "exact_sign_test": {
                "null": "no more than half of subjects exceed 90% accuracy",
                "subjects_above_90": int(np.sum(whole_values > .90)),
                "subjects_below_90": int(np.sum(whole_values < .90)),
                "p_value_one_sided": float(sign_result.pvalue),
            },
            "wilcoxon_signed_rank": {
                "null": "subject accuracy is not shifted above 90%",
                "statistic": float(signed_rank.statistic),
                "p_value_one_sided": float(signed_rank.pvalue),
                "note": "secondary test; the clustered confidence interval is primary",
            },
        },
        "per_subject": rows,
    }


def arrhythmia_table(true: np.ndarray, binary: np.ndarray, subtype: np.ndarray, system: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for label, name in enumerate(ARRHYTHMIA_CLASSES[1:], start=1):
        mask = true == label
        support = int(mask.sum())
        rows.append({
            "type": name,
            "test_beats": support,
            "detected_abnormal": int(np.sum(binary[mask])),
            "correct_subtype": int(np.sum(subtype[mask] == label)),
            "correct_end_to_end": int(np.sum(system[mask] == label)),
            "detection_recall": float(np.mean(binary[mask])) if support else 0.0,
            "subtype_accuracy": float(np.mean(subtype[mask] == label)) if support else 0.0,
            "end_to_end_recall": float(np.mean(system[mask] == label)) if support else 0.0,
        })
    return rows


def create_figures(result: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    table = result["arrhythmia_table"]
    names = [row["type"] for row in table]
    total = np.asarray([row["test_beats"] for row in table])
    correct = np.asarray([row["correct_end_to_end"] for row in table])
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.bar(names, total, color="#cbd5e1", label="Labelled test beats")
    ax.bar(names, correct, color="#0f766e", label="Correct end-to-end")
    ax.set(title="Final model: fixed 20% temporal holdout", ylabel="Beat count")
    ax.grid(axis="y", alpha=.2); ax.legend()
    for index, row in enumerate(table):
        ax.text(index, total[index] + max(total) * .015, f"{100*row['end_to_end_recall']:.1f}%", ha="center", fontsize=8, weight="bold")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "final_arrhythmia_counts.pdf", bbox_inches="tight"); plt.close(fig)

    metrics = result["headline"]
    labels = ["Abnormality\ndetection", "Conditional subtype\nclassification", "Whole-system\naccuracy", "Whole-system\nmacro F1"]
    values = [metrics["binary_accuracy"], metrics["subtype_accuracy"], metrics["whole_system_accuracy"], metrics["whole_system_macro_f1"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(labels, values, color=["#2563eb", "#7c3aed", "#0f766e", "#d97706"])
    ax.axhline(.90, color="#dc2626", linestyle="--", label="90% target")
    ax.set_ylim(0, 1.03); ax.set_ylabel("Pooled test score"); ax.set_title("Final model temporal-test results")
    ax.grid(axis="y", alpha=.2); ax.legend()
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.015, f"{100*value:.2f}%", ha="center", weight="bold")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "final_metrics.pdf", bbox_inches="tight"); plt.close(fig)

    matrix = np.asarray(result["test_whole_system"]["confusion_matrix"], dtype=float)
    row_total = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    row_percent = 100.0 * matrix / row_total
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            colour = "white" if row_percent[row, column] > 52 else "#111827"
            ax.text(column, row, f"{int(matrix[row, column]):,}\n{row_percent[row, column]:.1f}%", ha="center", va="center", fontsize=8, color=colour)
    ax.set_xticks(range(len(ARRHYTHMIA_CLASSES)), ARRHYTHMIA_CLASSES)
    ax.set_yticks(range(len(ARRHYTHMIA_CLASSES)), ARRHYTHMIA_CLASSES)
    ax.set_xlabel("System output"); ax.set_ylabel("Reference label")
    ax.set_title("Final model whole-system confusion matrix")
    fig.colorbar(image, ax=ax, label="Percentage within reference class")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "final_confusion_matrix.pdf", bbox_inches="tight"); plt.close(fig)

    uncertainty = result["clustered_uncertainty"]
    keys = ["binary", "subtype", "whole"]
    ci_labels = ["Abnormality\ndetection", "Conditional subtype\nclassification", "Whole system"]
    points = np.asarray([uncertainty["point_estimate_subject_macro"][key] for key in keys])
    lower = np.asarray([uncertainty["ci_95"][key][0] for key in keys])
    upper = np.asarray([uncertainty["ci_95"][key][1] for key in keys])
    positions = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.errorbar(positions, points, yerr=np.vstack([points-lower, upper-points]), fmt="o", markersize=8, capsize=7, color="#0f766e", ecolor="#2563eb", linewidth=2)
    ax.axhline(.90, color="#dc2626", linestyle="--", label="90% target")
    ax.set_xticks(positions, ci_labels); ax.set_ylim(.88, 1.01)
    ax.set_ylabel("Equal-subject accuracy"); ax.set_title("Subject-clustered 95% confidence intervals")
    ax.grid(axis="y", alpha=.2); ax.legend(loc="lower left")
    for x, point, lo, hi in zip(positions, points, lower, upper):
        ax.text(x, hi+.004, f"{100*point:.2f}%\n[{100*lo:.2f}, {100*hi:.2f}]", ha="center", fontsize=8, weight="bold")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "final_clustered_ci.pdf", bbox_inches="tight"); plt.close(fig)

    subject_rows = uncertainty["per_subject"]
    subject_names = [row["subject"] for row in subject_rows]
    subject_values = np.asarray([row["whole"] for row in subject_rows])
    order = np.argsort(subject_values)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    colours = np.where(subject_values[order] >= .90, "#0f766e", "#dc2626")
    ax.bar(np.arange(len(order)), subject_values[order], color=colours, width=.82)
    ax.axhline(.90, color="#111827", linestyle="--", linewidth=1.3, label="90% target")
    ax.set_xticks(np.arange(len(order)), np.asarray(subject_names)[order], rotation=70, fontsize=7)
    ax.set_ylim(0, 1.03); ax.set_ylabel("Whole-system accuracy")
    ax.set_xlabel("MIT-BIH subject/record cluster"); ax.set_title(f"Performance across the {len(subject_rows)} known-patient temporal test clusters")
    ax.grid(axis="y", alpha=.2); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "final_subject_accuracy.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    started = time.time()
    torch.set_num_threads(4)
    dataset = load_dataset()
    records = dataset["records"]
    split = temporal_split(records)
    first = load_first_features(dataset)
    rr = dataset["rr15"][:, 7:]
    ae = load_autoencoder_features(dataset)
    binary_features = np.concatenate([first, rr, ae[:, :4]], axis=1).astype(np.float32)
    # Both supervised stages use the same single-lead evidence vector:
    # channel-1 morphology/spectrum (223), causal RR context (15), and four
    # frozen-autoencoder residual summaries = 242 values.  No second channel
    # is read during training, validation, temporal testing, or deployment.
    class_features = binary_features

    # Check raw window separation, including a conservative resampling margin.
    boundary_separation = {}
    for record in np.unique(records):
        train_indices = split["development"][records[split["development"]] == record]
        test_indices = split["test"][records[split["test"]] == record]
        selection_indices = split["selection_train"][records[split["selection_train"]] == record]
        validation_indices = split["validation"][records[split["validation"]] == record]
        gap = int(dataset["peaks"][test_indices[0]] - dataset["peaks"][train_indices[-1]] - 512)
        selection_gap = int(dataset["peaks"][validation_indices[0]] - dataset["peaks"][selection_indices[-1]] - 512)
        if min(gap, selection_gap) < 64:
            raise ValueError(f"Insufficient sample separation in record {record}")
        boundary_separation[str(record)] = {"test_gap_samples": gap, "selection_gap_samples": selection_gap}

    best_binary, best_class, threshold, binary_selection, class_selection = select_models(
        dataset, records, binary_features, class_features, split
    )
    development = split["development"]
    final_binary = fit_tree(binary_features, dataset["binary"], records, development, best_binary, SEED + 100)
    abnormal_development = development[dataset["classes"][development] > 0]
    final_class = fit_tree(class_features, dataset["classes"], records, abnormal_development, best_class, SEED + 101)

    test = split["test"]
    probability = final_binary.predict_proba(binary_features[test])[:, 1]
    binary_prediction = probability > threshold
    subtype_prediction = final_class.predict(class_features[test])
    whole_prediction = np.zeros(len(test), dtype=np.int64)
    whole_prediction[binary_prediction] = subtype_prediction[binary_prediction]
    supported = dataset["classes"][test] >= 0
    abnormal_supported = dataset["classes"][test] > 0

    test_binary = binary_metrics(dataset["binary"][test], probability, binary_prediction)
    test_subtype = subtype_metrics(dataset["classes"][test][abnormal_supported], subtype_prediction[abnormal_supported])
    test_whole = multiclass_metrics(dataset["classes"][test][supported], whole_prediction[supported])
    intervals = clustered_intervals(
        records[test], dataset["binary"][test], binary_prediction,
        dataset["classes"][test], subtype_prediction, whole_prediction,
    )
    table = arrhythmia_table(dataset["classes"][test], binary_prediction, subtype_prediction, whole_prediction)

    result = {
        "protocol": {
            "name": "Final within-patient temporal holdout",
            "lead_design": "one channel at every stage: the first stored NSRDB ECG channel for normal-only autoencoder development and MLII for both supervised stages",
            "autoencoder_training": "frozen checkpoint developed from complete recordings of all 18 MIT-BIH normal-sinus subjects; 15 subjects for fitting and 3 different subjects for validation",
            "classifier_training": "first 80% of chronologically ordered accepted beats per MLII record, less 16 beats",
            "temporal_test": "last 20% of accepted beats per record (not exact elapsed time)",
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing": "resample to 128 Hz; bandpass and z-score within each 512-sample window",
            "boundary_sample_audit": boundary_separation,
            "excluded_records": {"102": "no MLII; V5 and V2 only", "104": "no MLII; V5 and V2 only"},
            "records": int(len(np.unique(records))),
            "independent_subjects": int(len(np.unique(subject_groups(records)))),
            "patient_overlap": True,
            "signal_overlap": False,
            "test_used_in_this_rerun_selection": False,
            "test_previously_observed_in_earlier_development": True,
            "generalization_claim": "retrospective known-patient temporal rerun; test previously observed; not confirmatory or unseen-patient validation",
            "split_audit": split["audit"],
        },
        "selected_binary": {"candidate": best_binary, "threshold_from_64_to_80_validation": threshold},
        "dataset_counts": {"accepted": len(records), "development": len(development),
                           "candidate_training": len(split["selection_train"]), "validation": len(split["validation"]),
                           "subtype_development": len(abnormal_development),
                           "candidate_subtype": int(np.sum(dataset["classes"][split["selection_train"]] > 0)),
                           "validation_subtype": int(np.sum(dataset["classes"][split["validation"]] > 0))},
        "selected_subtype": best_class,
        "binary_selection": binary_selection,
        "subtype_selection": class_selection,
        "test_binary": test_binary,
        "test_subtype_on_true_supported_arrhythmias": test_subtype,
        "test_whole_system": test_whole,
        "arrhythmia_table": table,
        "clustered_uncertainty": intervals,
        "headline": {
            "binary_accuracy": test_binary["accuracy"],
            "binary_balanced_accuracy": test_binary["balanced_accuracy"],
            "subtype_accuracy": test_subtype["accuracy"],
            "subtype_macro_f1": test_subtype["macro_f1"],
            "whole_system_accuracy": test_whole["accuracy"],
            "whole_system_macro_f1": test_whole["macro_f1"],
            "whole_system_subject_macro_accuracy": intervals["point_estimate_subject_macro"]["whole"],
            "whole_system_subject_clustered_ci_95": intervals["ci_95"]["whole"],
            "whole_system_ci_width": intervals["ci_width"]["whole"],
            "target_point_accuracy_met": bool(test_whole["accuracy"] >= .90),
            "target_lower_ci_above_90_met": bool(intervals["ci_95"]["whole"][0] > .90),
            "target_ci_width_at_most_3pp_met": bool(intervals["ci_width"]["whole"] <= .03),
        },
        "artifacts": {"model": str(MODEL_PATH), "result": str(RESULT_PATH)},
        "elapsed_seconds": time.time() - started,
    }
    bundle = {
        "system_kind": "final_mlii_temporal_holdout",
        "binary_model": final_binary,
        "class_model": final_class,
        "subtype_model": final_class,
        "binary_threshold": threshold,
        "selected_binary": best_binary,
        "selected_subtype": best_class,
        "classes": ARRHYTHMIA_CLASSES,
        "class_names": ARRHYTHMIA_CLASSES,
        "feature_version": "mlii-morphology-spectrum-rr15-nsrdb-primary-ae4-temporal-v4",
        "preprocessing_version": PREPROCESSING_VERSION,
        "autoencoder_sha256": hashlib.sha256(AUTOENCODER_PATH.read_bytes()).hexdigest(),
        "feature_count": int(binary_features.shape[1]),
        "lead_count": 1,
        "lead_input": "MLII only; selected by signal name",
        "required_lead": "MLII",
        "normal_autoencoder_input": "first stored MIT-BIH NSRDB ECG channel",
        "normal_autoencoder_wfdb_channel_name": "ECG1",
        "rr_feature_count": 15,
        "requires_ae_features": True,
        "ae_feature_count": 4,
        "supports_single_lead_fallback": False,
        "reject_low_confidence": False,
        "clinical_use": False,
        "patient_overlap_in_evaluation": True,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    np.savez_compressed(REPORT_DIR / "experiments/corrected_test_predictions.npz",
                        records=records[test], peaks=dataset["peaks"][test],
                        truth_binary=dataset["binary"][test], truth_class=dataset["classes"][test],
                        probability=probability, pred_binary=binary_prediction,
                        pred_subtype=subtype_prediction, pred_system=whole_prediction)
    result["artifacts"]["model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    result["artifacts"]["autoencoder_sha256"] = bundle["autoencoder_sha256"]
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    create_figures(result)
    # Existing illustration is retained; quantitative plots use this rerun.
    print(json.dumps(result["headline"], indent=2), flush=True)


if __name__ == "__main__":
    main()
