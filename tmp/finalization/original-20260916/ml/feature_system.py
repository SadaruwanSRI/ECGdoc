"""Evidence-driven hierarchical ECG system using morphology and RR features.

The feature set follows the central idea of de Chazal et al.: combine beat
morphology with heartbeat-interval context.  Two record-wise supervised models
are trained:

* a binary normal/abnormal detector using every valid binary annotation;
* a six-class rhythm classifier using only labels supported by this project.

The final decision is hierarchical.  A beat is N below the binary threshold;
otherwise the highest-probability abnormal class is returned.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from app.core.config import settings
from app.ml.classifier import ARRHYTHMIA_CLASSES
from app.ml.train_hierarchical import (
    EXPANDED_TEST_RECORDS,
    ORIGINAL_TEST_RECORDS,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
    HierarchicalTrainConfig,
    _class_metrics,
    _indices_for_records,
    binary_metrics,
    choose_binary_threshold,
    load_or_build_dataset,
)


@dataclass
class FeatureSystemConfig:
    seed: int = 42
    binary_trees: int = 120
    class_trees: int = 160
    n_jobs: int = -1
    cache_features: str = str(
        settings.DATASET_DIR / "mitdb" / "hierarchical_features_mlii_v1.npz"
    )


def _pool(signal: np.ndarray, bins: int) -> np.ndarray:
    usable = (signal.shape[1] // bins) * bins
    return signal[:, :usable].reshape(len(signal), bins, usable // bins).mean(axis=2)


def build_features(windows: np.ndarray, rr_features: np.ndarray) -> np.ndarray:
    """Create transparent morphology, change, spectrum, and RR features."""
    full_pool = _pool(windows, 64)
    # The annotated R peak is at index 200. This range includes QRS and T-wave
    # morphology while retaining some pre-QRS context.
    local = windows[:, 104:344]
    local_pool = _pool(local, 80)
    derivative_pool = _pool(np.diff(local, axis=1), 60)

    spectrum = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(windows.shape[1], d=1.0 / 128.0)
    bands = [(0.5, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 30), (30, 40), (40, 50)]
    band_power = []
    total_power = np.maximum(
        spectrum[:, (frequencies >= 0.5) & (frequencies <= 50)].sum(axis=1),
        1e-8,
    )
    for low, high in bands:
        power = spectrum[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
        band_power.append(np.log1p(power / total_power))
    band_power_array = np.stack(band_power, axis=1)

    derivative = np.diff(windows, axis=1)
    statistics = np.stack(
        [
            windows.mean(axis=1),
            windows.std(axis=1),
            windows.min(axis=1),
            windows.max(axis=1),
            np.ptp(windows, axis=1),
            np.mean(np.abs(windows), axis=1),
            np.sqrt(np.mean(windows ** 2, axis=1)),
            derivative.std(axis=1),
            np.max(np.abs(derivative), axis=1),
            skew(windows, axis=1, bias=False),
            kurtosis(windows, axis=1, fisher=True, bias=False),
        ],
        axis=1,
    )
    statistics = np.nan_to_num(statistics, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate(
        [
            full_pool,
            local_pool,
            derivative_pool,
            band_power_array,
            statistics,
            rr_features,
        ],
        axis=1,
    ).astype(np.float32)


def extend_rr_features(
    base_rr_features: np.ndarray,
    previous_rr_history: np.ndarray,
) -> np.ndarray:
    """Append eight causal rhythm-context measurements to the seven base values.

    ``previous_rr_history`` contains intervals that have already ended at or
    before the beat being scored. Therefore these measurements can be
    reproduced during streaming without seeing a later beat.
    """
    base = np.asarray(base_rr_features, dtype=np.float32).reshape(-1)
    if base.shape != (7,):
        raise ValueError(f"Expected seven base RR features, got {base.shape}")
    history = np.asarray(previous_rr_history, dtype=np.float64).reshape(-1)
    if history.size == 0:
        history = np.asarray([float(base[0])], dtype=np.float64)
    history = history[-20:]
    median = float(np.median(history))
    q25, q75 = np.percentile(history, [25, 75])
    mad = float(np.median(np.abs(history - median)))
    differences = np.diff(history)
    rmssd = (
        float(np.sqrt(np.mean(differences ** 2)))
        if differences.size
        else 0.0
    )
    pnn50 = (
        float(np.mean(np.abs(differences) > 0.05))
        if differences.size
        else 0.0
    )
    safe_median = max(median, 1e-3)
    range_ratio = float((np.max(history) - np.min(history)) / safe_median)
    if history.size >= 3:
        positions = np.arange(history.size, dtype=np.float64)
        trend = float(np.polyfit(positions, history, 1)[0] / safe_median)
    else:
        trend = 0.0
    previous_interval = (
        float(history[-2]) if history.size > 1 else float(history[-1])
    )
    delta_previous = float((history[-1] - previous_interval) / safe_median)
    context = np.asarray(
        [
            median,
            float(q75 - q25),
            mad,
            rmssd,
            pnn50,
            range_ratio,
            trend,
            delta_previous,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, context]).astype(np.float32)


def load_or_build_features(
    dataset: dict[str, np.ndarray],
    config: FeatureSystemConfig,
    force_rebuild: bool = False,
) -> np.ndarray:
    path = Path(config.cache_features)
    if path.exists() and not force_rebuild:
        cached = np.load(path)
        if np.array_equal(cached["records"], dataset["records"]):
            return cached["features"]
    features = build_features(dataset["windows"], dataset["rr_features"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, records=dataset["records"])
    return features


def _supported_macro_f1(
    true: np.ndarray,
    predictions: np.ndarray,
) -> float:
    matrix = confusion_matrix(
        true, predictions, labels=np.arange(len(ARRHYTHMIA_CLASSES))
    )
    f1_values = []
    for index in range(len(ARRHYTHMIA_CLASSES)):
        support = matrix[index].sum()
        if not support:
            continue
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_values.append(
            2 * precision * recall / max(1e-8, precision + recall)
        )
    return float(np.mean(f1_values)) if f1_values else 0.0


def _fit_binary_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: FeatureSystemConfig,
) -> tuple[ExtraTreesClassifier, float, list[dict[str, Any]]]:
    candidates = [
        {"max_depth": 14, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"max_depth": 20, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"max_depth": None, "min_samples_leaf": 5, "max_features": 0.5},
    ]
    results: list[dict[str, Any]] = []
    best: tuple[float, float, ExtraTreesClassifier, float] | None = None
    for candidate in candidates:
        model = ExtraTreesClassifier(
            n_estimators=config.binary_trees,
            class_weight="balanced",
            random_state=config.seed,
            n_jobs=config.n_jobs,
            criterion="entropy",
            bootstrap=False,
            **candidate,
        )
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        threshold, _ = choose_binary_threshold(y_validation, probability)
        metrics = binary_metrics(y_validation, probability, threshold)
        result = {**candidate, "threshold": threshold, **metrics}
        results.append(result)
        rank = (metrics["balanced_accuracy"], metrics["f1"])
        if best is None or rank > best[:2]:
            best = (*rank, model, threshold)
        print(
            "[feature:binary] "
            f"depth={candidate['max_depth']} leaf={candidate['min_samples_leaf']} "
            f"val_BA={metrics['balanced_accuracy']:.4f} "
            f"val_F1={metrics['f1']:.4f}"
        )
    assert best is not None
    return best[2], best[3], results


def _fit_class_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: FeatureSystemConfig,
) -> tuple[ExtraTreesClassifier, list[dict[str, Any]]]:
    candidates = [
        {"max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"max_depth": 24, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
    ]
    results: list[dict[str, Any]] = []
    best_score = -1.0
    best_accuracy = -1.0
    best_model: ExtraTreesClassifier | None = None
    for candidate in candidates:
        model = ExtraTreesClassifier(
            n_estimators=config.class_trees,
            class_weight="balanced",
            random_state=config.seed,
            n_jobs=config.n_jobs,
            criterion="entropy",
            bootstrap=False,
            **candidate,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        macro_f1 = _supported_macro_f1(y_validation, prediction)
        accuracy = float(np.mean(prediction == y_validation))
        results.append({**candidate, "macro_f1": macro_f1, "accuracy": accuracy})
        print(
            "[feature:class] "
            f"depth={candidate['max_depth']} leaf={candidate['min_samples_leaf']} "
            f"val_macroF1={macro_f1:.4f} val_accuracy={accuracy:.4f}"
        )
        if (macro_f1, accuracy) > (best_score, best_accuracy):
            best_score = macro_f1
            best_accuracy = accuracy
            best_model = model
    assert best_model is not None
    return best_model, results


def _aligned_class_probabilities(
    model: ExtraTreesClassifier,
    features: np.ndarray,
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    aligned = np.zeros((len(features), len(ARRHYTHMIA_CLASSES)), dtype=np.float64)
    aligned[:, model.classes_.astype(int)] = probabilities
    return aligned


def _evaluate(
    binary_model: ExtraTreesClassifier,
    class_model: ExtraTreesClassifier,
    threshold: float,
    features: np.ndarray,
    binary_labels: np.ndarray,
    class_labels: np.ndarray,
) -> dict[str, Any]:
    binary_probability = binary_model.predict_proba(features)[:, 1]
    class_probability = _aligned_class_probabilities(class_model, features)
    return {
        "binary": binary_metrics(binary_labels, binary_probability, threshold),
        "hierarchical_classification": _class_metrics(
            class_labels,
            class_probability,
            binary_probability,
            threshold,
        ),
        "binary_probability": binary_probability,
        "class_probability": class_probability,
    }


def train_feature_system(
    config: FeatureSystemConfig,
    dataset: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    dataset = dataset or load_or_build_dataset(HierarchicalTrainConfig())
    features = load_or_build_features(dataset, config)
    records = dataset["records"]
    train = _indices_for_records(records, TRAIN_RECORDS)
    validation = _indices_for_records(records, VALIDATION_RECORDS)
    original_test = _indices_for_records(records, ORIGINAL_TEST_RECORDS)
    expanded_test = _indices_for_records(records, EXPANDED_TEST_RECORDS)

    binary_model, threshold, binary_search = _fit_binary_candidates(
        features[train],
        dataset["binary"][train],
        features[validation],
        dataset["binary"][validation],
        config,
    )
    train_supported = train[dataset["classes"][train] >= 0]
    validation_supported = validation[dataset["classes"][validation] >= 0]
    class_model, class_search = _fit_class_candidates(
        features[train_supported],
        dataset["classes"][train_supported],
        features[validation_supported],
        dataset["classes"][validation_supported],
        config,
    )

    validation_result = _evaluate(
        binary_model,
        class_model,
        threshold,
        features[validation],
        dataset["binary"][validation],
        dataset["classes"][validation],
    )
    original_test_result = _evaluate(
        binary_model,
        class_model,
        threshold,
        features[original_test],
        dataset["binary"][original_test],
        dataset["classes"][original_test],
    )
    expanded_test_result = _evaluate(
        binary_model,
        class_model,
        threshold,
        features[expanded_test],
        dataset["binary"][expanded_test],
        dataset["classes"][expanded_test],
    )

    model_path = settings.MODEL_DIR / f"feature_system_{int(time.time())}.joblib"
    bundle = {
        "binary_model": binary_model,
        "class_model": class_model,
        "binary_threshold": threshold,
        "class_names": ARRHYTHMIA_CLASSES,
        "feature_version": "morphology-spectrum-rr-v2",
        "config": asdict(config),
        "records": {
            "train": TRAIN_RECORDS,
            "validation": VALIDATION_RECORDS,
            "original_test": ORIGINAL_TEST_RECORDS,
            "expanded_test": EXPANDED_TEST_RECORDS,
        },
    }
    joblib.dump(bundle, model_path, compress=3)
    compact = {
        "model_path": str(model_path),
        "config": asdict(config),
        "feature_count": int(features.shape[1]),
        "records": bundle["records"],
        "dataset_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "original_test": int(len(original_test)),
            "expanded_test": int(len(expanded_test)),
        },
        "binary_model_search": binary_search,
        "class_model_search": class_search,
        "selected_threshold": threshold,
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
    return {
        "summary": compact,
        "bundle": bundle,
        "validation_arrays": validation_result,
        "original_test_arrays": original_test_result,
        "expanded_test_arrays": expanded_test_result,
    }


def save_summary(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )


class ImprovedECGSystem:
    """Load the single-lead hierarchical feature system for beat inference.

    Both the abnormality detector and subtype classifier use the same MLII
    morphology, RR context, and optional frozen-autoencoder residual features.
    """

    def __init__(self, model_path: str | Path) -> None:
        bundle = joblib.load(model_path)
        self.binary_model = bundle["binary_model"]
        self.class_model = bundle["class_model"]
        self.binary_threshold = float(bundle["binary_threshold"])
        self.system_kind = str(bundle.get("system_kind", "hierarchical_feature_system"))
        self.class_names = list(bundle.get("class_names", ARRHYTHMIA_CLASSES))
        self.rr_feature_count = int(bundle.get("rr_feature_count", 7))
        self.requires_ae_features = bool(bundle.get("requires_ae_features", False))
        self.ae_feature_count = int(bundle.get("ae_feature_count", 0))
        self.reject_low_confidence = bool(bundle.get("reject_low_confidence", True))
        self.lead_count = int(bundle.get("lead_count", 1))
        self.required_lead = str(bundle.get("required_lead", "MLII"))
        if self.lead_count != 1:
            raise ValueError(
                f"This software supports the final one-lead system, but the "
                f"bundle declares {self.lead_count} leads"
            )
        if self.rr_feature_count not in (7, 15):
            raise ValueError(
                f"Unsupported RR feature count: {self.rr_feature_count}"
            )
        self.model_path = str(model_path)

    @staticmethod
    def _validate_window(window: np.ndarray) -> np.ndarray:
        array = np.asarray(window, dtype=np.float32)
        if array.shape != (512,):
            raise ValueError(f"Expected a 512-sample beat window, got {array.shape}")
        return array

    def _validate_rr(self, rr_features: np.ndarray) -> np.ndarray:
        array = np.asarray(rr_features, dtype=np.float32)
        if array.shape != (self.rr_feature_count,):
            raise ValueError(
                f"Expected {self.rr_feature_count} RR features, got {array.shape}"
            )
        return array

    def predict(
        self,
        mlii_window: np.ndarray,
        rr_features: np.ndarray,
        threshold: float | None = None,
        ae_features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        mlii_window = self._validate_window(mlii_window)
        rr_features = self._validate_rr(rr_features)
        first_features = build_features(
            mlii_window[None, :], rr_features[None, :]
        )
        if self.requires_ae_features:
            if ae_features is None:
                raise ValueError(
                    "This model requires autoencoder reconstruction features"
                )
            ae_array = np.asarray(ae_features, dtype=np.float32).reshape(-1)
            if ae_array.shape != (self.ae_feature_count,):
                raise ValueError(
                    f"Expected {self.ae_feature_count} autoencoder features, "
                    f"got {ae_array.shape}"
                )
            binary_features = np.concatenate(
                [first_features, ae_array[None, :]], axis=1
            )
        else:
            ae_array = np.empty(0, dtype=np.float32)
            binary_features = first_features
        abnormal_probability = float(
            self.binary_model.predict_proba(binary_features)[0, 1]
        )
        decision_threshold = (
            self.binary_threshold if threshold is None else float(threshold)
        )
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("The abnormal-probability threshold must be between 0 and 1")
        is_anomaly = abnormal_probability > decision_threshold

        class_features = binary_features
        class_model = self.class_model
        mode = "MLII-only"

        probabilities = {name: 0.0 for name in self.class_names}
        predicted_class = "N"
        confidence = 1.0 - abnormal_probability
        if is_anomaly and class_model is not None and class_features is not None:
            raw_probabilities = class_model.predict_proba(class_features)[0]
            aligned = np.zeros(len(self.class_names), dtype=np.float64)
            aligned[np.asarray(class_model.classes_, dtype=int)] = raw_probabilities
            abnormal_index = 1 + int(np.argmax(aligned[1:]))
            abnormal_class_probability = float(aligned[abnormal_index])
            if self.reject_low_confidence and (
                abnormal_class_probability < 0.50
                or float(aligned[0]) >= abnormal_class_probability
            ):
                predicted_class = "Unclassified abnormal"
                confidence = abnormal_probability
            else:
                predicted_class = self.class_names[abnormal_index]
                confidence = abnormal_class_probability
            probabilities = {
                name: float(aligned[index])
                for index, name in enumerate(self.class_names)
            }
        elif not is_anomaly:
            probabilities["N"] = float(1.0 - abnormal_probability)

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_probability": abnormal_probability,
            "threshold": decision_threshold,
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "classification_mode": mode,
            "lead": self.required_lead,
        }
