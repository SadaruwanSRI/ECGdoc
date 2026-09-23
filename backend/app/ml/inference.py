"""Inference engine - load saved ECG models and score ECG windows.

The final research pathway uses a beat-aligned MLII hierarchy. A frozen
normal-only autoencoder supplies four residual features, while supervised
Extra Trees models make the abnormality and subtype decisions. The older
sliding reconstruction mode remains available as a fallback for legacy
autoencoder-only models.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model import build_model
from app.ml.data import preprocess_signal


class InferenceEngine:
    """Load once, score many. Thread-safe enough for single-worker FastAPI.

    Supports the final classifier bundle as an optional second artifact:
    1. The autoencoder reconstructs the beat to supply residual features.
    2. The MLII hierarchy combines morphology, RR timing, and residuals.
    3. Legacy autoencoder-only detection is used only when no hierarchy exists.
    """

    def __init__(
        self,
        model_path: str,
        threshold: Optional[float] = None,
        classifier_path: Optional[str] = None,
        improved_threshold: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load autoencoder (Model 1)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        skip_scale = float(cfg.get("skip_scale", 1.0))
        self.model = build_model(input_length=settings.WINDOW_SAMPLES,
                                 skip_scale=skip_scale).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.threshold = float(threshold if threshold is not None
                               else ckpt.get("threshold", 0.05))
        self.threshold_k = float(ckpt.get("threshold_k", settings.DEFAULT_THRESHOLD_K))
        self.architecture = ckpt.get("architecture", {})
        self.val_loss = float(ckpt.get("val_loss", float("nan")))

        # Load classifier (Model 2) — optional
        self.classifier = None
        self.improved_system = None
        self.improved_threshold = improved_threshold
        self.classifier_architecture = None
        if classifier_path:
            try:
                if str(classifier_path).lower().endswith(".joblib"):
                    from app.ml.feature_system import ImprovedECGSystem

                    self.improved_system = ImprovedECGSystem(classifier_path)
                    expected_hash = self.improved_system.autoencoder_sha256
                    if expected_hash and hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != expected_hash:
                        raise ValueError("The classifier requires its paired frozen autoencoder checkpoint")
                    self.classifier_architecture = {
                        "name": self.improved_system.system_kind,
                        "lead_count": self.improved_system.lead_count,
                        "binary_threshold": self.improved_system.binary_threshold,
                        "active_binary_threshold": self.decision_threshold,
                        "requires_rr_context": True,
                        "requires_autoencoder_features": self.improved_system.requires_ae_features,
                    }
                    self.class_names = self.improved_system.class_names
                    return
                from app.ml.classifier import ECGClassifier, ARRHYTHMIA_CLASSES
                clf_ckpt = torch.load(classifier_path, map_location=self.device,
                                       weights_only=False)
                clf_arch = clf_ckpt.get("architecture", {})
                head_arch = clf_arch.get("classification_head", {})
                encoder_state = dict(ckpt["state_dict"])
                encoder_state["_skip_scale"] = skip_scale
                self.classifier = ECGClassifier(
                    encoder_state_dict=encoder_state,
                    input_length=settings.WINDOW_SAMPLES,
                    hidden_dim=int(head_arch.get("hidden_dim", 256)),
                    freeze_encoder=bool(head_arch.get("encoder_frozen", False)),
                    use_raw_branch=bool(head_arch.get("raw_morphology_branch", True)),
                ).to(self.device)
                self.classifier.load_state_dict(clf_ckpt["state_dict"])
                self.classifier.eval()
                self.classifier_architecture = clf_arch
                self.class_names = ARRHYTHMIA_CLASSES
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load requested classifier {classifier_path}: {exc}"
                ) from exc

    @torch.no_grad()
    def score_window(self, window: np.ndarray) -> Tuple[float, np.ndarray, bool]:
        """Score a single 512-sample window (already preprocessed).

        Returns (anomaly_score, reconstruction, is_anomaly).
        """
        if window.shape != (settings.WINDOW_SAMPLES,):
            raise ValueError(
                f"Window shape {window.shape} != ({settings.WINDOW_SAMPLES},)"
            )
        x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(self.device)
        x_hat = self.model(x)
        mae = (x - x_hat).abs().mean().item()
        is_anomaly = mae > self.threshold
        return mae, x_hat.squeeze().cpu().numpy(), is_anomaly

    @torch.no_grad()
    def score_stream_chunk(self, chunk: np.ndarray,
                           src_fs: int = settings.SAMPLING_RATE_HZ,
                           t_offset: int = 0,
                           already_preprocessed: bool = False,
                           ) -> List[dict]:
        """Score a chunk of arbitrary length by sliding a 512-sample window.

        Returns a list of {t, value, prediction, anomaly_score, is_anomaly} dicts,
        one per sample (with reconstruction / score computed per-window and
        broadcast across the window).

        Parameters
        ----------
        chunk : np.ndarray
            1-D ECG signal chunk (any length).
        src_fs : int
            Sampling frequency of the input chunk in Hz.
        t_offset : int
            Sample index to start the t values at. Pass the cumulative sample
            count from previous chunks so t increases monotonically across
            chunks (otherwise the chart overlaps on itself).

        Also stores the preprocessed chunk in `self._last_chunk` so the caller
        can extract a 512-sample window centered on any sample index (used by
        the alert renderer to capture the abnormal waveform).
        """
        # Preprocess the chunk (resample → bandpass → z-score)
        chunk = (
            np.asarray(chunk, dtype=np.float32).reshape(-1)
            if already_preprocessed
            else preprocess_signal(chunk, src_fs)
        )
        original_n = len(chunk)
        n = original_n
        win = settings.WINDOW_SAMPLES
        if n < win:
            # Pad with zeros if too short
            chunk = np.pad(chunk, (0, win - n))
            n = win

        # Stash for window extraction (used by extract_window_at)
        self._last_chunk = chunk
        self._last_win_size = win

        per_sample_pred = np.zeros(n, dtype=np.float32)
        per_sample_score = np.zeros(n, dtype=np.float32)
        per_sample_anomaly = np.zeros(n, dtype=bool)

        # Slide with stride = 64 (~0.5 s at 128 Hz) for smooth updates
        stride = 64
        window_starts = list(range(0, n - win + 1, stride))
        final_start = n - win
        if not window_starts or window_starts[-1] != final_start:
            window_starts.append(final_start)
        for i in window_starts:
            window = chunk[i:i + win]
            x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(self.device)
            x_hat = self.model(x)
            mae = (x - x_hat).abs().mean().item()
            is_anomaly = mae > self.threshold

            # Broadcast window-level outputs across the window's samples
            pred = x_hat.squeeze().cpu().numpy()
            per_sample_pred[i:i + win] = pred
            per_sample_score[i:i + win] = mae
            per_sample_anomaly[i:i + win] = is_anomaly

        out = []
        for i in range(original_n):
            out.append({
                "t": i + t_offset,   # monotonically increasing across chunks
                "value": float(chunk[i]),
                "prediction": float(per_sample_pred[i]),
                "anomaly_score": float(per_sample_score[i]),
                "is_anomaly": bool(per_sample_anomaly[i]),
            })
        return out

    def extract_window_at(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (input, reconstruction) for the 512-sample window centered at index t.

        Uses the chunk stored from the last `score_stream_chunk` call.
        If t is near the start/end of the chunk, the window is aligned to the edge.
        """
        chunk = getattr(self, '_last_chunk', None)
        win = getattr(self, '_last_win_size', settings.WINDOW_SAMPLES)
        if chunk is None:
            raise RuntimeError("extract_window_at called before score_stream_chunk")
        n = len(chunk)
        # Center the window on t, but clamp to valid range
        start = max(0, min(t - win // 2, n - win))
        window = chunk[start:start + win]
        if len(window) < win:
            window = np.pad(window, (0, win - len(window)))
        recon = self.reconstruct_window(window)
        return window, recon

    @torch.no_grad()
    def reconstruct_window(self, window: np.ndarray) -> np.ndarray:
        """Run the model on a single 512-sample window and return the reconstruction.

        Used by the alert renderer to capture the actual ECG segment + its
        reconstruction at the moment an anomaly is detected, so the UI and
        PDF report can display the abnormal waveform.
        """
        if window.shape != (settings.WINDOW_SAMPLES,):
            raise ValueError(f"Window shape {window.shape} != ({settings.WINDOW_SAMPLES},)")
        x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(self.device)
        x_hat = self.model(x)
        return x_hat.squeeze().cpu().numpy()

    def info(self) -> dict:
        return {
            "model_path": self.model_path,
            "threshold": self.threshold,
            "threshold_k": self.threshold_k,
            "architecture": self.architecture,
            "val_loss": self.val_loss,
            "device": str(self.device),
            "has_classifier": (
                self.classifier is not None or self.improved_system is not None
            ),
            "has_improved_system": self.improved_system is not None,
            "decision_threshold": self.decision_threshold,
            "analysis_mode": (
                "beat-aligned-hierarchical"
                if self.improved_system is not None
                else "sliding-reconstruction"
            ),
            "classifier_architecture": self.classifier_architecture,
        }

    @property
    def decision_threshold(self) -> float:
        """Return the cut-off used by the active anomaly decision stage."""
        if self.improved_system is not None:
            if self.improved_threshold is not None:
                return float(self.improved_threshold)
            return float(self.improved_system.binary_threshold)
        return float(self.threshold)

    def classify_beat(
        self,
        window: np.ndarray,
        rr_features: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Classify one R-peak-aligned lead with the improved hierarchy.

        The first seven RR values are previous RR, next RR, local RR mean,
        previous/local, next/local, local coefficient of variation, and heart
        rate. Newer bundles append eight causal rhythm-history values. A
        separate method is necessary because an arbitrary sliding window does
        not contain explicit R-peak interval context.
        """
        if self.improved_system is None:
            return None
        ae_features = None
        if self.improved_system.requires_ae_features:
            x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                reconstruction = self.model(x)
                residual = x - reconstruction
                absolute = residual.abs()
                ae_features = np.asarray(
                    [
                        float(absolute.mean().item()),
                        float(residual.square().mean().item()),
                        float(absolute[:, :, 104:344].mean().item()),
                        float(torch.diff(residual, dim=2).abs().mean().item()),
                    ],
                    dtype=np.float32,
                )
        return self.improved_system.predict(
            window,
            rr_features,
            threshold=self.decision_threshold,
            ae_features=ae_features,
        )

    @torch.no_grad()
    def classify_window(self, window: np.ndarray) -> Optional[Dict[str, Any]]:
        """Classify the arrhythmia type of a 512-sample window.

        Returns None if no classifier is loaded.
        Returns {
            "class": "PVC",
            "class_name": "Premature Ventricular Contraction",
            "confidence": 0.942,
            "probabilities": {"N": 0.012, "PVC": 0.942, ...},
        } if classifier is available.
        """
        if self.classifier is None:
            return None

        from app.ml.classifier import ARRHYTHMIA_CLASSES, ARRHYTHMIA_NAMES

        x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1).squeeze()

        top_idx = probs.argmax().item()
        top_class = ARRHYTHMIA_CLASSES[top_idx]
        confidence = probs[top_idx].item()

        probabilities = {
            ARRHYTHMIA_CLASSES[i]: probs[i].item()
            for i in range(len(ARRHYTHMIA_CLASSES))
        }

        return {
            "class": top_class,
            "class_name": ARRHYTHMIA_NAMES[top_class],
            "confidence": confidence,
            "probabilities": probabilities,
        }


# Singleton cache: model_id → engine
_ENGINES: dict[str, InferenceEngine] = {}


def get_engine(
    model_path: str,
    threshold: Optional[float] = None,
    classifier_path: Optional[str] = None,
    improved_threshold: Optional[float] = None,
) -> InferenceEngine:
    key = f"{model_path}:{threshold}:{classifier_path}:{improved_threshold}"
    if key not in _ENGINES:
        _ENGINES[key] = InferenceEngine(
            model_path,
            threshold=threshold,
            classifier_path=classifier_path,
            improved_threshold=improved_threshold,
        )
    return _ENGINES[key]
