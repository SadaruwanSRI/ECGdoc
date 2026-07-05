"""Inference engine — load a saved model and score ECG windows in real time.

Supports two-step inference:
  Step 1: Autoencoder → anomaly detection (reconstruction error > threshold)
  Step 2: Classifier → arrhythmia type identification (if anomaly detected)
"""
from __future__ import annotations

from pathlib import Path
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

    Supports an optional classifier model for two-step inference:
    1. Autoencoder detects anomaly (reconstruction error > threshold)
    2. Classifier identifies arrhythmia type (only if anomaly detected)
    """

    def __init__(self, model_path: str, threshold: Optional[float] = None,
                 classifier_path: Optional[str] = None) -> None:
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
        self.classifier_architecture = None
        if classifier_path:
            try:
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
            except Exception as e:
                print(f"[inference] Failed to load classifier: {e}")
                self.classifier = None

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
        chunk = preprocess_signal(chunk, src_fs)
        n = len(chunk)
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
        for i in range(0, n - win + 1, stride):
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
        for i in range(n):
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
            "has_classifier": self.classifier is not None,
            "classifier_architecture": self.classifier_architecture,
        }

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


def get_engine(model_path: str, threshold: Optional[float] = None,
               classifier_path: Optional[str] = None) -> InferenceEngine:
    key = f"{model_path}:{threshold}:{classifier_path}"
    if key not in _ENGINES:
        _ENGINES[key] = InferenceEngine(model_path, threshold=threshold,
                                         classifier_path=classifier_path)
    return _ENGINES[key]
