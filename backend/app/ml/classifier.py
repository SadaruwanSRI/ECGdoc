"""Arrhythmia Classifier Model

A classification model that identifies the type of arrhythmia in an ECG window.
Uses transfer learning: reuses the encoder from the anomaly detection autoencoder
(frozen) and adds a classification head on top.

Architecture:
    ECG (1, 512)
        │
        ▼
    ┌──────────────────────┐
    │ ENCODER (FROZEN)     │  ← from Model 1 (autoencoder)
    │ Conv1d layers        │
    └──────────┬───────────┘
               │
               ▼
    Latent (1024, 16)
               │
               ▼
    ┌──────────────────────┐
    │ CLASSIFICATION HEAD  │  ← trained on MIT-BIH annotations
    │ Global Avg Pool      │  (1024,16) → (1024,)
    │ → Dense(128) + ReLU  │
    │ → Dropout(0.3)       │
    │ → Dense(6) + Softmax │
    └──────────┬───────────┘
               │
               ▼
    [N, PVC, PAC, LBBB, RBBB, AFib]
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model import ECGAutoencoder, ENCODER_CHANNELS, DECODER_CHANNELS


# The 6 arrhythmia classes we classify
ARRHYTHMIA_CLASSES = ["N", "PVC", "PAC", "LBBB", "RBBB", "AFib"]

# Human-readable names
ARRHYTHMIA_NAMES = {
    "N":    "Normal Sinus Rhythm",
    "PVC":  "Premature Ventricular Contraction",
    "PAC":  "Premature Atrial Contraction",
    "LBBB": "Left Bundle Branch Block",
    "RBBB": "Right Bundle Branch Block",
    "AFib": "Atrial Fibrillation",
}

# MIT-BIH Arrhythmia Database beat symbols used by this project.
#
# Supported classifier targets:
#   N/e/j -> N, L -> LBBB, R -> RBBB, A/a/J/S -> PAC, V/E -> PVC.
# AFib is a rhythm annotation in aux_note, not a beat symbol; normal-looking
# beats inside AFIB/AFL rhythm intervals are labeled AFib.
#
# Ambiguous or non-target abnormal beat symbols such as F, /, f, Q are kept as
# anomalies for binary detection evaluation but are skipped for classifier
# training so they do not contaminate the six supported class labels.
MITBIH_TO_CLASS = {
    "N": "N",
    "e": "N",
    "j": "N",
    "L": "LBBB",
    "R": "RBBB",
    "V": "PVC",
    "E": "PVC",
    "A": "PAC",
    "a": "PAC",
    "J": "PAC",
    "S": "PAC",
}

MITBIH_NORMAL_SYMBOLS = {"N", "e", "j"}
MITBIH_TARGET_SYMBOLS = set(MITBIH_TO_CLASS)
MITBIH_ABNORMAL_SYMBOLS = {
    "L", "R", "A", "a", "J", "S", "V", "E",
    "F", "/", "f", "Q", "r", "B", "n",
}
MITBIH_IGNORED_SYMBOLS = {
    "[", "]", "!", "x", "|", "~", "+", "s", "T", "*", "D", '"', "=",
}


def mitbih_symbol_to_class(symbol: str,
                           rhythm: Optional[str] = None,
                           include_unsupported: bool = False) -> Optional[str]:
    """Map a MIT-BIH annotation symbol/rhythm to one of our classifier classes.

    Returns None for non-beat markers and unsupported abnormal beat types unless
    include_unsupported=True, in which case unsupported abnormal beats are mapped
    to the closest available class only for display/evaluation fallbacks.
    """
    cls = MITBIH_TO_CLASS.get(symbol)
    if rhythm == "AFib" and cls == "N":
        return "AFib"
    if cls is not None:
        return cls
    if include_unsupported:
        if symbol == "F":
            return "PVC"
        if symbol in {"/", "f"}:
            return "N"
    return None


def mitbih_symbol_is_anomaly(symbol: str, rhythm: Optional[str] = None) -> Optional[bool]:
    """Return binary anomaly truth for MIT-BIH symbols.

    None means the annotation is not a beat label we should score.
    """
    if rhythm == "AFib" and symbol in MITBIH_NORMAL_SYMBOLS:
        return True
    if symbol in MITBIH_NORMAL_SYMBOLS:
        return False
    if symbol in MITBIH_ABNORMAL_SYMBOLS or symbol in MITBIH_TARGET_SYMBOLS:
        return True
    if symbol in MITBIH_IGNORED_SYMBOLS:
        return None
    return None


class RawMorphologyBranch(nn.Module):
    """Trainable beat-morphology branch directly from the ECG waveform."""

    def __init__(self, out_features: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """Classification head using encoder context plus optional raw morphology."""

    def __init__(self, latent_channels: int = 1024, num_classes: int = 6,
                 hidden_dim: int = 256, dropout: float = 0.35,
                 raw_features: int = 0) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        in_features = latent_channels * 2 + raw_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.in_features = in_features
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor,
                raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits for an ECG latent representation."""
        avg = self.avg_pool(x).squeeze(-1)
        maxv = self.max_pool(x).squeeze(-1)
        features = [avg, maxv]
        if raw_features is not None:
            features.append(raw_features)
        return self.net(torch.cat(features, dim=1))


class ECGClassifier(nn.Module):
    """Arrhythmia classifier using transfer learning from the autoencoder encoder.

    The encoder is loaded from a trained Model 1 and frozen (no gradient updates).
    Only the classification head is trained.

    Uses the full ECGAutoencoder (including U-Net skip connections and dilated
    bottleneck) as the feature extractor. We call encode() which returns
    (latent, skips) — we only use the latent for classification.
    """

    def __init__(self, encoder_state_dict: Dict[str, Any],
                 input_length: int = 512,
                 num_classes: int = 6,
                 hidden_dim: int = 256,
                 dropout: float = 0.35,
                 freeze_encoder: bool = False,
                 use_raw_branch: bool = True) -> None:
        super().__init__()

        # Build the full autoencoder (we only use the encoder part)
        skip_scale = float(encoder_state_dict.pop("_skip_scale", 1.0)) if "_skip_scale" in encoder_state_dict else 1.0
        self.autoencoder = ECGAutoencoder(
            input_channels=1,
            input_length=input_length,
            skip_scale=skip_scale,
        )
        self.autoencoder.load_state_dict(encoder_state_dict)

        for param in self.autoencoder.parameters():
            param.requires_grad = not freeze_encoder

        # Build the classification head
        latent_channels = ENCODER_CHANNELS[-1]  # 256 (new U-Net architecture)
        self.use_raw_branch = use_raw_branch
        self.raw_branch = RawMorphologyBranch(out_features=128, dropout=0.2) if use_raw_branch else None
        self.head = ClassificationHead(
            latent_channels=latent_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            raw_features=128 if use_raw_branch else 0,
        )

        self.num_classes = num_classes
        self.input_length = input_length
        self.freeze_encoder = freeze_encoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder to get the latent representation (ignoring skips)."""
        latent, _ = self.autoencoder.encode(x)
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ECG window → encoder (frozen) → classification head → logits.

        Args:
            x: ECG window (B, 1, 512)

        Returns:
            Logits (B, num_classes). Apply softmax to get probabilities.
        """
        grad_enabled = any(p.requires_grad for p in self.autoencoder.parameters())
        with torch.set_grad_enabled(grad_enabled and self.training):
            latent = self.encode(x)  # (B, 256, 32)
        raw_features = self.raw_branch(x) if self.raw_branch is not None else None
        logits = self.head(latent, raw_features=raw_features)   # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Predict the arrhythmia class for a single window.

        Returns:
            {
                "class": "PVC",
                "class_name": "Premature Ventricular Contraction",
                "confidence": 0.942,
                "probabilities": {"N": 0.012, "PVC": 0.942, ...},
            }
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.unsqueeze(0).unsqueeze(0))  # (1, 6)
            probs = F.softmax(logits, dim=1).squeeze()          # (6,)

        # Get top class
        top_idx = probs.argmax().item()
        top_class = ARRHYTHMIA_CLASSES[top_idx]
        confidence = probs[top_idx].item()

        probabilities = {
            ARRHYTHMIA_CLASSES[i]: probs[i].item()
            for i in range(self.num_classes)
        }

        return {
            "class": top_class,
            "class_name": ARRHYTHMIA_NAMES[top_class],
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def count_trainable_parameters(self) -> int:
        """Count only the trainable parameters (classification head)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Count all parameters (autoencoder + head)."""
        return sum(p.numel() for p in self.parameters())

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "name": "ECG-Classifier",
            "encoder": "Dilated-U-Net-Autoencoder (frozen)",
            "encoder_channels": ENCODER_CHANNELS,
            "latent_channels": ENCODER_CHANNELS[-1],
            "classification_head": {
                "pooling": "AdaptiveAvgPool1d + AdaptiveMaxPool1d",
                "input_features": self.head.in_features,
                "hidden_dim": self.head.hidden_dim,
                "normalization": "LayerNorm",
                "activation": "GELU",
                "dropout": "0.35",
                "raw_morphology_branch": self.use_raw_branch,
                "encoder_frozen": self.freeze_encoder,
                "output": "Softmax",
            },
            "num_classes": self.num_classes,
            "class_names": ARRHYTHMIA_CLASSES,
            "trainable_parameters": self.count_trainable_parameters(),
            "total_parameters": self.count_total_parameters(),
        }


def build_classifier(encoder_model_path: str,
                      input_length: int = 512,
                      hidden_dim: int = 256,
                      dropout: float = 0.35,
                      freeze_encoder: bool = False,
                      use_raw_branch: bool = True) -> ECGClassifier:
    """Factory: load an autoencoder's encoder and build a classifier on top.

    Args:
        encoder_model_path: Path to the .pt file from Model 1 (autoencoder).
        input_length: Input window length (default 512).

    Returns:
        ECGClassifier instance with frozen encoder + fresh classification head.
    """
    ckpt = torch.load(encoder_model_path, map_location="cpu", weights_only=False)
    encoder_state = dict(ckpt["state_dict"])
    encoder_state["_skip_scale"] = float(ckpt.get("config", {}).get("skip_scale", 1.0))
    classifier = ECGClassifier(
        encoder_state_dict=encoder_state,
        input_length=input_length,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        use_raw_branch=use_raw_branch,
    )
    return classifier


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "/home/z/my-project/backend")

    # Build a dummy encoder state dict
    model = ECGAutoencoder(input_length=512)
    classifier = ECGClassifier(model.state_dict(), input_length=512)

    x = torch.randn(1, 1, 512)
    logits = classifier(x)
    print(f"Input shape:    {x.shape}")
    print(f"Logits shape:   {logits.shape}")
    print(f"Trainable params: {classifier.count_trainable_parameters():,}")
    print(f"Total params:     {classifier.count_total_parameters():,}")

    result = classifier.predict(x.squeeze())
    print(f"Prediction: {result}")
