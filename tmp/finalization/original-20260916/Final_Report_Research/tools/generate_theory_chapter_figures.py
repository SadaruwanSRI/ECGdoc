from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "pic" / "generated"

NAVY = "#183a59"
TEAL = "#0f766e"
ORANGE = "#c2410c"
PURPLE = "#6d28d9"
RED = "#b91c1c"
GOLD = "#b45309"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, bbox_inches="tight")
    plt.close(fig)


def beat_wave(t: np.ndarray, shift: float = 0.0, wide: bool = False) -> np.ndarray:
    if wide:
        return (
            0.08 * np.exp(-((t - 0.20 - shift) / 0.040) ** 2)
            - 0.16 * np.exp(-((t - 0.42 - shift) / 0.035) ** 2)
            + 0.78 * np.exp(-((t - 0.49 - shift) / 0.055) ** 2)
            - 0.46 * np.exp(-((t - 0.58 - shift) / 0.050) ** 2)
            + 0.24 * np.exp(-((t - 0.76 - shift) / 0.090) ** 2)
        )
    return (
        0.12 * np.exp(-((t - 0.22 - shift) / 0.035) ** 2)
        - 0.20 * np.exp(-((t - 0.43 - shift) / 0.018) ** 2)
        + 1.05 * np.exp(-((t - 0.47 - shift) / 0.012) ** 2)
        - 0.33 * np.exp(-((t - 0.505 - shift) / 0.018) ** 2)
        + 0.30 * np.exp(-((t - 0.70 - shift) / 0.075) ** 2)
    )


def style_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, weight="bold", color=INK, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9ca3af")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)


def evidence_scales() -> None:
    fig = plt.figure(figsize=(11.4, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.12, 1.0], wspace=0.24)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.5, "Two ECG evidence scales used by the final model",
                  fontsize=15, weight="bold", color=INK, va="center")

    t = np.linspace(0, 1, 700)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, beat_wave(t), color=NAVY, lw=2.4)
    ax1.fill_between(t, -0.45, 1.10, where=(t >= 0.39) & (t <= 0.56),
                     color=TEAL, alpha=0.12, label="central QRS region")
    for x, y, label in [(0.22, 0.15, "P"), (0.43, -0.18, "Q"), (0.47, 1.05, "R"),
                        (0.505, -0.33, "S"), (0.70, 0.30, "T")]:
        ax1.scatter([x], [y], s=24, color=TEAL, zorder=4)
        ax1.text(x, y + (0.13 if y >= 0 else -0.15), label, ha="center",
                 va="center", fontsize=9, weight="bold", color=TEAL)
    ax1.annotate("shape, width, polarity and slopes",
                 xy=(0.48, 0.78), xytext=(0.09, 0.96),
                 arrowprops=dict(arrowstyle="-|>", lw=1.3, color=TEAL),
                 fontsize=8.5, color=TEAL)
    ax1.set_xlabel("time within one beat")
    ax1.set_ylabel("standardised voltage")
    ax1.set_yticks([])
    style_axis(ax1, "A. Beat-scale morphology")

    ax2 = fig.add_subplot(gs[1, 1])
    peaks = np.array([0.4, 1.2, 2.05, 2.61, 3.57, 4.38])
    heights = np.ones_like(peaks)
    ax2.vlines(peaks, 0, heights, color=ORANGE, lw=3)
    ax2.scatter(peaks, heights, color=ORANGE, s=42, zorder=4)
    for a, b in zip(peaks[:-1], peaks[1:]):
        ax2.annotate("", xy=(a, 0.28), xytext=(b, 0.28),
                     arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.2))
        ax2.text((a + b) / 2, 0.36, f"{b-a:.2f}s", ha="center",
                 fontsize=8, color=PURPLE)
    ax2.text(peaks[3], 1.13, "short interval", ha="center", fontsize=8.5,
             weight="bold", color=RED)
    ax2.annotate("", xy=(peaks[3], 1.03), xytext=(peaks[3], 1.20),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax2.set_ylim(-0.05, 1.35)
    ax2.set_xlabel("time across several beats")
    ax2.set_yticks([])
    style_axis(ax2, "B. Rhythm-scale RR timing")
    ax2.text(0.02, -0.22,
             "The final feature vector joins these scales with four normal-reconstruction residuals.",
             transform=ax2.transAxes, fontsize=8.7, color=INK)
    save(fig, "theory_evidence_scales.pdf")


def feature_math() -> None:
    fig = plt.figure(figsize=(11.4, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.12, 1, 1], hspace=0.55, wspace=0.26)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.5, "Transparent operations that create the final feature vector",
                  fontsize=15, weight="bold", color=INK, va="center")

    t = np.linspace(0, 1, 512)
    x = beat_wave(t) + 0.015 * np.sin(2 * np.pi * 16 * t)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, x, color=NAVY, lw=1.8)
    bins = np.linspace(0, 1, 17)
    means = [x[(t >= bins[i]) & (t < bins[i + 1])].mean() for i in range(len(bins) - 1)]
    centers = (bins[:-1] + bins[1:]) / 2
    ax1.step(centers, means, where="mid", color=TEAL, lw=2.2, label="pooled shape")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.set_xlabel(r"$z_k = |B_k|^{-1}\sum_{t\in B_k}x_t$")
    ax1.set_yticks([])
    style_axis(ax1, "A. Pooling keeps shape and reduces small shifts")

    ax2 = fig.add_subplot(gs[1, 1])
    dx = np.diff(x, prepend=x[0])
    ax2.plot(t, dx, color=ORANGE, lw=1.7)
    ax2.axhline(0, color="#9ca3af", lw=0.8)
    ax2.set_xlabel(r"$\Delta x_t=x_t-x_{t-1}$")
    ax2.set_yticks([])
    style_axis(ax2, "B. Derivatives describe QRS sharpness")

    ax3 = fig.add_subplot(gs[2, 0])
    freqs = np.linspace(0.5, 50, 160)
    power = (
        0.45 * np.exp(-((freqs - 5) / 3.0) ** 2)
        + 0.35 * np.exp(-((freqs - 12) / 4.5) ** 2)
        + 0.14 * np.exp(-((freqs - 30) / 7.0) ** 2)
    )
    bands = [(0.5, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 30), (30, 40), (40, 50)]
    colors = [NAVY, TEAL, ORANGE, PURPLE, GOLD, RED, "#475569", "#0891b2"]
    for (lo, hi), color in zip(bands, colors):
        mask = (freqs >= lo) & (freqs < hi)
        ax3.fill_between(freqs[mask], 0, power[mask], color=color, alpha=0.55)
    ax3.plot(freqs, power, color=INK, lw=1.1)
    ax3.set_xlabel(r"$\log(1+P_{band}/P_{0.5-50Hz})$")
    ax3.set_ylabel("relative power")
    ax3.set_yticks([])
    style_axis(ax3, "C. Spectral bands summarise frequency content")

    ax4 = fig.add_subplot(gs[2, 1])
    recon = 0.88 * beat_wave(t, shift=0.003)
    residual = x - recon
    ax4.plot(t, x, color=NAVY, lw=1.8, label="input")
    ax4.plot(t, recon, color=TEAL, lw=1.7, linestyle="--", label="reconstruction")
    ax4.fill_between(t, x, recon, color=RED, alpha=0.18, label="residual")
    ax4.legend(frameon=False, fontsize=8, loc="upper right")
    ax4.set_xlabel(r"$e_t=x_t-\hat{x}_t$")
    ax4.set_yticks([])
    style_axis(ax4, "D. Autoencoder residuals add normal-structure evidence")
    save(fig, "theory_feature_math.pdf")


def reconstruction_principle() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 5.6), sharex=True)
    fig.suptitle("Normal-only reconstruction: residual is evidence, not diagnosis",
                 fontsize=15, weight="bold", color=INK, x=0.02, ha="left")
    t = np.linspace(0, 1, 512)
    normal = beat_wave(t)
    normal_recon = 0.98 * beat_wave(t, shift=0.002)
    different = beat_wave(t, wide=True)
    different_recon = 0.78 * beat_wave(t, shift=0.005)

    pairs = [
        (axes[0, 0], normal, normal_recon, "A. Familiar normal morphology", TEAL),
        (axes[0, 1], np.abs(normal - normal_recon), None, "B. Small residual", TEAL),
        (axes[1, 0], different, different_recon, "C. Unfamiliar morphology", RED),
        (axes[1, 1], np.abs(different - different_recon), None, "D. Larger residual feature", RED),
    ]
    for ax, signal, recon, title, color in pairs:
        if recon is None:
            ax.plot(t, signal, color=color, lw=2.0)
            ax.fill_between(t, 0, signal, color=color, alpha=0.18)
        else:
            ax.plot(t, signal, color=NAVY, lw=1.9, label="input")
            ax.plot(t, recon, color=color, lw=1.8, linestyle="--", label="reconstruction")
            ax.fill_between(t, signal, recon, color=color, alpha=0.16)
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_yticks([])
        ax.set_xlabel("time")
        style_axis(ax, title)
    axes[1, 1].text(0.02, -0.32,
                    "The supervised detector receives four residual summaries and decides together with morphology and RR evidence.",
                    transform=axes[1, 1].transAxes, fontsize=8.6, color=INK)
    save(fig, "theory_reconstruction_principle.pdf")


def hierarchy_probability() -> None:
    fig = plt.figure(figsize=(11.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.12, 1.0], width_ratios=[1.25, 1.0], wspace=0.30)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.5, "Hierarchical probability logic in the final model",
                  fontsize=15, weight="bold", color=INK, va="center")

    ax1 = fig.add_subplot(gs[1, 0])
    probs = np.linspace(0, 1, 300)
    ax1.plot(probs, probs, color=NAVY, lw=2.0)
    ax1.axvspan(0, 0.6208333, color=TEAL, alpha=0.12)
    ax1.axvspan(0.6208333, 1, color=RED, alpha=0.10)
    ax1.axvline(0.6208333, color=RED, lw=2.2, linestyle="--")
    ax1.scatter([0.48, 0.79], [0.48, 0.79], s=85, color=[TEAL, RED], zorder=4)
    ax1.text(0.27, 0.82, "Output N", color=TEAL, fontsize=12, weight="bold")
    ax1.text(0.70, 0.25, "Run subtype model", color=RED, fontsize=12, weight="bold")
    ax1.text(0.6208333, 1.02, "threshold 0.6208333", ha="center",
             fontsize=8.5, color=RED, weight="bold")
    ax1.set_xlabel(r"binary probability $P(\mathrm{abnormal}\mid x)$")
    ax1.set_ylabel("gate response")
    ax1.set_yticks([])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.08)
    style_axis(ax1, "A. Binary abnormal gate")

    ax2 = fig.add_subplot(gs[1, 1])
    labels = ["PVC", "PAC", "LBBB", "RBBB", "AFib"]
    values = np.array([0.19, 0.08, 0.11, 0.54, 0.08])
    y = np.arange(len(labels))
    colors = [NAVY, TEAL, ORANGE, RED, PURPLE]
    ax2.barh(y, values, color=colors, alpha=0.86)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 0.65)
    for yi, value in zip(y, values):
        ax2.text(value + 0.015, yi, f"{value:.2f}", va="center", fontsize=8.5)
    ax2.text(0.52, 3, "max", color=RED, fontsize=10, weight="bold", va="center")
    ax2.set_xlabel(r"subtype probability $q_c$")
    style_axis(ax2, "B. Conditional subtype selection")
    save(fig, "theory_hierarchy_probability.pdf")


def validation_units() -> None:
    fig = plt.figure(figsize=(11.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.12, 1.0], wspace=0.28)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.5, "Why validation must respect subject clusters",
                  fontsize=15, weight="bold", color=INK, va="center")

    ax1 = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(7)
    centers = [(0.18, 0.72), (0.50, 0.72), (0.80, 0.72), (0.30, 0.32), (0.67, 0.30)]
    sizes = [38, 18, 30, 12, 42]
    for idx, ((cx, cy), size) in enumerate(zip(centers, sizes), start=1):
        pts = rng.normal([cx, cy], [0.045, 0.045], size=(size, 2))
        ax1.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.65, color=[NAVY, TEAL, ORANGE, PURPLE, RED][idx - 1])
        ax1.add_patch(Circle((cx, cy), 0.105, fill=False, lw=1.5, color=[NAVY, TEAL, ORANGE, PURPLE, RED][idx - 1]))
        ax1.text(cx, cy + 0.14, f"subject {idx}", ha="center", fontsize=8.3, weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("A. Many beats, but fewer independent subjects", loc="left",
                  fontsize=12, weight="bold", color=INK)
    ax1.spines[:].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 1])
    sample = rng.normal(0.965, 0.012, 10000)
    sample = np.clip(sample, 0.90, 1.0)
    ax2.hist(sample, bins=34, color=TEAL, alpha=0.75, edgecolor="white")
    lo, hi = np.percentile(sample, [2.5, 97.5])
    ax2.axvline(lo, color=RED, lw=2, linestyle="--")
    ax2.axvline(hi, color=RED, lw=2, linestyle="--")
    ax2.text(lo, ax2.get_ylim()[1] * 0.88, "2.5%", ha="right", color=RED, fontsize=8.5, weight="bold")
    ax2.text(hi, ax2.get_ylim()[1] * 0.88, "97.5%", ha="left", color=RED, fontsize=8.5, weight="bold")
    ax2.set_xlabel("subject-clustered accuracy")
    ax2.set_yticks([])
    style_axis(ax2, "B. Bootstrap resamples subjects, not individual beats")
    ax2.text(0.02, -0.20,
             "This supports a known-patient temporal-continuation claim, not an unseen-patient claim.",
             transform=ax2.transAxes, fontsize=8.7, color=INK)
    save(fig, "theory_validation_units.pdf")


def main() -> None:
    evidence_scales()
    feature_math()
    reconstruction_principle()
    hierarchy_probability()
    validation_units()


if __name__ == "__main__":
    main()
