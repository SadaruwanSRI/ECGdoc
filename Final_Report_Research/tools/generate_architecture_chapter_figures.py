from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "pic" / "generated"

NAVY = "#183a59"
TEAL = "#0f766e"
ORANGE = "#c2410c"
PURPLE = "#6d28d9"
RED = "#b91c1c"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11.5, weight="bold", color=INK, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9ca3af")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)


def arrow(ax, xy1, xy2, color=INK, label: str | None = None, rad=0.0) -> None:
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
        color=color, connectionstyle=f"arc3,rad={rad}",
    ))
    if label:
        ax.text((xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2 + 0.03, label,
                ha="center", va="center", fontsize=7.6, color=color,
                bbox=dict(fc="white", ec="none", alpha=0.88, boxstyle="round,pad=0.16"))


def beat_wave(t: np.ndarray, wide: bool = False) -> np.ndarray:
    if wide:
        return (
            0.08 * np.exp(-((t - 0.20) / 0.040) ** 2)
            - 0.16 * np.exp(-((t - 0.42) / 0.035) ** 2)
            + 0.78 * np.exp(-((t - 0.49) / 0.055) ** 2)
            - 0.46 * np.exp(-((t - 0.58) / 0.050) ** 2)
            + 0.24 * np.exp(-((t - 0.76) / 0.090) ** 2)
        )
    return (
        0.12 * np.exp(-((t - 0.22) / 0.035) ** 2)
        - 0.20 * np.exp(-((t - 0.43) / 0.018) ** 2)
        + 1.05 * np.exp(-((t - 0.47) / 0.012) ** 2)
        - 0.33 * np.exp(-((t - 0.505) / 0.018) ** 2)
        + 0.30 * np.exp(-((t - 0.70) / 0.075) ** 2)
    )


def current_system_flow() -> None:
    fig = plt.figure(figsize=(11.6, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.12, 1.0], wspace=0.38)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.18)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.5, "Final ECG system: from one MLII beat to one decision",
                  fontsize=15, weight="bold", color=INK, va="center")

    t = np.linspace(0, 4.0, 512)
    normalised = np.linspace(0, 1, 512)
    x = beat_wave(normalised) + 0.018 * np.sin(2 * np.pi * 6 * normalised)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, x, color=NAVY, lw=2.0)
    r_time = 200 / 128
    ax1.axvline(r_time, color=RED, lw=2.0, linestyle="--")
    ax1.fill_between(t, x.min() - .1, x.max() + .1, where=(t <= r_time),
                     color=TEAL, alpha=0.08)
    ax1.fill_between(t, x.min() - .1, x.max() + .1, where=(t > r_time),
                     color=ORANGE, alpha=0.07)
    ax1.set_ylim(x.min() - 0.15, x.max() + 0.28)
    ax1.text(r_time, x.max() + 0.08, "R peak\nindex 200", ha="center",
             fontsize=8.5, color=RED, weight="bold")
    ax1.set_xlabel("4.0 s MLII window")
    ax1.set_yticks([])
    style_axis(ax1, "A. R-aligned\ninput")

    ax2 = fig.add_subplot(gs[1, 1])
    groups = [
        ("waveform", 223, NAVY),
        ("RR", 15, PURPLE),
        ("AE", 4, RED),
    ]
    left = 0
    for label, width, color in groups:
        ax2.barh([0], [width], left=[left], color=color, height=0.30)
        if label == "waveform":
            ax2.text(left + width / 2, 0, f"{label}\n{width}", ha="center",
                     va="center", color="white", fontsize=9, weight="bold")
        left += width
    ax2.annotate("RR 15", xy=(230.5, 0.15), xytext=(190, 0.31),
                 ha="center", va="center", fontsize=8, weight="bold", color=PURPLE,
                 arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.0))
    ax2.annotate("AE 4", xy=(240, 0.15), xytext=(239, 0.44),
                 ha="right", va="center", fontsize=8, weight="bold", color=RED,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))
    ax2.set_xlim(0, 242)
    ax2.set_ylim(-0.55, 0.55)
    ax2.set_xlabel("242 feature positions")
    ax2.set_yticks([])
    style_axis(ax2, "B. Reused 242-feature\nvector")
    ax2.text(121, -0.43, "shape + rhythm + normal-reconstruction evidence",
             ha="center", fontsize=8.5, color=INK)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axvspan(0, 0.6208333, color=TEAL, alpha=0.13)
    ax3.axvspan(0.6208333, 1, color=RED, alpha=0.11)
    ax3.axvline(0.6208333, color=RED, linestyle="--", lw=2.0)
    ax3.scatter([0.42, 0.78], [0.65, 0.65], s=95, color=[TEAL, RED], zorder=4)
    ax3.text(0.31, 0.78, "N", color=TEAL, fontsize=15, weight="bold")
    ax3.text(0.69, 0.78, "subtype", color=RED, fontsize=12, weight="bold")
    ax3.text(0.6208333, 0.12, "0.6208333", ha="center", fontsize=8.2,
             color=RED, weight="bold")
    ax3.set_xlabel(r"$P(\mathrm{abnormal})$")
    ax3.set_yticks([])
    style_axis(ax3, "C. Binary gate,\nthen subtype")
    fig.text(0.50, 0.045,
             "The autoencoder supplies four residual features. It does not make the final class decision alone.",
             ha="center", fontsize=9, color=INK)
    save(fig, "architecture_current_system_flow.pdf")


def feature_vector() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    ax.set_axis_off()
    ax.set_xlim(0, 242)
    ax.set_ylim(0, 1)

    groups = [
        ("whole-window\nmean bins", 64, NAVY),
        ("central\nmorphology", 80, TEAL),
        ("central\nderivative", 60, ORANGE),
        ("frequency\npowers", 8, "#7c2d12"),
        ("signal\nstatistics", 11, "#4338ca"),
        ("RR\ntiming", 15, PURPLE),
        ("AE\nresiduals", 4, RED),
    ]
    start = 0
    for label, width, color in groups:
        ax.add_patch(Rectangle((start, 0.43), width, 0.24, facecolor=color, edgecolor="white", lw=1.0))
        ax.text(start + width / 2, 0.55, str(width), ha="center", va="center",
                color="white", fontsize=10, weight="bold")
        ax.text(start + width / 2, 0.30, label, ha="center", va="top",
                fontsize=7.8, color=INK)
        start += width
    ax.annotate("", xy=(0, 0.77), xytext=(242, 0.77),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(121, 0.86, "242 values used by the binary detector and subtype classifier",
            ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(111.5, 0.10, "223 waveform, spectrum, and statistics values",
            ha="center", fontsize=9, color=NAVY, weight="bold")
    ax.text(232.5, 0.10, "timing + AE",
            ha="center", fontsize=9, color=PURPLE, weight="bold")
    save(fig, "architecture_feature_vector.pdf")


def training_live_map() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.94, "Training, testing, and installed use of the final bundle",
            fontsize=15, weight="bold", color=INK)

    y1, y2, y3 = 0.72, 0.47, 0.22
    ax.hlines([y1, y2, y3], 0.08, 0.93, colors=[TEAL, NAVY, RED], lw=3, alpha=0.75)
    ax.text(0.03, y1, "AE reference", va="center", fontsize=10, weight="bold", color=TEAL)
    ax.text(0.03, y2, "MITDB MLII", va="center", fontsize=10, weight="bold", color=NAVY)
    ax.text(0.03, y3, "installed use", va="center", fontsize=10, weight="bold", color=RED)

    markers = [
        (0.17, y1, "NSRDB ECG1\n18 normal subjects", TEAL),
        (0.43, y1, "frozen autoencoder\n4 residual features", TEAL),
        (0.17, y2, "first 64%\nfit candidates", NAVY),
        (0.40, y2, "64-80%\nvalidation", ORANGE),
        (0.57, y2, "16-beat gap", MUTED),
        (0.73, y2, "final 20%\ntemporal test", RED),
        (0.31, y3, "MLII session\nor replay", RED),
        (0.55, y3, "same 242\nfeatures", PURPLE),
        (0.78, y3, "frozen decision\nno retraining", RED),
    ]
    for x, y, text, color in markers:
        ax.scatter([x], [y], s=140, color=color, zorder=4, edgecolor="white", lw=1.5)
        ax.text(x, y + 0.075, text, ha="center", va="bottom", fontsize=8.4, color=INK)
    arrow(ax, (0.43, y1 - 0.02), (0.55, y3 + 0.03), TEAL, "AE residuals", rad=-0.18)
    arrow(ax, (0.40, y2 - 0.02), (0.55, y3 + 0.03), ORANGE, "selected models", rad=0.08)
    ax.text(0.50, 0.06,
            "The performance claim is known-patient temporal continuation: later MLII from patients whose earlier MLII was used for development.",
            ha="center", fontsize=9, color=INK)
    save(fig, "architecture_training_live_map.pdf")


def streaming_window() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.set_axis_off()
    ax.set_xlim(0, 512)
    ax.set_ylim(-1.2, 1.2)

    t = np.linspace(0, 1, 512)
    x = beat_wave(t)
    ax.plot(np.arange(512), x, color=NAVY, lw=2.0)
    ax.add_patch(Rectangle((0, -1.05), 200, 0.18, facecolor=TEAL, alpha=0.18, edgecolor=TEAL, lw=1.2))
    ax.add_patch(Rectangle((200, -1.05), 312, 0.18, facecolor=ORANGE, alpha=0.18, edgecolor=ORANGE, lw=1.2))
    ax.axvline(200, color=RED, lw=2.4, linestyle="--")
    ax.text(200, 1.03, "R peak index 200", ha="center", fontsize=10, color=RED, weight="bold")
    ax.text(100, -0.76, "previous 200 samples", ha="center", fontsize=8.8, color=TEAL, weight="bold")
    ax.text(356, -0.76, "following 312 samples", ha="center", fontsize=8.8, color=ORANGE, weight="bold")
    ax.annotate("", xy=(0, -0.48), xytext=(512, -0.48),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.6))
    ax.text(256, -0.37, "512 samples at 128 Hz = 4.0 seconds",
            ha="center", fontsize=11.5, weight="bold", color=NAVY)
    ax.text(256, -1.10,
            "Streaming classification waits until the beat has previous and next RR information, then scores this fixed R-aligned window.",
            ha="center", fontsize=8.8, color=INK)
    save(fig, "architecture_streaming_window.pdf")


def main() -> None:
    current_system_flow()
    feature_vector()
    training_live_map()
    streaming_window()


if __name__ == "__main__":
    main()
