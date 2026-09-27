"""Draw compact, code-aligned diagrams for the theory chapter."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "pic" / "generated"
NAVY = "#17324d"
TEAL = "#168c8c"
PALE = "#eaf4f4"
BLUE = "#eaf0f6"
ORANGE = "#fff1df"
INK = "#23323d"


def box(ax, x, y, w, h, title, body, fill=BLUE, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=1.2, edgecolor=NAVY, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * .72, title, ha="center", va="center",
            color=NAVY, fontsize=fontsize, weight="bold")
    ax.text(x + w / 2, y + h * .35, body, ha="center", va="center",
            color=INK, fontsize=fontsize - 1, linespacing=1.3)


def arrow(ax, start, end, colour=TEAL, curved=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.6, color=colour,
                                connectionstyle=f"arc3,rad={curved}"))


def learning_map():
    fig, ax = plt.subplots(figsize=(7.4, 5.35))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 5.35)
    ax.axis("off")
    ax.text(.2, 5.05, "How one ECG beat becomes a supported output",
            fontsize=14, weight="bold", color=NAVY)
    box(ax, .2, 3.85, 2.05, .82, "MLII waveform", "one channel, 128 Hz", fontsize=10)
    box(ax, 2.68, 3.85, 2.05, .82, "Prepared beat", "512 samples; filter + scale", PALE, 10)
    box(ax, 5.16, 3.85, 2.05, .82, "Shape + timing", "223 waveform + 15 RR", ORANGE, 10)
    arrow(ax, (2.28, 4.26), (2.65, 4.26))
    arrow(ax, (4.76, 4.26), (5.13, 4.26))
    box(ax, .2, 2.35, 2.05, .82, "Frozen autoencoder", "normal ECG reference", fontsize=10)
    box(ax, 2.68, 2.35, 2.05, .82, "Residual features", "four mismatch values", ORANGE, 10)
    box(ax, 5.16, 2.35, 2.05, .82, "Feature row", "242 values per beat", PALE, 10)
    arrow(ax, (3.7, 3.81), (1.27, 3.2), curved=.14)
    arrow(ax, (2.28, 2.76), (2.65, 2.76))
    arrow(ax, (4.76, 2.76), (5.13, 2.76))
    arrow(ax, (6.18, 3.82), (6.18, 3.2))
    box(ax, 1.23, .65, 2.2, .82, "Binary model", "240 trees: normal/abnormal", PALE, 10)
    box(ax, 4.05, .65, 2.2, .82, "Subtype model", "260 trees: five groups", PALE, 10)
    arrow(ax, (6.18, 2.31), (2.34, 1.51), curved=-.14)
    arrow(ax, (3.46, 1.06), (4.02, 1.06))
    ax.text(2.33, .38, "Normal branch gives N", ha="center", fontsize=9, color=NAVY)
    fig.tight_layout(pad=.25)
    fig.savefig(OUT / "theory_learning_map.pdf", bbox_inches="tight")
    plt.close(fig)


def training_mechanisms():
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    ax.text(.15, 4.75, "A. Autoencoder: repeated weight updates", fontsize=13,
            weight="bold", color=NAVY)
    xs = [.15, 2.0, 3.85, 5.7]
    labels_a = [
        ("Clean batch", "256 windows"),
        ("Add noise", "Gaussian SD 0.03"),
        ("Reconstruct", "encoder + decoder"),
        ("MSE + Adam", "gradient update"),
    ]
    for x, (title, body) in zip(xs, labels_a):
        box(ax, x, 3.38, 1.55, .9, title, body,
            ORANGE if title == "Add noise" else PALE, 9)
    for x in xs[:-1]:
        arrow(ax, (x + 1.58, 3.83), (x + 1.82, 3.83))
    ax.text(.2, 2.97, "Repeat for each batch; validate on held-out clean windows after each epoch.",
            fontsize=9, color=INK)
    ax.plot([.15, 7.25], [2.65, 2.65], color="#cbd5dc", linewidth=1)
    ax.text(.15, 2.3, "B. Extra Trees: recursive node splits", fontsize=13,
            weight="bold", color=NAVY)
    labels_b = [
        ("Labelled rows", "242 values + weight"),
        ("Random tests", "feature + threshold"),
        ("Entropy drop", "select best split"),
        ("Tree ensemble", "average leaf scores"),
    ]
    for x, (title, body) in zip(xs, labels_b):
        box(ax, x, .95, 1.55, .9, title, body,
            ORANGE if title == "Entropy drop" else PALE, 9)
    for x in xs[:-1]:
        arrow(ax, (x + 1.58, 1.4), (x + 1.82, 1.4))
    ax.text(.2, .53, "Repeat splitting within each tree; compare finished models on validation data.",
            fontsize=9, color=INK)
    fig.tight_layout(pad=.25)
    fig.savefig(OUT / "theory_training_mechanisms.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learning_map()
    training_mechanisms()
