"""Generate Chapter 3 figures from the locally stored PhysioNet ECG data."""

from pathlib import Path
from math import gcd

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


REPORT = Path(__file__).resolve().parents[1]
DATA = REPORT.parent / "backend" / "app" / "storage" / "datasets"
OUTPUT = REPORT / "pic" / "generated"
FS = 128
WINDOW = 512
R_BEFORE = 200

NAVY = "#18324b"
TEAL = "#078f92"
ORANGE = "#e07a1f"
PURPLE = "#6f42c1"
GREY = "#657384"
LIGHT = "#eef4f6"


def resample(signal: np.ndarray, source_rate: int) -> np.ndarray:
    common = gcd(int(source_rate), FS)
    return resample_poly(signal, FS // common, int(source_rate) // common).astype(np.float32)


def bandpass(signal: np.ndarray) -> np.ndarray:
    sos = butter(4, [0.5, 50.0], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def standardise(signal: np.ndarray) -> np.ndarray:
    return ((signal - signal.mean()) / (signal.std() + 1e-8)).astype(np.float32)


def style_axis(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel, color=NAVY, fontsize=8)
    axis.grid(True, alpha=0.18, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7, colors=GREY)


def cleaning_figure() -> None:
    import wfdb

    # Healthy example: a real piece from the first stored NSRDB channel.
    healthy = wfdb.rdrecord(str(DATA / "nsrdb" / "16265"))
    healthy_fs = int(healthy.fs)
    healthy_start = 60 * 60 * healthy_fs
    healthy_raw = healthy.p_signal[
        healthy_start:healthy_start + 8 * healthy_fs, 0
    ].astype(np.float32)
    healthy_filtered = bandpass(resample(healthy_raw, healthy_fs))
    healthy_window = standardise(healthy_filtered[2 * FS:2 * FS + WINDOW])

    # Arrhythmia-system example: a real annotated MLII beat from MITDB record 100.
    abnormal = wfdb.rdrecord(str(DATA / "mitdb" / "100"))
    annotation = wfdb.rdann(str(DATA / "mitdb" / "100"), "atr")
    abnormal_fs = int(abnormal.fs)
    eligible = annotation.sample[(annotation.sample > 60 * abnormal_fs) &
                                 (annotation.sample < 120 * abnormal_fs)]
    raw_peak = int(eligible[len(eligible) // 2])
    abnormal_raw = abnormal.p_signal[
        raw_peak - 4 * abnormal_fs:raw_peak + 4 * abnormal_fs, 0
    ].astype(np.float32)
    abnormal_filtered = bandpass(resample(abnormal_raw, abnormal_fs))
    peak = 4 * FS
    abnormal_window = standardise(
        abnormal_filtered[peak - R_BEFORE:peak - R_BEFORE + WINDOW]
    )

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 6.2), sharex="row",
                             constrained_layout=True)
    fig.suptitle("Real ECG cleaning for the two one-channel model inputs",
                 fontsize=13, fontweight="bold", color=NAVY)
    columns = [
        ("A. Autoencoder: NSRDB record 16265, first channel (ECG1)",
         healthy_raw, healthy_fs, healthy_filtered, healthy_window, False),
        ("B. Detector/classifier: MITDB record 100, MLII",
         abnormal_raw, abnormal_fs, abnormal_filtered, abnormal_window, True),
    ]
    for col, (title, raw, source_rate, filtered, window, aligned) in enumerate(columns):
        axes[0, col].plot(np.arange(raw.size) / source_rate, raw,
                          color=GREY, linewidth=0.75)
        axes[0, col].set_title(title + "\n1. Raw source signal",
                               fontsize=8.2, fontweight="bold", color=NAVY)
        style_axis(axes[0, col], "mV")
        axes[0, col].set_xlabel("source time (s)", fontsize=7, color=NAVY)

        axes[1, col].plot(np.arange(filtered.size) / FS, filtered,
                          color=TEAL, linewidth=0.8)
        axes[1, col].set_title("2. Resample to 128 Hz + 0.5--50 Hz band-pass",
                               fontsize=8.2, fontweight="bold", color=NAVY)
        style_axis(axes[1, col], "mV")
        axes[1, col].set_xlabel("time after resampling (s)", fontsize=7, color=NAVY)

        axes[2, col].plot(np.arange(WINDOW) / FS, window,
                          color=PURPLE if col == 0 else "#d62728", linewidth=0.85)
        axes[2, col].axhline(0, color=GREY, linewidth=0.5)
        if aligned:
            axes[2, col].axvline(R_BEFORE / FS, color=ORANGE, linestyle="--",
                                 linewidth=1.0, label="R peak at sample 200")
            axes[2, col].legend(loc="upper right", fontsize=6.5, frameon=False)
            end_text = "3. Standardise each R-aligned 4 s beat window"
        else:
            end_text = "3. Standardise each non-overlapping 4 s window"
        axes[2, col].set_title(end_text, fontsize=8.2, fontweight="bold", color=NAVY)
        style_axis(axes[2, col], "z score")
        axes[2, col].set_xlabel("model-window time (s)", fontsize=7, color=NAVY)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "real_data_cleaning_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def representative_index(windows: np.ndarray, indices: np.ndarray) -> int:
    subset = windows[indices]
    mean = subset.mean(axis=0)
    distance = np.mean((subset - mean) ** 2, axis=1)
    return int(indices[int(np.argmin(distance))])


def dataset_overview_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Dataset roles and evaluation boundaries",
                 fontsize=13, fontweight="bold", color=NAVY)

    def box(x, y, w, h, title, text, edge, face):
        patch = plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.1, edgecolor=edge, facecolor=face)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.035, title, ha="left", va="top",
                fontsize=8.4, fontweight="bold", color=edge)
        ax.text(x + 0.02, y + h - 0.105, text, ha="left", va="top",
                fontsize=7.0, color=NAVY, linespacing=1.18)

    def arrow(x1, y1, x2, y2, label=None, colour=GREY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    linewidth=1.25, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.025, label,
                    ha="center", va="bottom", fontsize=6.8, color=colour)

    box(0.04, 0.58, 0.30, 0.28, "NSRDB healthy source",
        "18 long-duration records\nfirst stored ECG channel only\n15 subjects fit AE\n3 subjects validate AE",
        TEAL, "#eaf7f6")
    box(0.04, 0.16, 0.30, 0.28, "MITDB labeled source",
        "46 MLII-containing records\nrecords 102/104 excluded\n45 subject clusters\nexpert beat/rhythm labels",
        PURPLE, "#f2edfb")

    box(0.42, 0.60, 0.23, 0.24, "Frozen normal model",
        "Denoising U-Net autoencoder\n512-sample windows\nfour residual features",
        TEAL, "#f2fbfa")
    box(0.42, 0.20, 0.23, 0.24, "MLII feature table",
        "R-aligned MLII windows\nmorphology + spectrum\nRR context + AE residuals",
        PURPLE, "#f8f5fc")
    box(0.72, 0.57, 0.23, 0.28, "Development",
        "first 64%: fit candidates\nnext 16%: choose models\n16-beat gap before test",
        ORANGE, "#fff4e8")
    box(0.72, 0.16, 0.23, 0.28, "Temporal test",
        "final 20% of each record\n20,988 binary beats\n20,062 supported six-class beats",
        NAVY, "#eef4f6")

    arrow(0.34, 0.72, 0.42, 0.72, "healthy only", TEAL)
    arrow(0.34, 0.30, 0.42, 0.30, "MLII labels", PURPLE)
    arrow(0.54, 0.60, 0.54, 0.44, "residuals", TEAL)
    arrow(0.65, 0.32, 0.72, 0.70, "80%", ORANGE)
    arrow(0.65, 0.30, 0.72, 0.30, "20%", NAVY)

    ax.text(0.50, 0.055,
            "The temporal test uses later signal from the same MITDB patients; it is not an unseen-patient test.",
            ha="center", va="bottom", fontsize=7.1, color=GREY)
    fig.savefig(OUTPUT / "dataset_role_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def lead_position_figure() -> None:
    chest = plt.imread(REPORT / "pic" / "source" / "mlii_chest_electrodes.png")
    fig = plt.figure(figsize=(8.2, 4.9))
    body = fig.add_axes([0.055, 0.11, 0.43, 0.78])
    lead_info = fig.add_axes([0.54, 0.11, 0.41, 0.78])
    fig.suptitle("Lead positions and one-channel rule used by the model",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.975)

    body.imshow(chest, extent=(0, 1, 0, 1), aspect="auto")
    body.axis("off")
    electrode_points = {
        "RA (-)\nright upper chest": (0.32, 0.795, 0.03, 0.88),
        "LA (+)\nleft lower chest/rib": (0.77, 0.38, 0.50, 0.50),
        "RL reference\nlower right torso": (0.22, 0.18, 0.03, 0.28),
    }
    for label, (px, py, tx, ty) in electrode_points.items():
        body.annotate(
            label,
            xy=(px, py),
            xytext=(tx, ty),
            xycoords=body.transAxes,
            textcoords=body.transAxes,
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=NAVY,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor=TEAL, alpha=0.90),
            arrowprops=dict(arrowstyle="->", color=TEAL, linewidth=1.3),
        )
    body.annotate(
        "MLII signal direction",
        xy=(0.75, 0.40),
        xytext=(0.38, 0.64),
        xycoords=body.transAxes,
        textcoords=body.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        color=PURPLE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                  edgecolor=PURPLE, alpha=0.88),
        arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=1.8),
    )
    body.set_title("A. Realistic MLII torso electrode placement", loc="left",
                   fontsize=9.5, fontweight="bold", color=NAVY)
    lead_info.axis("off")
    lead_info.set_title("B. What each model actually receives", loc="left",
                        fontsize=9.5, fontweight="bold", color=NAVY)
    boxes = [
        (0.02, 0.88, "NORMAL AUTOENCODER",
         "18 NSRDB subjects\nfirst stored channel only\nWFDB name: ECG1\nplacement not documented", TEAL, "#eaf7f6"),
        (0.02, 0.57, "ABNORMALITY DETECTOR",
         "MITDB MLII only\nMLII selected by signal name\nrecord 114: MLII is channel 2\nrecords 102/104 excluded", PURPLE, "#f2edfb"),
        (0.02, 0.27, "SUBTYPE CLASSIFIER",
         "same MLII window\nsame 242-value feature vector\nsame four autoencoder residuals\nno second ECG lead is used", ORANGE, "#fff4e8"),
    ]
    for x, y, title, text, edge, face in boxes:
        lead_info.text(
            x, y,
            f"{title}\n{text}",
            transform=lead_info.transAxes,
            fontsize=7.35,
            color=NAVY,
            va="top",
            linespacing=1.18,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=face,
                      edgecolor=edge, linewidth=1.05),
        )
    lead_info.text(
        0.02, 0.035,
        "Therefore the performance numbers describe MLII temporal continuation, not a two-lead or unseen-patient result.",
        transform=lead_info.transAxes,
        fontsize=7.0,
        color=GREY,
        va="bottom",
    )

    fig.savefig(OUTPUT / "real_lead_positions.pdf", bbox_inches="tight")
    plt.close(fig)


def preparation_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Data preparation during training and during unseen use",
                 y=0.985, fontsize=13, fontweight="bold", color=NAVY)

    def panel(x, y, w, h, title, colour):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#fbfdff",
                                   edgecolor=colour, linewidth=1.2))
        ax.text(x + 0.015, y + h - 0.025, title, ha="left", va="top",
                fontsize=9.3, fontweight="bold", color=colour)

    def box(x, y, w, h, text, edge=TEAL, face=LIGHT, fs=7.1):
        patch = plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.008",
            linewidth=1.0, edgecolor=edge, facecolor=face)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=NAVY, linespacing=1.15)

    def arrow(x1, y1, x2, y2, colour=GREY, label=None, dy=0.012):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    linewidth=1.15, shrinkA=1, shrinkB=1))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label,
                    ha="center", va="bottom", fontsize=6.2, color=GREY)

    # A. Model-development path.
    panel(0.015, 0.505, 0.97, 0.425,
          "A. MODEL TRAINING - labels are available and parameters are updated", TEAL)
    box(0.035, 0.755, 0.15, 0.105,
        "18 NSRDB subjects\ncomplete first channel\nnormal ECG", TEAL, "#e9f8f7", fs=6.7)
    box(0.225, 0.755, 0.16, 0.105,
        "128 Hz\nband-pass filter\nstandardise", TEAL, "#f2fbfa", fs=6.7)
    box(0.425, 0.755, 0.15, 0.105,
        "345,850 clean\n4 s windows\n[B, 1, 512]", TEAL, "#f2fbfa", fs=6.7)
    box(0.615, 0.755, 0.15, 0.105,
        "15 fit subjects\n3 validation subjects\nnoisy -> clean", TEAL, "#f2fbfa", fs=6.5)
    box(0.805, 0.755, 0.15, 0.105,
        "Saved, frozen\nautoencoder", TEAL, "#dff4f2", fs=7.4)
    for x1, x2 in [(0.185, 0.225), (0.385, 0.425), (0.575, 0.615), (0.765, 0.805)]:
        arrow(x1, 0.807, x2, 0.807, TEAL)

    box(0.035, 0.555, 0.15, 0.125,
        "46 MITDB records\nfirst 80% only\nMLII selected by name\nexpert labels", PURPLE, "#f3edfb")
    box(0.225, 0.555, 0.16, 0.125,
        "MLII waveform:\nresample, filter,\nstandardise and\nR-peak align", PURPLE, "#f8f5fc")
    box(0.425, 0.555, 0.15, 0.125,
        "512 MLII samples\n+ RR history\n+ frozen-AE\nresidual values", PURPLE, "#f8f5fc")
    box(0.605, 0.625, 0.18, 0.075,
        "MLII -> 242 values\nnormal / abnormal label", PURPLE, "#f3edfb", fs=6.4)
    box(0.815, 0.625, 0.14, 0.075,
        "Train/validate\nabnormality detector", PURPLE, "#e9def7", fs=6.4)
    box(0.605, 0.525, 0.18, 0.075,
        "MLII: same 242 values\n5-class abnormal label", ORANGE, "#fff4e8", fs=6.1)
    box(0.815, 0.525, 0.14, 0.075,
        "Train/validate\nsubtype classifier", ORANGE, "#ffead2", fs=6.4)
    arrow(0.185, 0.617, 0.225, 0.617, PURPLE)
    arrow(0.385, 0.617, 0.425, 0.617, PURPLE)
    arrow(0.575, 0.617, 0.605, 0.662, PURPLE)
    arrow(0.785, 0.662, 0.815, 0.662, PURPLE)
    arrow(0.575, 0.617, 0.605, 0.562, ORANGE)
    arrow(0.785, 0.562, 0.815, 0.562, ORANGE)

    # B. Finished-system path for a fixed temporal-test/new signal.
    panel(0.015, 0.055, 0.97, 0.405,
          "B. UNSEEN USE - frozen parameters; no expert label enters prediction", ORANGE)
    box(0.035, 0.285, 0.15, 0.105,
        "Unseen ECG\nMLII only\nLead-II-axis electrode\nplacement required", ORANGE, "#fff4e8", fs=6.25)
    box(0.225, 0.285, 0.16, 0.105,
        "Apply the same\n128 Hz, filtering,\nstandardisation and\nR alignment", ORANGE, "#fff9f2")
    box(0.425, 0.285, 0.15, 0.105,
        "MLII window\n+ RR history\n+ frozen-AE residual", ORANGE, "#fff9f2")
    box(0.615, 0.285, 0.15, 0.105,
        "242-value input\nFROZEN abnormality\ndetector", PURPLE, "#f3edfb")
    box(0.805, 0.335, 0.15, 0.065,
        "NORMAL\nreturn N", TEAL, "#e9f8f7", fs=7.0)
    box(0.615, 0.115, 0.15, 0.105,
        "If ABNORMAL:\nreuse the same\n242-value input", ORANGE, "#fff4e8")
    box(0.805, 0.115, 0.15, 0.105,
        "FROZEN subtype model\nPVC / PAC / LBBB\nRBBB / AFib", ORANGE, "#ffead2", fs=6.6)
    for x1, x2 in [(0.185, 0.225), (0.385, 0.425), (0.575, 0.615)]:
        arrow(x1, 0.337, x2, 0.337, ORANGE)
    arrow(0.765, 0.355, 0.805, 0.367, TEAL)
    arrow(0.690, 0.285, 0.690, 0.220, ORANGE)
    arrow(0.765, 0.167, 0.805, 0.167, ORANGE)
    ax.text(0.495, 0.078,
            "The fixed final 20% of each eligible MLII record follows path B. Expert annotations are used only after prediction to score the result.",
            ha="center", va="center", fontsize=6.4, color=GREY)

    fig.savefig(OUTPUT / "real_data_preparation_flow.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dataset_overview_figure()
    cleaning_figure()
    lead_position_figure()
    preparation_figure()
    print("Generated Chapter 3 real-data figures in", OUTPUT)
