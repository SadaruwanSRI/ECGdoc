"""Generate Chapter 3 figures from the locally stored PhysioNet ECG data."""

from pathlib import Path
import sys

try:
    from . import thesis_diagrams
except ImportError:
    import thesis_diagrams
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
    sys.path.insert(0, str(REPORT.parent / "backend"))
    from app.ml.data import bandpass_filter, resample_to
    from app.ml.beat_preparation import prepare_beat

    # Preserve the archived normal example used in the real-data audit.
    healthy_source = np.load(DATA / "nsrdb/16272.npy", mmap_mode="r")
    healthy_raw = np.asarray(healthy_source[:WINDOW])
    healthy_filtered = bandpass_filter(
        np.array(healthy_source[:30 * 60 * FS + WINDOW]), FS
    )[:WINDOW]
    healthy_window = np.array(np.load(
        DATA / "nsrdb-primary-full/16272_primary_full.npy", mmap_mode="r"
    )[0])
    design = np.column_stack([healthy_filtered, np.ones(WINDOW)])
    affine = np.linalg.lstsq(design, healthy_window, rcond=None)[0]
    np.testing.assert_allclose(design @ affine, healthy_window, atol=2e-6)

    # Arrhythmia-system example: a real annotated MLII beat from MITDB record 100.
    abnormal = wfdb.rdrecord(str(DATA / "mitdb" / "100"))
    annotation = wfdb.rdann(str(DATA / "mitdb" / "100"), "atr")
    abnormal_fs = int(abnormal.fs)
    eligible = annotation.sample[(annotation.sample > 60 * abnormal_fs) &
                                 (annotation.sample < 120 * abnormal_fs)]
    raw_peak = int(eligible[len(eligible) // 2])
    resampled = resample_to(abnormal.p_signal[:, 0].astype(np.float32), abnormal_fs, FS)
    peak = int(round(raw_peak * FS / abnormal_fs))
    abnormal_raw = resampled[peak - R_BEFORE:peak - R_BEFORE + WINDOW]
    sos = butter(4, [0.5, 50.0], btype="bandpass", fs=FS, output="sos")
    abnormal_filtered = sosfiltfilt(sos, abnormal_raw.astype(np.float64))
    abnormal_window = prepare_beat(abnormal_raw)
    np.testing.assert_allclose(standardise(abnormal_filtered), abnormal_window)

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 6.2), sharex="row",
                             constrained_layout=True)
    fig.suptitle("Real ECG cleaning for the two one-channel model inputs",
                 fontsize=13, fontweight="bold", color=NAVY)
    columns = [
        ("A. Autoencoder: NSRDB 16272 / ECG1",
         healthy_raw, FS, healthy_filtered, healthy_window, False),
        ("B. Detector/classifier: MITDB record 100, MLII",
         abnormal_raw, FS, abnormal_filtered, abnormal_window, True),
    ]
    for col, (title, raw, source_rate, filtered, window, aligned) in enumerate(columns):
        axes[0, col].plot(np.arange(raw.size) / source_rate, raw,
                          color=GREY, linewidth=0.75)
        axes[0, col].set_title(title + "\n1. Unfiltered four-second segment at 128 Hz",
                               fontsize=8.2, fontweight="bold", color=NAVY)
        style_axis(axes[0, col], "mV")
        axes[0, col].set_xlabel("source time (s)", fontsize=7, color=NAVY)

        axes[1, col].plot(np.arange(filtered.size) / FS, filtered,
                          color=TEAL, linewidth=0.8)
        filter_title = ("2. Filter this beat window: 0.5--50 Hz" if aligned else
                        "2. Filter source block; retain first four seconds")
        axes[1, col].set_title(filter_title,
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
            end_text = "3. Actual cached input; retain archived scaling"
        axes[2, col].set_title(end_text, fontsize=8.2, fontweight="bold", color=NAVY)
        style_axis(axes[2, col], "z score" if aligned else "cached amplitude")
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
    result = thesis_diagrams.corrected_result()
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
        "Denoising U-Net AE\n512-sample windows\nfour residual features",
        TEAL, "#f2fbfa")
    box(0.42, 0.20, 0.23, 0.24, "MLII feature table",
        "R-aligned MLII windows\nmorphology + spectrum\nRR context + AE residuals",
        PURPLE, "#f8f5fc")
    box(0.72, 0.57, 0.23, 0.28, "Development",
        "first 64%: fit candidates\nnext 16%: choose models\n16-beat gaps at boundaries",
        ORANGE, "#fff4e8")
    box(0.72, 0.16, 0.23, 0.28, "Temporal test",
        f"final 20% of accepted beats\n{result['test_binary']['support']:,} binary beats\n{result['test_whole_system']['support']:,} six-class beats",
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
    thesis_diagrams.lead_positions()


def preparation_figure() -> None:
    thesis_diagrams.data_preparation()


if __name__ == "__main__":
    dataset_overview_figure()
    cleaning_figure()
    lead_position_figure()
    preparation_figure()
    print("Generated Chapter 3 real-data figures in", OUTPUT)
