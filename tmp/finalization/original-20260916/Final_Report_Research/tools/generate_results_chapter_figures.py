"""Generate lightweight Results chapter figures from the saved final JSON.

This script avoids rerunning the experiment. It reads
experiments/final_temporal_holdout.json and creates explanatory figures for the
report's testing protocol and test-set composition.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


matplotlib.use("Agg")

REPORT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = REPORT_DIR / "experiments" / "final_temporal_holdout.json"
OUTPUT_DIR = REPORT_DIR / "pic" / "generated"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def add_box(ax, xy, width, height, title, body, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 0.03,
        xy[1] + height - 0.055,
        title,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        color=color,
    )
    ax.text(
        xy[0] + 0.03,
        xy[1] + height - 0.125,
        body,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#111827",
        linespacing=1.12,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#475569",
        )
    )


def testing_protocol_figure(result: dict) -> None:
    protocol = result["protocol"]
    test_binary = result["test_binary"]
    test_whole = result["test_whole_system"]
    subtype = result["test_subtype_on_true_supported_arrhythmias"]
    unsupported = test_binary["support"] - test_whole["support"]

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.02,
        0.965,
        "How the final system was tested",
        fontsize=15,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.02,
        0.925,
        "Fixed within-patient temporal holdout: earlier MLII from each patient trains/selects the model; only the final 20% timeline is scored.",
        fontsize=9,
        color="#334155",
    )

    add_box(
        ax,
        (0.03, 0.63),
        0.25,
        0.24,
        "MITDB source",
        f"{protocol['records']} MLII-containing records\nrecords 102 and 104 excluded\n{protocol['independent_subjects']} subject clusters\none lead: MLII only",
        "#2563eb",
    )
    add_box(
        ax,
        (0.37, 0.63),
        0.25,
        0.24,
        "Development region",
        "first 64%: candidate fitting\n64-80%: validation selection\n16-beat boundary gap removed\nfinal models refit on earlier 80%",
        "#0f766e",
    )
    add_box(
        ax,
        (0.71, 0.63),
        0.25,
        0.24,
        "Temporal test region",
        f"final 20% of every eligible record\n{test_binary['support']:,} accepted beats\nno signal overlap with development\nknown-patient continuation",
        "#dc2626",
    )
    arrow(ax, (0.285, 0.75), (0.365, 0.75))
    arrow(ax, (0.625, 0.75), (0.705, 0.75))

    y = 0.48
    x0 = 0.06
    width = 0.88
    ax.text(0.03, 0.555, "Chronological split inside every eligible record", fontsize=10.5, weight="bold", color="#111827")
    segments = [
        ("candidate train", 0.64, "#93c5fd"),
        ("validation", 0.16, "#86efac"),
        ("test", 0.20, "#fca5a5"),
    ]
    current = x0
    for label, fraction, color in segments:
        rect = Rectangle((current, y), width * fraction, 0.07, facecolor=color, edgecolor="#334155", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(current + width * fraction / 2, y + 0.035, f"{label}\n{int(fraction * 100)}%", ha="center", va="center", fontsize=8.4, color="#111827")
        current += width * fraction
    ax.plot([x0 + width * 0.80, x0 + width * 0.80], [y - 0.035, y + 0.10], color="#7f1d1d", linestyle="--", linewidth=1.2)
    ax.text(x0 + width * 0.80, y - 0.055, "test boundary", ha="center", va="top", fontsize=8, color="#7f1d1d")

    add_box(
        ax,
        (0.03, 0.07),
        0.28,
        0.26,
        "1. Binary gate",
        f"all held-out beats scored\n{test_binary['normal_support']:,} normal\n{test_binary['abnormal_support']:,} abnormal\naccuracy {pct(test_binary['accuracy'])}",
        "#2563eb",
    )
    add_box(
        ax,
        (0.36, 0.07),
        0.28,
        0.26,
        "2. Subtype model",
        f"true supported abnormal beats only\n{subtype['support']:,} PVC/PAC/LBBB/RBBB/AFib beats\nconditional accuracy {pct(subtype['accuracy'])}\nunsupported abnormal beats: {unsupported:,}",
        "#7c3aed",
    )
    add_box(
        ax,
        (0.69, 0.07),
        0.28,
        0.26,
        "3. Whole system",
        f"N plus five supported arrhythmias\n{test_whole['support']:,} scored beats\naccuracy {pct(test_whole['accuracy'])}\nmacro F1 {result['headline']['whole_system_macro_f1']:.4f}",
        "#0f766e",
    )
    arrow(ax, (0.315, 0.20), (0.355, 0.20))
    arrow(ax, (0.645, 0.20), (0.685, 0.20))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_testing_protocol.pdf", bbox_inches="tight")
    plt.close(fig)


def test_composition_figure(result: dict) -> None:
    test_binary = result["test_binary"]
    per_class = result["test_whole_system"]["per_class"]
    arrhythmias = result["arrhythmia_table"]
    unsupported = test_binary["support"] - result["test_whole_system"]["support"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.suptitle("What data was used for the final temporal test", fontsize=14, weight="bold")

    ax = axes[0]
    groups = ["Normal", "Supported\narrhythmias", "Unsupported\nabnormal"]
    values = [
        test_binary["normal_support"],
        result["test_subtype_on_true_supported_arrhythmias"]["support"],
        unsupported,
    ]
    colors = ["#0f766e", "#2563eb", "#94a3b8"]
    bars = ax.bar(groups, values, color=colors)
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("Held-out beats")
    ax.set_title("Binary test support: 20,988 beats", loc="left", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:,}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(
        0.02,
        -0.22,
        "Unsupported abnormal symbols help test the normal/abnormal gate,\nbut they are not scored in the six-class confusion matrix.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#334155",
    )

    ax = axes[1]
    class_names = ["N"] + [row["type"] for row in arrhythmias]
    supports = [per_class["N"]["support"]] + [row["test_beats"] for row in arrhythmias]
    correct = [round(per_class["N"]["support"] * per_class["N"]["recall"])] + [row["correct_end_to_end"] for row in arrhythmias]
    x = range(len(class_names))
    ax.bar(x, supports, color="#cbd5e1", label="reference beats")
    ax.bar(x, correct, color="#0f766e", label="correct final output")
    ax.set_ylim(0, max(supports) * 1.28)
    ax.set_xticks(list(x), class_names)
    ax.set_ylabel("Supported beats")
    ax.set_title("Six-class whole-system support: 20,062 beats", loc="left", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    for idx, (support, ok) in enumerate(zip(supports, correct)):
        recall = ok / support if support else 0.0
        ax.text(idx, support + max(supports) * 0.018, f"{support:,}\n{100*recall:.1f}%", ha="center", va="bottom", fontsize=7.8, weight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "final_test_composition.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    testing_protocol_figure(result)
    test_composition_figure(result)
    print(OUTPUT_DIR / "final_testing_protocol.pdf")
    print(OUTPUT_DIR / "final_test_composition.pdf")


if __name__ == "__main__":
    main()
