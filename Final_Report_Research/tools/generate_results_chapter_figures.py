"""Generate lightweight Results chapter figures from the saved final JSON.

This script avoids rerunning the experiment. It reads
experiments/corrected_temporal_holdout.json and creates explanatory figures for the
report's testing protocol and test-set composition.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import thesis_diagrams
except ImportError:
    import thesis_diagrams

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


matplotlib.use("Agg")

REPORT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = REPORT_DIR / "experiments" / "corrected_temporal_holdout.json"
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
    thesis_diagrams.testing_protocol(result)


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
    ax.set_title(f"Binary test support: {test_binary['support']:,} beats", loc="left", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:,}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(
        0.02,
        -0.29,
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
    ax.set_title(f"Six-class whole-system support: {result['test_whole_system']['support']:,} beats", loc="left", fontsize=10, weight="bold")
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
