"""Generate clean Experimental Method chapter figures from the final result JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


matplotlib.use("Agg")

REPORT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = REPORT_DIR / "experiments" / "final_temporal_holdout.json"
OUTPUT_DIR = REPORT_DIR / "pic" / "generated"

INK = "#111827"
MUTED = "#64748b"
GRID = "#e5e7eb"
NAVY = "#183a59"
TEAL = "#0f766e"
ORANGE = "#c2410c"
PURPLE = "#6d28d9"
RED = "#b91c1c"
GOLD = "#b45309"
SLATE = "#475569"


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def clean_axis(ax, title: str | None = None) -> None:
    if title:
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
        ax.text((xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2 + 0.025, label,
                ha="center", va="center", fontsize=7.5, color=color,
                bbox=dict(fc="white", ec="none", boxstyle="round,pad=0.12", alpha=0.9))


def method_workflow_figure(result: dict) -> None:
    protocol = result["protocol"]
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.95, "Experimental method for the final temporal-holdout run",
            fontsize=15, weight="bold", color=INK)
    ax.text(0.02, 0.91,
            "One-lead MLII experiment: select with earlier beats, freeze decisions, score only the final 20% timeline.",
            fontsize=9, color=SLATE)

    lane_y = [0.76, 0.58, 0.40, 0.22]
    lane_labels = ["Data", "Feature evidence", "Model development", "Final validation"]
    lane_colors = [NAVY, PURPLE, TEAL, RED]
    for y, label, color in zip(lane_y, lane_labels, lane_colors):
        ax.hlines(y, 0.12, 0.93, color=color, lw=3, alpha=0.75)
        ax.text(0.03, y, label, va="center", fontsize=10, weight="bold", color=color)

    events = [
        (0.18, lane_y[0], "NSRDB ECG1\nnormal reference", NAVY),
        (0.38, lane_y[0], f"MITDB MLII\n{protocol['records']} records", NAVY),
        (0.22, lane_y[1], "223 waveform\nstatistics", PURPLE),
        (0.43, lane_y[1], "15 RR\ntiming", PURPLE),
        (0.61, lane_y[1], "4 AE\nresiduals", PURPLE),
        (0.18, lane_y[2], "0-64%\nfit candidates", TEAL),
        (0.36, lane_y[2], "64-80%\nvalidation", ORANGE),
        (0.55, lane_y[2], "0-80%\nfinal refit", TEAL),
        (0.73, lane_y[2], "80-100%\ntemporal test", RED),
        (0.25, lane_y[3], "binary\nall beats", RED),
        (0.47, lane_y[3], "subtype\nsupported abnormal", RED),
        (0.69, lane_y[3], "whole system\nsix classes", RED),
        (0.86, lane_y[3], "clustered\n95% CI", RED),
    ]
    for x, y, label, color in events:
        ax.scatter([x], [y], s=165, color=color, edgecolor="white", lw=1.6, zorder=4)
        ax.text(x, y + 0.065, label, ha="center", va="bottom", fontsize=8.2, color=INK)

    arrow(ax, (0.18, lane_y[0] - 0.025), (0.61, lane_y[1] + 0.025), NAVY, "frozen AE")
    arrow(ax, (0.38, lane_y[0] - 0.025), (0.43, lane_y[1] + 0.025), NAVY, "MLII beats")
    arrow(ax, (0.61, lane_y[1] - 0.025), (0.36, lane_y[2] + 0.025), PURPLE, "242 features", rad=0.10)
    arrow(ax, (0.55, lane_y[2] - 0.025), (0.25, lane_y[3] + 0.025), TEAL, "frozen model", rad=-0.12)
    arrow(ax, (0.73, lane_y[2] - 0.025), (0.69, lane_y[3] + 0.025), RED, "test only")

    ax.text(0.50, 0.06,
            "The test timeline is not used for candidate choice, threshold choice, or final fitting in this rerun.",
            ha="center", fontsize=9, color=INK)
    save(fig, "method_final_workflow.pdf")


def split_audit_figure(result: dict) -> None:
    audit = result["protocol"]["split_audit"]
    records = np.asarray(list(audit))
    accepted = np.asarray([audit[record]["accepted_beats"] for record in records])
    development = np.asarray([audit[record]["development_beats_after_gap"] for record in records])
    gap = np.asarray([audit[record]["boundary_gap_beats"] for record in records])
    test = np.asarray([audit[record]["temporal_test_beats"] for record in records])
    order = np.argsort(accepted)
    records = records[order]
    development = development[order]
    gap = gap[order]
    test = test[order]

    y = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(10.9, 7.3))
    ax.barh(y, development, color="#7dd3fc", label="final fitting region")
    ax.barh(y, gap, left=development, color="#f59e0b", label="16-beat boundary gap")
    ax.barh(y, test, left=development + gap, color="#f87171", label="final 20% temporal test")
    ax.set_yticks(y, records, fontsize=6.2)
    ax.set_xlabel("Accepted MLII beats per record")
    ax.set_ylabel("MITDB record")
    clean_axis(ax, "Temporal split audit for every eligible MITDB MLII record")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.text(
        0.01, -0.105,
        "Each horizontal bar is one record kept in chronological order. The red segment is the only fixed test region.",
        transform=ax.transAxes, fontsize=8.3, color=SLATE,
    )
    save(fig, "method_temporal_split_audit.pdf")


def model_selection_figure(result: dict) -> None:
    binary_rows = result["binary_selection"]
    subtype_rows = result["subtype_selection"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    fig.suptitle("Validation-region model selection", fontsize=15, weight="bold", x=0.02, ha="left")

    ax = axes[0]
    names = [row["candidate"]["name"] for row in binary_rows]
    values = np.asarray([row["validation"]["balanced_accuracy"] for row in binary_rows])
    x = np.arange(len(values))
    ax.plot(x, values, color="#94a3b8", lw=2.0)
    ax.scatter(x, values, s=110, color="#cbd5e1", edgecolor="white", lw=1.3, zorder=3)
    best = int(np.argmax(values))
    ax.scatter([best], [values[best]], s=170, color=TEAL, edgecolor="white", lw=1.4, zorder=4)
    ax.set_xticks(x, names)
    ax.set_ylim(max(0.955, values.min() - 0.006), 1.001)
    ax.set_ylabel("Balanced accuracy")
    clean_axis(ax, "A. Binary detector candidates")
    for xi, row, value in zip(x, binary_rows, values):
        ax.text(xi, value + 0.0014, f"{100*value:.2f}%\nthr {row['threshold']:.3f}",
                ha="center", va="bottom", fontsize=7.4, weight="bold", color=INK)

    ax = axes[1]
    names = [row["candidate"]["name"] for row in subtype_rows]
    values = np.asarray([row["validation"]["macro_f1"] for row in subtype_rows])
    x = np.arange(len(values))
    ax.plot(x, values, color="#94a3b8", lw=2.0)
    ax.scatter(x, values, s=110, color="#cbd5e1", edgecolor="white", lw=1.3, zorder=3)
    best = int(np.argmax(values))
    ax.scatter([best], [values[best]], s=170, color=PURPLE, edgecolor="white", lw=1.4, zorder=4)
    ax.set_xticks(x, names)
    ax.set_ylim(max(0.945, values.min() - 0.008), 1.001)
    ax.set_ylabel("Macro F1")
    clean_axis(ax, "B. Subtype candidates")
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.0014, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7.6, weight="bold", color=INK)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "method_model_selection.pdf")


def evaluation_order_figure(result: dict) -> None:
    binary = result["test_binary"]
    subtype = result["test_subtype_on_true_supported_arrhythmias"]
    whole = result["test_whole_system"]

    fig = plt.figure(figsize=(11.4, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.12, 1.0], width_ratios=[1.10, 1.0], wspace=0.30)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.5, "Evaluation order and test support", fontsize=15,
                  weight="bold", color=INK, va="center")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_axis_off()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    stages = [
        (0.12, 0.70, "1", "Binary gate", f"{binary['support']:,} held-out beats", NAVY),
        (0.48, 0.70, "2", "Subtype naming", f"{subtype['support']:,} supported abnormal beats", PURPLE),
        (0.84, 0.70, "3", "Whole system", f"{whole['support']:,} supported six-class beats", TEAL),
    ]
    for x, y, number, title, body, color in stages:
        ax1.scatter([x], [y], s=950, color=color, alpha=0.92, edgecolor="white", lw=2.0, zorder=3)
        ax1.text(x, y, number, ha="center", va="center", fontsize=18, weight="bold", color="white")
        ax1.text(x, y - 0.18, title, ha="center", fontsize=10.5, weight="bold", color=INK)
        ax1.text(x, y - 0.27, body, ha="center", fontsize=8.3, color=SLATE)
    arrow(ax1, (0.22, 0.70), (0.38, 0.70), SLATE)
    arrow(ax1, (0.58, 0.70), (0.74, 0.70), SLATE)
    ax1.text(0.50, 0.15,
             "Separate supports identify where errors happen: abnormal gate, conditional subtype name, or complete deployed decision.",
             ha="center", fontsize=8.8, color=INK)

    ax2 = fig.add_subplot(gs[1, 1])
    labels = ["binary\nall", "subtype\nsupported abnormal", "whole\nsix-class"]
    supports = np.asarray([binary["support"], subtype["support"], whole["support"]])
    colors = [NAVY, PURPLE, TEAL]
    bars = ax2.bar(labels, supports, color=colors, alpha=0.86)
    ax2.set_ylabel("Beats included in score")
    clean_axis(ax2, "Support differs by evaluation question")
    for bar, support in zip(bars, supports):
        ax2.text(bar.get_x() + bar.get_width() / 2, support + 350, f"{support:,}",
                 ha="center", va="bottom", fontsize=8.5, weight="bold")
    ax2.set_ylim(0, max(supports) * 1.20)
    save(fig, "method_evaluation_order.pdf")


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    method_workflow_figure(result)
    split_audit_figure(result)
    model_selection_figure(result)
    evaluation_order_figure(result)
    for name in [
        "method_final_workflow.pdf",
        "method_temporal_split_audit.pdf",
        "method_model_selection.pdf",
        "method_evaluation_order.pdf",
    ]:
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
