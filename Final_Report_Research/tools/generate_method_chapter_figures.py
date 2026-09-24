"""Generate clean Experimental Method chapter figures from the final result JSON."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import thesis_diagrams
except ImportError:
    import thesis_diagrams

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


matplotlib.use("Agg")

REPORT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = REPORT_DIR / "experiments" / "corrected_temporal_holdout.json"
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
    thesis_diagrams.method_workflow(result)


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
    values = [result["test_binary"]["support"],
              result["test_subtype_on_true_supported_arrhythmias"]["support"],
              result["test_whole_system"]["support"]]
    labels = ["Binary detection\nAll accepted test beats",
              "Subtype classification\nTrue supported abnormal beats",
              "Complete system\nNormal + supported abnormal"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(left=.38, right=.92, top=.79, bottom=.29)
    bars = ax.barh(range(3), values, color=[NAVY, PURPLE, TEAL], height=.53)
    ax.set_yticks(range(3), labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.19)
    ax.set_xlabel("Number of test beats", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    for bar, value in zip(bars, values):
        ax.text(value+350, bar.get_y()+bar.get_height()/2, f"{value:,}",
                va="center", fontsize=10, weight="bold", color=INK)
    fig.text(.06,.92,"Three distinct evaluation populations", fontsize=12,
             weight="bold", color=INK)
    fig.text(.06,.08,"These groups overlap. Unsupported abnormal labels enter only binary evaluation.",
             fontsize=9.5, color=SLATE)
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
