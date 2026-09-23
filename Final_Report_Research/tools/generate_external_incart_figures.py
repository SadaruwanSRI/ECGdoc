"""Generate figures and a LaTeX fragment for the INCART external check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.ml.data import _download_record, preprocess_signal  # noqa: E402


RESULT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "storage"
    / "evaluations"
    / "latest_external_evaluation.json"
)
FIG_DIR = ROOT / "pic" / "generated"
TEX_PATH = ROOT / "content" / "final" / "5_results_external_incart.tex"


def pct(value: object) -> str:
    return f"{float(value) * 100:.2f}\\%"


def number4(value: object) -> str:
    return f"{float(value):.4f}"


def safe_tex(text: object) -> str:
    return str(text or "-").replace("&", "\\&").replace("%", "\\%")


def load_result() -> dict:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    return payload


def select_examples(result: dict, limit: int = 6) -> list[dict]:
    examples = []
    threshold = float(result.get("threshold") or 0.0)
    for row in result.get("examples", []):
        item = dict(row)
        score = float(item.get("score") or 0.0)
        item["predicted_anomaly"] = bool(item.get("predicted_anomaly", score > threshold))
        if item.get("true_anomaly") and item["predicted_anomaly"]:
            item["kind"] = "Detected abnormal"
        elif item.get("true_anomaly"):
            item["kind"] = "Missed abnormal"
        elif item["predicted_anomaly"]:
            item["kind"] = "False alarm"
        else:
            item["kind"] = "Correct normal"
        examples.append(item)

    rank = {
        "Detected abnormal": 0,
        "False alarm": 1,
        "Missed abnormal": 2,
        "Correct normal": 3,
    }
    examples.sort(
        key=lambda ex: (
            rank.get(ex["kind"], 9),
            -float(ex.get("score") or 0.0),
            int(ex.get("sample") or 0),
        )
    )
    return examples[:limit]


def plot_protocol() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8), constrained_layout=True)
    ax.set_axis_off()
    boxes = [
        (0.04, 0.55, 0.25, 0.28, "NSRDB normal ECG", "Frozen normal-reference\nautoencoder"),
        (0.37, 0.55, 0.25, 0.28, "MITDB MLII", "Temporal development +\nfinal temporal holdout"),
        (0.70, 0.55, 0.25, 0.28, "INCART lead II", "External, unseen-source\npost-freeze check"),
    ]
    colors = ["#dbeafe", "#dcfce7", "#fef3c7"]
    for (x, y, w, h, title, body), color in zip(boxes, colors):
        rect = plt.Rectangle((x, y), w, h, fc=color, ec="#475569", lw=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top",
                fontsize=11, weight="bold", color="#0f172a")
        ax.text(x + w / 2, y + 0.08, body, ha="center", va="bottom",
                fontsize=9, color="#334155")
    ax.annotate("", xy=(0.37, 0.69), xytext=(0.29, 0.69),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))
    ax.annotate("", xy=(0.70, 0.69), xytext=(0.62, 0.69),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))
    ax.text(
        0.5,
        0.30,
        "The main thesis result remains the MITDB temporal holdout. "
        "INCART is added as a separate external check after model selection.",
        ha="center",
        va="center",
        fontsize=10,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.text(
        0.5,
        0.12,
        "Lead-domain warning: INCART lead II is not identical to MITDB MLII.",
        ha="center",
        va="center",
        fontsize=9,
        color="#7c2d12",
    )
    fig.savefig(FIG_DIR / "external_incart_protocol.pdf")
    plt.close(fig)


def plot_metrics(result: dict) -> None:
    detection = result["detection"]
    confusion = detection["confusion"]
    metrics = [
        ("Accuracy", detection["accuracy"]),
        ("Balanced", detection["balanced_accuracy"]),
        ("Sensitivity", detection["sensitivity"]),
        ("Specificity", detection["specificity"]),
        ("Precision", detection["precision"]),
        ("F1", detection["f1"]),
    ]
    fig, (ax_bar, ax_conf) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [3, 2]},
        constrained_layout=True,
    )
    labels = [m[0] for m in metrics]
    values = [float(m[1]) * 100 for m in metrics]
    bars = ax_bar.bar(labels, values, color="#168C8C")
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel("Score (%)")
    ax_bar.set_title("INCART external-check metrics", loc="left", fontweight="bold")
    ax_bar.tick_params(axis="x", rotation=25, labelsize=8)
    ax_bar.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, min(value + 2, 102),
                    f"{value:.1f}%", ha="center", fontsize=8)

    matrix = np.array([
        [int(confusion.get("tp", 0)), int(confusion.get("fn", 0))],
        [int(confusion.get("fp", 0)), int(confusion.get("tn", 0))],
    ])
    ax_conf.imshow(matrix, cmap="YlGnBu")
    ax_conf.set_xticks([0, 1], ["Predicted\nabnormal", "Predicted\nnormal"], fontsize=8)
    ax_conf.set_yticks([0, 1], ["Real\nabnormal", "Real\nnormal"], fontsize=8)
    ax_conf.set_title("Binary confusion", loc="left", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax_conf.text(j, i, str(matrix[i, j]), ha="center", va="center",
                         fontsize=14, fontweight="bold", color="#0f172a")
    for spine in ax_conf.spines.values():
        spine.set_visible(False)
    fig.savefig(FIG_DIR / "external_incart_metrics_confusion.pdf")
    plt.close(fig)


def plot_class_summary(result: dict) -> None:
    classifier = result.get("classifier") or {}
    per_class = classifier.get("per_class") or {}
    supported = [(cls, row) for cls, row in per_class.items() if int(row.get("support") or 0) > 0]
    labels = [cls for cls, _ in supported]
    recall = [float(row.get("recall") or 0.0) * 100 for _, row in supported]
    precision = [float(row.get("precision") or 0.0) * 100 for _, row in supported]

    fig, ax = plt.subplots(figsize=(7.5, 3.7), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, recall, width, label="Recall", color="#17324D")
    ax.bar(x + width / 2, precision, width, label="Precision", color="#168C8C")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Supported INCART classes in the sampled subset", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for idx, (_, row) in enumerate(supported):
        ax.text(idx, 102, f"n={int(row.get('support') or 0)}", ha="center", fontsize=8)
    fig.savefig(FIG_DIR / "external_incart_class_summary.pdf")
    plt.close(fig)


def plot_examples(result: dict, examples: list[dict]) -> None:
    dataset = result["dataset"]
    record_cache: dict[str, np.ndarray] = {}
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.3), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, example in zip(axes, examples):
        record = str(example["record"])
        if record not in record_cache:
            raw, src_fs = _download_record(
                record,
                str(dataset["physionet_slug"]),
                required_lead=str(dataset["lead_name"]),
            )
            record_cache[record] = preprocess_signal(raw, src_fs)
        signal = record_cache[record]
        center = int(example["sample"])
        half = settings.WINDOW_SAMPLES // 2
        start = max(0, center - half)
        end = min(len(signal), center + half)
        seg = signal[start:end]
        t = (np.arange(start, end) - center) / settings.SAMPLING_RATE_HZ

        ax.plot(t, seg, color="#0f172a", linewidth=0.85)
        ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.0)
        ax.axvspan(-0.04, 0.04, color="#fecaca", alpha=0.55, lw=0)
        true_label = f"{example.get('true_class') or 'OTHER'} ({example.get('symbol')})"
        pred_class = example.get("predicted_class") or (
            "Abnormal" if example.get("predicted_anomaly") else "N"
        )
        pred_decision = "abnormal" if example.get("predicted_anomaly") else "normal"
        score = float(example.get("score") or 0.0)
        threshold = float(result.get("threshold") or 0.0)
        ax.set_title(
            f"{example['kind']}: {record} sample {center}",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.text(
            0.99,
            0.96,
            f"Real: {true_label}\nPred: {pred_decision} / {pred_class}\n"
            f"Score {score:.3f} vs {threshold:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.92),
        )
        ax.set_xlabel("Time from annotated beat (s)", fontsize=7.5)
        ax.set_ylabel("Amplitude", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(examples):]:
        ax.set_axis_off()
    fig.savefig(FIG_DIR / "external_incart_examples.pdf")
    plt.close(fig)


def write_tex(payload: dict, examples: list[dict]) -> None:
    result = payload["result"]
    dataset = result["dataset"]
    detection = result["detection"]
    classifier = result.get("classifier") or {}
    confusion = detection["confusion"]
    records_ok = [r for r in result.get("records", []) if r.get("status") == "ok"]
    example_rows = []
    for idx, ex in enumerate(examples, start=1):
        pred_class = ex.get("predicted_class") or (
            "Abnormal" if ex.get("predicted_anomaly") else "N"
        )
        pred_decision = "Abnormal" if ex.get("predicted_anomaly") else "Normal"
        example_rows.append(
            f"{idx} & {safe_tex(ex.get('record'))}/{int(ex.get('sample') or 0)} "
            f"& {safe_tex(ex.get('true_class'))} ({safe_tex(ex.get('symbol'))}) "
            f"& {pred_decision} / {safe_tex(pred_class)} "
            f"& {float(ex.get('score') or 0.0):.3f}\\\\"
        )

    text = rf"""
\section{{External INCART unseen-dataset check}}\label{{sec:external-incart}}

After the final MITDB temporal-holdout system had been fixed, a separate
external check was run on the St Petersburg INCART 12-lead Arrhythmia Database.
This is not the primary thesis benchmark and it is not used for model
selection. It is included to show how the frozen system behaves on an unseen
source database with a different acquisition setting. Lead II is used because
it is the closest practical counterpart to MITDB MLII, but it is not identical
to MLII. Therefore, this section measures both external-patient shift and
lead/device-domain shift.

\begin{{figure}}[H]
\centering
\includegraphics[width=.98\textwidth]{{pic/generated/external_incart_protocol.pdf}}
\caption{{Separation between the main MITDB temporal result and the INCART
external check. The external result is reported after model selection and is
not used to tune the frozen system.}}
\label{{fig:external-incart-protocol}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{Limited external INCART check on the latest saved evaluation artifact.}}
\label{{tab:external-incart-summary}}
\begin{{tabular}}{{ll}}
\toprule
Item & Value\\
\midrule
Dataset & {safe_tex(dataset.get('name'))}\\
Lead and mode & {safe_tex(dataset.get('lead_name'))}, {safe_tex(dataset.get('mode'))}\\
Records scored & {len(records_ok)} ({', '.join(safe_tex(r.get('record')) for r in records_ok)})\\
Annotated beats & {int(result.get('n_beats') or 0)}\\
Saved artifact time & {safe_tex(payload.get('saved_at'))}\\
Decision threshold & {float(result.get('threshold') or 0.0):.4f}\\
Accuracy & {pct(detection.get('accuracy'))}\\
Balanced accuracy & {pct(detection.get('balanced_accuracy'))}\\
Sensitivity / specificity & {pct(detection.get('sensitivity'))} / {pct(detection.get('specificity'))}\\
Precision / F1 & {pct(detection.get('precision'))} / {pct(detection.get('f1'))}\\
AUROC / AUPRC & {number4(detection.get('auroc'))} / {number4(detection.get('auprc'))}\\
Confusion counts & TP={confusion.get('tp', 0)}, TN={confusion.get('tn', 0)}, FP={confusion.get('fp', 0)}, FN={confusion.get('fn', 0)}\\
Subtype macro F1 & {number4(classifier.get('macro_f1')) if classifier else '-'}\\
\bottomrule
\end{{tabular}}
\end{{table}}

On this limited INCART subset, the frozen hierarchy reached {pct(detection.get('accuracy'))}
binary accuracy and {pct(detection.get('balanced_accuracy'))} balanced accuracy.
All {confusion.get('tp', 0)} annotated abnormal beats in the subset were detected
and {confusion.get('tn', 0)} normal beats were rejected, while {confusion.get('fp', 0)}
normal beats were false alarms and {confusion.get('fn', 0)} abnormal beats were
missed. Because the subset is small and contains only {int(result.get('n_beats') or 0)}
accepted beats from {len(records_ok)} INCART record, this result should be read
as an external sanity check, not as a definitive external clinical validation.

\begin{{figure}}[H]
\centering
\includegraphics[width=.96\textwidth]{{pic/generated/external_incart_metrics_confusion.pdf}}
\caption{{Metric bars and binary confusion matrix for the limited INCART
external check. Balanced accuracy is shown beside pooled accuracy because the
subset is class-imbalanced.}}
\label{{fig:external-incart-metrics}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=.75\textwidth]{{pic/generated/external_incart_class_summary.pdf}}
\caption{{Class-level precision and recall for classes that appear in the
limited INCART subset. Missing classes are not plotted because they have zero
support in this saved evaluation.}}
\label{{fig:external-incart-classes}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{Example annotated INCART beats used for visual inspection.}}
\label{{tab:external-incart-examples}}
\begin{{tabular}}{{rllll}}
\toprule
No. & Record/sample & Real type & Predicted decision/type & Score\\
\midrule
{chr(10).join(example_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=.98\textwidth]{{pic/generated/external_incart_examples.pdf}}
\caption{{Example INCART ECG segments centred on annotated beats. Each panel
shows the real annotation type, the model's abnormal/normal decision, the
predicted anomaly type, and the score-threshold comparison.}}
\label{{fig:external-incart-examples}}
\end{{figure}}
"""
    TEX_PATH.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_result()
    result = payload["result"]
    examples = select_examples(result)
    plot_protocol()
    plot_metrics(result)
    plot_class_summary(result)
    plot_examples(result, examples)
    write_tex(payload, examples)
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
