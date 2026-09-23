"""Generate physician-ready PDF reports for ECG sessions.

Uses ReportLab — no external PDF engine needed. Layout:
- Cover banner (title + session metadata)
- Patient / session info table
- ECG waveform plot (matplotlib → PNG → embed)
- Detection summary (total beats, anomalies, duration, threshold)
- Alerts table (timestamp, severity, score)
- Physician notes section (blank lines for signing)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in ('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from app.core.config import settings
from app.db.session import get_db
from app.ml.health_info import get_health_info
from app.ml.data import _download_record, preprocess_signal
from sqlalchemy import text


PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONFIDENCE_THRESHOLD = 0.50
MIN_REPORTABLE_EXTERNAL_BEATS = 100
MAX_EVALUATION_EXAMPLE_FIGURES = 6


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontName="Helvetica-Bold", fontSize=22,
                                textColor=colors.HexColor("#0f172a"),
                                alignment=TA_LEFT, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontName="Helvetica", fontSize=11,
                                   textColor=colors.HexColor("#475569"),
                                   alignment=TA_LEFT, spaceAfter=18),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontName="Helvetica-Bold", fontSize=13,
                             textColor=colors.HexColor("#0f172a"),
                             spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontName="Helvetica", fontSize=10.5,
                               leading=15, alignment=TA_JUSTIFY,
                               textColor=colors.HexColor("#1e293b")),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontName="Helvetica", fontSize=9,
                                textColor=colors.HexColor("#64748b")),
    }


def _pdf_text(value: object) -> str:
    """Escape text before inserting it into ReportLab Paragraph markup."""
    return escape(str(value or ""))


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _num(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _number4(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_final_temporal_result() -> Optional[dict]:
    path = (
        settings.PROJECT_ROOT
        / "Final_Report_Research"
        / "experiments"
        / "corrected_temporal_holdout.json"
    )
    return _read_json(path)


def _load_latest_external_evaluation() -> Optional[dict]:
    path = settings.STORAGE_DIR / "evaluations" / "latest_external_evaluation.json"
    payload = _read_json(path)
    if not payload:
        return None
    result = payload.get("result") or {}
    dataset = result.get("dataset") or {}
    if dataset.get("validation_kind") != "external":
        return None
    if int(result.get("n_beats") or 0) < MIN_REPORTABLE_EXTERNAL_BEATS:
        return None
    return payload


def _alert_classification(alert: dict) -> Optional[dict]:
    """Return classification stored on an alert or inside its context."""
    cls = alert.get("classification")
    if cls:
        return cls
    ctx = alert.get("context", {}) or {}
    return ctx.get("classification")


def _alert_identity(alert: dict) -> tuple:
    """Build a stable identity for one detected beat.

    New live alerts contain ``beat_id``. Older reports are deduplicated by the
    R-peak/sample location stored in ``context.t``. The final fallback is only
    used for historical alerts that contain neither field.
    """
    ctx = alert.get("context", {}) or {}
    beat_id = ctx.get("beat_id")
    if beat_id:
        return ("beat", str(beat_id))
    beat_t = ctx.get("t")
    if beat_t is not None:
        try:
            return ("sample", int(round(float(beat_t))))
        except (TypeError, ValueError):
            return ("sample", str(beat_t))
    cls = _alert_classification(alert) or {}
    return (
        "legacy",
        str(alert.get("timestamp") or ""),
        str(cls.get("class") or ""),
        round(float(alert.get("anomaly_score") or 0.0), 6),
    )


def _deduplicate_alerts(alerts: list[dict]) -> list[dict]:
    """Return one alert per detected beat, keeping the strongest duplicate."""
    unique: dict[tuple, dict] = {}
    order: list[tuple] = []
    for alert in alerts:
        key = _alert_identity(alert)
        existing = unique.get(key)
        if existing is None:
            unique[key] = alert
            order.append(key)
            continue
        existing_cls = _alert_classification(existing) or {}
        candidate_cls = _alert_classification(alert) or {}
        existing_strength = (
            float(existing.get("anomaly_score") or 0.0),
            float(existing_cls.get("confidence") or 0.0),
        )
        candidate_strength = (
            float(alert.get("anomaly_score") or 0.0),
            float(candidate_cls.get("confidence") or 0.0),
        )
        if candidate_strength > existing_strength:
            unique[key] = alert
    return [unique[key] for key in order]


def _model_output_summary(alerts: list[dict]) -> list[dict]:
    """Group unique beats by model output and select one waveform per output."""
    grouped: dict[str, dict] = {}
    for alert in _deduplicate_alerts(alerts):
        cls = _alert_classification(alert) or {}
        code = str(cls.get("class") or "Abnormal")
        if code == "N":
            # An alert means the binary detector crossed its threshold. A
            # normal subtype prediction must therefore be shown as uncertain,
            # rather than as a contradictory normal finding.
            code = "Unclassified abnormal"
            class_name = "Unclassified abnormal pattern"
        else:
            class_name = str(cls.get("class_name") or code)
        entry = grouped.setdefault(code, {
            "class": code,
            "class_name": class_name,
            "count": 0,
            "probability_sum": 0.0,
            "confidence_sum": 0.0,
            "example": alert,
        })
        probability = float(alert.get("anomaly_score") or 0.0)
        confidence = float(cls.get("confidence") or 0.0)
        entry["count"] += 1
        entry["probability_sum"] += probability
        entry["confidence_sum"] += confidence

        example = entry["example"]
        example_cls = _alert_classification(example) or {}
        current_rank = (
            float(example.get("anomaly_score") or 0.0)
            - float(example.get("threshold") or 0.0),
            float(example_cls.get("confidence") or 0.0),
        )
        candidate_rank = (
            probability - float(alert.get("threshold") or 0.0),
            confidence,
        )
        if candidate_rank > current_rank:
            entry["example"] = alert

    rows = []
    for entry in grouped.values():
        entry["avg_probability"] = (
            entry["probability_sum"] / max(1, entry["count"])
        )
        entry["avg_confidence"] = (
            entry["confidence_sum"] / max(1, entry["count"])
        )
        rows.append(entry)
    rows.sort(
        key=lambda row: (
            row["count"],
            row["avg_probability"],
            row["avg_confidence"],
        ),
        reverse=True,
    )
    return rows


def _high_confidence_alerts(alerts: list[dict],
                            min_confidence: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    selected = []
    for alert in alerts:
        cls = _alert_classification(alert)
        if not cls:
            continue
        confidence = float(cls.get("confidence") or 0.0)
        arrhythmia_class = cls.get("class")
        if (
            confidence >= min_confidence
            and arrhythmia_class
            and arrhythmia_class not in {"N", "Unclassified abnormal"}
        ):
            item = dict(alert)
            item["classification"] = cls
            selected.append(item)
    return selected


def _arrhythmia_summary(alerts: list[dict]) -> tuple[list[dict], Optional[dict]]:
    """Group high-confidence arrhythmia detections and pick a main type."""
    grouped: dict[str, dict] = {}
    for alert in _high_confidence_alerts(alerts):
        cls = alert["classification"]
        code = cls.get("class")
        if not code:
            continue
        entry = grouped.setdefault(code, {
            "class": code,
            "class_name": cls.get("class_name", code),
            "count": 0,
            "confidence_sum": 0.0,
            "max_confidence": 0.0,
            "example": alert,
        })
        confidence = float(cls.get("confidence") or 0.0)
        entry["count"] += 1
        entry["confidence_sum"] += confidence
        if confidence > entry["max_confidence"]:
            entry["max_confidence"] = confidence
            entry["example"] = alert

    rows = []
    for entry in grouped.values():
        entry["avg_confidence"] = entry["confidence_sum"] / max(1, entry["count"])
        rows.append(entry)
    rows.sort(key=lambda r: (r["count"], r["max_confidence"]), reverse=True)
    return rows, rows[0] if rows else None


def _selected_external_examples(result: dict) -> list[dict]:
    examples = result.get("examples") or []
    threshold = float(result.get("threshold") or 0.0)
    enriched = []
    seen = set()
    for ex in examples:
        key = (ex.get("record"), ex.get("sample"))
        if key in seen:
            continue
        seen.add(key)
        item = dict(ex)
        score = float(item.get("score") or 0.0)
        item["predicted_anomaly"] = bool(item.get("predicted_anomaly", score > threshold))
        item["error_kind"] = (
            "true_positive"
            if item.get("true_anomaly") and item["predicted_anomaly"]
            else "false_negative"
            if item.get("true_anomaly")
            else "false_positive"
            if item["predicted_anomaly"]
            else "true_negative"
        )
        enriched.append(item)

    rank = {
        "true_positive": 0,
        "false_negative": 1,
        "false_positive": 2,
        "true_negative": 3,
    }
    enriched.sort(
        key=lambda ex: (
            rank.get(ex["error_kind"], 9),
            -float(ex.get("score") or 0.0),
            str(ex.get("record") or ""),
        )
    )
    return enriched[:MAX_EVALUATION_EXAMPLE_FIGURES]


def _predicted_type_label(example: dict) -> str:
    predicted_class = example.get("predicted_class")
    if predicted_class:
        return str(predicted_class)
    return "Unclassified abnormal" if example.get("predicted_anomaly") else "No anomaly type"


def _plot_external_evaluation_summary(result: dict, out_path: Path) -> None:
    detection = result.get("detection") or {}
    confusion = detection.get("confusion") or {}
    metrics = [
        ("Accuracy", detection.get("accuracy")),
        ("Balanced", detection.get("balanced_accuracy")),
        ("Sensitivity", detection.get("sensitivity")),
        ("Specificity", detection.get("specificity")),
        ("Precision", detection.get("precision")),
        ("F1", detection.get("f1")),
    ]
    labels = [name for name, value in metrics if value is not None]
    values = [float(value) * 100 for _, value in metrics if value is not None]

    fig, (ax_metrics, ax_confusion) = plt.subplots(
        1, 2, figsize=(10.8, 3.2), gridspec_kw={"width_ratios": [3, 2]},
        constrained_layout=True,
    )
    bars = ax_metrics.bar(labels, values, color="#e11d48")
    ax_metrics.set_ylim(0, 100)
    ax_metrics.set_ylabel("Percent")
    ax_metrics.set_title("External test metrics", loc="left", fontsize=10)
    ax_metrics.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax_metrics.tick_params(axis="y", labelsize=8)
    ax_metrics.grid(axis="y", alpha=0.25, linewidth=0.4)
    for bar, value in zip(bars, values):
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 98),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#475569",
        )

    matrix = np.asarray([
        [int(confusion.get("tp", 0)), int(confusion.get("fn", 0))],
        [int(confusion.get("fp", 0)), int(confusion.get("tn", 0))],
    ])
    ax_confusion.imshow(matrix, cmap="Reds")
    ax_confusion.set_xticks([0, 1], ["Pred abnormal", "Pred normal"], fontsize=8)
    ax_confusion.set_yticks([0, 1], ["Real abnormal", "Real normal"], fontsize=8)
    ax_confusion.set_title("Binary confusion", loc="left", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax_confusion.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                color="#0f172a",
                fontweight="bold",
            )
    for spine in ax_confusion.spines.values():
        spine.set_visible(False)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_external_evaluation_example(result: dict, example: dict,
                                      out_path: Path, idx: int) -> bool:
    dataset = result.get("dataset") or {}
    try:
        raw_signal, src_fs = _download_record(
            str(example.get("record")),
            str(dataset.get("physionet_slug") or ""),
            required_lead=str(dataset.get("lead_name") or ""),
        )
        signal = preprocess_signal(raw_signal, src_fs)
    except Exception:
        return False

    center = int(example.get("sample") or 0)
    half = settings.WINDOW_SAMPLES // 2
    start = max(0, center - half)
    end = min(len(signal), center + half)
    if end <= start:
        return False

    segment = signal[start:end].astype(float)
    t = (np.arange(start, end) - center) / settings.SAMPLING_RATE_HZ
    threshold = float(result.get("threshold") or 0.0)
    score = float(example.get("score") or 0.0)
    predicted_anomaly = bool(example.get("predicted_anomaly", score > threshold))
    example["predicted_anomaly"] = predicted_anomaly
    pred_type = _predicted_type_label(example)
    real_type = example.get("true_class") or (
        "Abnormal" if example.get("true_anomaly") else "Normal"
    )
    real_symbol = example.get("symbol") or "-"

    fig, ax = plt.subplots(figsize=(5.3, 2.45), constrained_layout=True)
    ax.plot(t, segment, color="#0f172a", linewidth=1.0)
    ax.axvline(0, color="#dc2626", linewidth=1.0, linestyle="--")
    ax.axvspan(-0.04, 0.04, color="#fecaca", alpha=0.6, lw=0)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_xlabel("Time from annotated beat (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Example {idx}: record {example.get('record')} sample {center}",
        fontsize=9,
        loc="left",
        color="#0f172a",
    )
    detail = (
        f"Real: {real_type} ({real_symbol})\n"
        f"Pred: {'anomaly' if predicted_anomaly else 'normal'} / {pred_type}\n"
        f"Score {score:.4f} vs threshold {threshold:.4f}"
    )
    ax.text(
        0.99,
        0.96,
        detail,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cbd5e1", alpha=0.92),
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _append_external_evaluation_figures(story: list, styles: dict,
                                        result: dict, session_id: str) -> None:
    summary_path = settings.REPORT_DIR / f"session_{session_id}_external_summary.png"
    _plot_external_evaluation_summary(result, summary_path)
    story.append(Paragraph("External Test Figures", styles["h2"]))
    story.append(Paragraph(
        "The chart summarizes the saved external INCART evaluation. The example "
        "segments below are centered on annotated beats and show real annotation "
        "type, predicted anomaly decision, predicted anomaly type when available, "
        "and the score-threshold comparison.",
        styles["body"],
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(RLImage(str(summary_path), width=170 * mm, height=50 * mm))

    examples = _selected_external_examples(result)
    if examples:
        example_rows = [[
            "Example",
            "Record / sample",
            "Real type",
            "Predicted decision",
            "Predicted type",
            "Score",
        ]]
        threshold = float(result.get("threshold") or 0.0)
        for idx, example in enumerate(examples, start=1):
            score = float(example.get("score") or 0.0)
            real_type = example.get("true_class") or (
                "Abnormal" if example.get("true_anomaly") else "Normal"
            )
            real_symbol = example.get("symbol") or "-"
            example_rows.append([
                str(idx),
                f"{example.get('record')} / {example.get('sample')}",
                Paragraph(_pdf_text(f"{real_type} ({real_symbol})"), styles["small"]),
                "Anomaly" if example.get("predicted_anomaly") else "Normal",
                Paragraph(_pdf_text(_predicted_type_label(example)), styles["small"]),
                f"{score:.4f} vs {threshold:.4f}",
            ])
        story.append(Spacer(1, 3 * mm))
        example_meta_tbl = Table(
            example_rows,
            colWidths=[15 * mm, 31 * mm, 34 * mm, 32 * mm, 32 * mm, 26 * mm],
            repeatRows=1,
        )
        example_meta_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(example_meta_tbl)

    image_rows = []
    generated = []
    for idx, example in enumerate(examples, start=1):
        img_path = settings.REPORT_DIR / f"session_{session_id}_external_example_{idx}.png"
        if _plot_external_evaluation_example(result, example, img_path, idx):
            generated.append((img_path, example))

    for i in range(0, len(generated), 2):
        left = RLImage(str(generated[i][0]), width=82 * mm, height=38 * mm)
        right = (
            RLImage(str(generated[i + 1][0]), width=82 * mm, height=38 * mm)
            if i + 1 < len(generated)
            else ""
        )
        image_rows.append([left, right])

    if image_rows:
        story.append(Spacer(1, 3 * mm))
        examples_tbl = Table(image_rows, colWidths=[85 * mm, 85 * mm])
        examples_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(examples_tbl)
    else:
        story.append(Paragraph(
            "No plottable ECG example segments were available for the saved "
            "external evaluation.",
            styles["small"],
        ))


def _append_validation_context(story: list, styles: dict, session_id: str) -> None:
    """Add research benchmark context to the session PDF."""
    story.append(Paragraph("Research Validation Context", styles["h2"]))
    story.append(Paragraph(
        "This section reports model-validation evidence separately from the "
        "patient/session recording above. It must not be read as a diagnosis for "
        "this session.",
        styles["body"],
    ))

    final_result = _load_final_temporal_result()
    if final_result:
        protocol = final_result.get("protocol") or {}
        test_binary = final_result.get("test_binary") or {}
        whole_system = final_result.get("test_whole_system") or {}
        generalization_claim = (
            protocol.get("generalization_claim")
            or "unseen temporal signal from known patients; not unseen-patient generalization"
        )
        final_rows = [
            ["Final Report benchmark", "Value"],
            ["Protocol", Paragraph(_pdf_text(protocol.get("name") or "Final within-patient temporal holdout"), styles["small"])],
            ["Generalization claim", Paragraph(_pdf_text(generalization_claim), styles["small"])],
            ["Binary accuracy", _pct(test_binary.get("accuracy"))],
            ["Binary balanced accuracy", _pct(test_binary.get("balanced_accuracy"))],
            ["Whole-system accuracy", _pct(whole_system.get("accuracy"))],
            ["Whole-system macro F1", _number4(whole_system.get("macro_f1"))],
        ]
        final_tbl = Table(final_rows, colWidths=[55 * mm, 110 * mm])
        final_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(Spacer(1, 2 * mm))
        story.append(final_tbl)
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            "The Final Report states that the fixed MITDB result is later signal "
            "from known patients and is not proof of cold-start performance on "
            "completely unseen patients.",
            styles["small"],
        ))
    else:
        story.append(Paragraph(
            "The saved Final Report temporal-holdout result was not found in the "
            "local project artifacts.",
            styles["small"],
        ))

    external_payload = _load_latest_external_evaluation()
    if external_payload:
        result = external_payload.get("result") or {}
        dataset = result.get("dataset") or {}
        detection = result.get("detection") or {}
        confusion = detection.get("confusion") or {}
        ok_records = [
            row for row in (result.get("records") or [])
            if row.get("status") == "ok"
        ]
        external_rows = [
            ["Totally unseen external dataset test", "Latest saved result"],
            ["Dataset", Paragraph(_pdf_text(dataset.get("name") or dataset.get("id") or "External dataset"), styles["small"])],
            ["Lead / mode", f"{dataset.get('lead_name') or '-'} / {dataset.get('mode') or '-'}"],
            ["Saved at", _pdf_text(external_payload.get("saved_at") or "-")],
            ["Records scored", f"{len(ok_records):,}"],
            ["Annotated beats", _num(result.get("n_beats"))],
            ["Threshold", f"{float(result.get('threshold') or 0.0):.4f}"],
            ["Accuracy", _pct(detection.get("accuracy"))],
            ["Balanced accuracy", _pct(detection.get("balanced_accuracy"))],
            ["Sensitivity / specificity", f"{_pct(detection.get('sensitivity'))} / {_pct(detection.get('specificity'))}"],
            ["F1 / precision", f"{_pct(detection.get('f1'))} / {_pct(detection.get('precision'))}"],
            ["AUROC / AUPRC", f"{_number4(detection.get('auroc'))} / {_number4(detection.get('auprc'))}"],
            ["Confusion TP/TN/FP/FN", f"{confusion.get('tp', 0)} / {confusion.get('tn', 0)} / {confusion.get('fp', 0)} / {confusion.get('fn', 0)}"],
        ]
        external_tbl = Table(external_rows, colWidths=[55 * mm, 110 * mm])
        external_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        external_block = [Spacer(1, 3 * mm), external_tbl]
        warning = dataset.get("warning")
        if warning:
            external_block.extend([
                Spacer(1, 2 * mm),
                Paragraph(_pdf_text(warning), styles["small"]),
            ])
        story.append(KeepTogether(external_block))
        _append_external_evaluation_figures(story, styles, result, session_id)
    else:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "No reportable external unseen-dataset evaluation has been saved yet. "
            "Run the Performance Test with the St Petersburg INCART 12-lead "
            f"Arrhythmia Database first using at least "
            f"{MIN_REPORTABLE_EXTERNAL_BEATS} annotated beats; the next generated "
            "session report will include that saved result here.",
            styles["small"],
        ))


def _plot_ecg(points: list, alerts: list, out_path: Path,
              max_points: int = 4000) -> None:
    """Plot ECG signal with anomaly regions highlighted."""
    if not points:
        # Empty placeholder
        fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
        ax.text(0.5, 0.5, "No ECG data recorded for this session.",
                ha="center", va="center", fontsize=11, color="#64748b")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Downsample if too many points
    if len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]

    t = [p["t"] for p in points]
    v = [p["value"] for p in points]
    a_mask = [p["is_anomaly"] for p in points]

    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    # Background highlight for anomaly regions
    i = 0
    while i < len(a_mask):
        if a_mask[i]:
            j = i
            while j + 1 < len(a_mask) and a_mask[j + 1]:
                j += 1
            ax.axvspan(t[i], t[j], color="#fecaca", alpha=0.5, lw=0)
            i = j + 1
        else:
            i += 1

    ax.plot(t, v, color="#0f172a", linewidth=0.7, label="ECG signal")
    # Mark alert points
    if alerts:
        labeled = 0
        for a in alerts[:30]:  # cap to avoid clutter
            ctx = a.get("context", {}) or {}
            tt = ctx.get("t")
            if tt is not None:
                ax.axvline(tt, color="#dc2626", linewidth=0.6, alpha=0.5)
                cls = _alert_classification(a)
                confidence = float((cls or {}).get("confidence") or 0.0)
                if cls and confidence >= CONFIDENCE_THRESHOLD and labeled < 10:
                    ax.text(
                        tt, 0.98, f"{cls.get('class', 'ARR')} {confidence:.0%}",
                        transform=ax.get_xaxis_transform(),
                        rotation=90, va="top", ha="right",
                        fontsize=7, color="#991b1b",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#fee2e2", ec="#fecaca", alpha=0.85),
                    )
                    labeled += 1

    ax.set_xlabel("Sample index (128 Hz)", fontsize=9)
    ax.set_ylabel("Amplitude (z-scored)", fontsize=9)
    ax.set_title("ECG Recording — Anomaly Detection", fontsize=11,
                 color="#0f172a", loc="left", pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_anomaly_waveform(alert: dict, out_path: Path, idx: int) -> None:
    """Plot one unique abnormal beat together with the model outputs."""
    ctx = alert.get("context", {}) or {}
    signal = ctx.get("signal", [])
    recon = ctx.get("reconstruction", [])
    fs = float(ctx.get("fs") or 64)
    classification = _alert_classification(alert)

    if not signal:
        fig, ax = plt.subplots(figsize=(5.2, 2.0), constrained_layout=True)
        ax.text(0.5, 0.5, "Waveform not available",
                ha="center", va="center", fontsize=10, color="#94a3b8")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    t = [i / fs for i in range(len(signal))]
    probabilities = (classification or {}).get("probabilities", {}) or {}
    has_probability_output = bool(probabilities)
    fig_height = 3.5 if has_probability_output else 2.2

    if has_probability_output:
        fig, (ax_wave, ax_probs) = plt.subplots(
            2, 1, figsize=(5.2, fig_height),
            gridspec_kw={"height_ratios": [3, 1.25]},
            constrained_layout=True,
        )
    else:
        fig, ax_wave = plt.subplots(
            figsize=(5.2, fig_height), constrained_layout=True
        )

    ax_wave.plot(t, signal, color="#0f172a", linewidth=1.0, label="ECG signal")
    r_peak_index = ctx.get("r_peak_index")
    if r_peak_index is not None:
        try:
            r_peak_time = float(r_peak_index) / fs
            ax_wave.axvspan(
                r_peak_time - 0.04,
                r_peak_time + 0.04,
                color="#fecaca",
                alpha=0.65,
                lw=0,
                label="Detected abnormal beat",
            )
            ax_wave.axvline(
                r_peak_time,
                color="#dc2626",
                linewidth=1.2,
                linestyle="--",
                label="R peak",
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if recon:
        ax_wave.plot(
            t, recon, color="#0891b2", linewidth=0.9, linestyle="--",
            label="Reconstruction",
        )
        ax_wave.fill_between(
            t, signal, recon, color="#fecaca", alpha=0.4,
            label="Reconstruction error",
        )

    severity = str(alert.get("severity") or "warning").upper()
    score = float(alert.get("anomaly_score") or 0.0)
    threshold = float(alert.get("threshold") or 0.0)
    margin = score - threshold
    probability_mode = ctx.get("score_type") == "abnormal_probability"

    if classification:
        display_code = str(classification.get("class") or "")
        if display_code == "N":
            class_name = "Unclassified abnormal pattern"
        else:
            class_name = str(
                classification.get("class_name")
                or display_code
                or "Unclassified abnormal pattern"
            )
        confidence = float(classification.get("confidence") or 0.0)
        if probability_mode:
            decision_line = (
                f"Abnormal probability {score:.1%} > threshold {threshold:.1%} "
                f"(margin {margin:+.1%})"
            )
        else:
            decision_line = (
                f"Reconstruction error {score:.4f} > threshold {threshold:.4f}"
            )
        ax_wave.set_title(
            f"Unique output #{idx}: {class_name} | "
            f"subtype confidence {confidence:.1%}\n{decision_line}",
            fontsize=8.4,
            color="#0f172a",
            loc="left",
            pad=4,
        )
    else:
        ax_wave.set_title(
            f"Unique output #{idx}: {severity}\n"
            f"Anomaly score {score:.4f} > threshold {threshold:.4f}",
            fontsize=8.4,
            color="#0f172a",
            loc="left",
            pad=4,
        )

    ax_wave.set_xlabel("Time (s)", fontsize=8)
    ax_wave.set_ylabel("Amplitude", fontsize=8)
    ax_wave.grid(True, alpha=0.3, linewidth=0.4)
    ax_wave.tick_params(axis="both", labelsize=7)
    ax_wave.spines["top"].set_visible(False)
    ax_wave.spines["right"].set_visible(False)
    ax_wave.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

    if has_probability_output:
        top_class = str((classification or {}).get("class") or "")
        classes = sorted(
            probabilities.keys(), key=lambda key: probabilities[key], reverse=True
        )
        values = [float(probabilities[key]) * 100 for key in classes]
        bar_colors = [
            "#dc2626" if key == top_class else "#94a3b8" for key in classes
        ]
        bars = ax_probs.barh(classes, values, color=bar_colors, height=0.6)
        ax_probs.set_xlabel("Subtype probability (%)", fontsize=8)
        ax_probs.set_title(
            "Subtype-classifier output (separate from abnormal probability)",
            fontsize=7.2,
            loc="left",
            color="#475569",
        )
        ax_probs.set_xlim(0, 105)
        ax_probs.tick_params(axis="both", labelsize=7)
        ax_probs.spines["top"].set_visible(False)
        ax_probs.spines["right"].set_visible(False)
        ax_probs.invert_yaxis()
        for bar, value in zip(bars, values):
            ax_probs.text(
                min(bar.get_width() + 1, 101),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=7,
                color="#475569",
            )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_session_report(session_id: str,
                            physician_name: Optional[str] = None,
                            notes: Optional[str] = None) -> Path:
    """Build the PDF and return the file path."""
    with get_db() as db:
        sess = db.execute(text("""
            SELECT s.id, s.userId, s.modelId, s.sourceType, s.sourceDetail,
                   s.startedAt, s.endedAt, s.status, s.totalBeats,
                   s.anomalyBeats, s.summaryJson,
                   u.email, u.name,
                   m.name, m.version, m.architecture, m.threshold
            FROM EcgSession s
            JOIN User u ON u.id = s.userId
            LEFT JOIN ModelVersion m ON m.id = s.modelId
            WHERE s.id = :id
        """), {"id": session_id}).fetchone()
        if not sess:
            raise ValueError(f"Session {session_id} not found")

        alerts = db.execute(text("""
            SELECT timestamp, anomalyScore, threshold, severity, message, contextJson
            FROM Alert WHERE sessionId = :id ORDER BY timestamp
        """), {"id": session_id}).fetchall()

        # Sample data points (cap at 4000 for the plot)
        pts = db.execute(text("""
            SELECT t, value, prediction, anomalyScore, isAnomaly
            FROM EcgDataPoint WHERE sessionId = :id
            ORDER BY t LIMIT 4000
        """), {"id": session_id}).fetchall()

    points = [{
        "t": p[0], "value": p[1], "prediction": p[2],
        "anomaly_score": p[3], "is_anomaly": bool(p[4]),
    } for p in pts]
    alert_list = []
    for a in alerts:
        ctx = json.loads(a[5] or "{}")
        classification = ctx.get("classification")
        alert_list.append({
            "timestamp": a[0], "anomaly_score": a[1], "threshold": a[2],
            "severity": a[3], "message": a[4], "context": ctx,
            "classification": classification,
        })
    # Historical databases can contain repeated Alert rows for the same
    # R-peak. Every downstream section uses this canonical one-beat list.
    alert_list = _deduplicate_alerts(alert_list)
    model_output_rows = _model_output_summary(alert_list)
    arrhythmia_rows, main_arrhythmia = _arrhythmia_summary(alert_list)

    # ---------- Render ECG plot ----------
    plot_path = settings.REPORT_DIR / f"session_{session_id}_plot.png"
    _plot_ecg(points, alert_list, plot_path)

    # ---------- Build PDF ----------
    out_path = settings.REPORT_DIR / f"session_{session_id}_report.pdf"
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"ECG Session Report — {session_id[:8]}",
        author="ECG Anomaly Detection System",
        subject="Physician review report",
    )
    s = _styles()
    story = []

    # ---- Header ----
    story.append(Paragraph("ECG Anomaly Detection — Session Report", s["title"]))
    story.append(Paragraph(
        f"Prepared for: {physician_name or 'Attending Physician'}  •  "
        f"Session ID: {session_id[:8]}  •  Generated: {sess[5]}",
        s["subtitle"],
    ))

    # ---- Session info table ----
    story.append(Paragraph("Session Information", s["h2"]))
    duration = "—"
    summary = json.loads(sess[10] or "{}")
    analysis_mode = summary.get("analysis_mode", "sliding-reconstruction")
    improved_mode = analysis_mode == "beat-aligned-hierarchical"
    count_unit = summary.get("count_unit", "beats")
    decision_threshold = summary.get("decision_threshold", sess[16])
    if summary.get("duration_s"):
        duration = f"{summary['duration_s']:.1f} s"
    elif sess[5] and sess[6]:
        # SQLite returns string timestamps; compute difference in Python
        from datetime import datetime
        try:
            fmt = "%Y-%m-%d %H:%M:%S"
            t0 = datetime.strptime(sess[5], fmt)
            t1 = datetime.strptime(sess[6], fmt)
            duration = f"{(t1 - t0).total_seconds():.1f} s"
        except Exception:
            pass

    info_data = [
        ["Patient / User", sess[12] or sess[11]],
        ["Source", f"{sess[3]} ({sess[4] or '—'})"],
        ["Started", str(sess[5])],
        ["Ended", str(sess[6] or '—')],
        ["Duration", duration],
        ["Status", sess[7]],
        ["Total beats analyzed", f"{sess[8]:,}"],
        ["Anomalous beats", f"{sess[9]:,}  ({(sess[9] / max(1, sess[8]) * 100):.2f}%)"],
        ["Unique alerted beats in report", f"{len(alert_list):,}"],
        ["Model", Paragraph(
            _pdf_text(
                f"{sess[13] or '—'} ({sess[14] or '—'}) • "
                f"{sess[15] or '—'}"
            ),
            s["small"],
        )],
        ["Analysis mode", (
            "Beat-aligned hierarchical morphology + RR"
            if improved_mode else "Sliding autoencoder reconstruction"
        )],
        [
            (
                "Abnormal-probability threshold"
                if improved_mode else "Detection threshold (MAE)"
            ),
            f"{decision_threshold:.4f}" if decision_threshold is not None else "—",
        ],
    ]
    info_tbl = Table(info_data, colWidths=[55 * mm, 110 * mm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0f172a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]))
    story.append(info_tbl)

    # ---- Research validation context ----
    _append_validation_context(story, s, session_id)

    # ---- ECG plot ----
    story.append(Paragraph("ECG Recording & Detection Results", s["h2"]))
    plot_explanation = (
        "The following plot shows the recorded ECG signal (z-score normalized). "
        "Red markers indicate R-peak-aligned beats whose supervised abnormal "
        "probability exceeded the validation-selected threshold."
        if improved_mode else
        "The following plot shows the recorded ECG signal (z-score normalized). "
        "Red-shaded regions indicate samples flagged by autoencoder "
        "reconstruction error."
    )
    story.append(Paragraph(plot_explanation, s["body"]))
    story.append(Spacer(1, 4 * mm))
    img = RLImage(str(plot_path), width=170 * mm, height=54 * mm)
    story.append(img)

    # ---- Main arrhythmia interpretation ----
    story.append(Paragraph("Main Arrhythmia Finding", s["h2"]))
    if main_arrhythmia:
        example = main_arrhythmia["example"]
        cls = example["classification"]
        info = get_health_info(main_arrhythmia["class"])
        story.append(Paragraph(
            f"<b>{_pdf_text(main_arrhythmia['class_name'])}</b> was the main "
            f"high-confidence arrhythmia type detected in this session. It appeared "
            f"{main_arrhythmia['count']} time(s) with average confidence "
            f"{main_arrhythmia['avg_confidence']:.1%} and maximum confidence "
            f"{main_arrhythmia['max_confidence']:.1%}.",
            s["body"],
        ))
        reason_text = (
            "Reason: the beat-aligned detector combined morphology, spectrum and "
            "RR timing, and its abnormal probability exceeded the saved threshold. "
            "The subtype model then assigned this label"
            if improved_mode else
            "Reason: the autoencoder first flagged the ECG segment by reconstruction "
            "error, and the transferred classifier then assigned this label"
        )
        story.append(Paragraph(
            f"{reason_text} with at least {CONFIDENCE_THRESHOLD:.0%} confidence. "
            "Lower-confidence labels are excluded to avoid over-interpreting "
            "uncertain model output.",
            s["body"],
        ))
        story.append(Paragraph(
            f"<b>Example:</b> time {example['timestamp']}, "
            f"{'abnormal probability' if improved_mode else 'anomaly score'} "
            f"{example['anomaly_score']:.4f}, threshold {example['threshold']:.4f}, "
            f"classifier confidence {float(cls.get('confidence') or 0):.1%}.",
            s["body"],
        ))
        story.append(Paragraph(
            f"<b>Clinical meaning:</b> {_pdf_text(info.get('what_is_it', ''))}",
            s["body"],
        ))
        story.append(Paragraph(
            f"<b>Why it matters:</b> {_pdf_text(info.get('is_dangerous', ''))}",
            s["body"],
        ))
    elif alert_list:
        story.append(Paragraph(
            "Anomalous ECG regions were detected, but no arrhythmia classification "
            f"reached the {CONFIDENCE_THRESHOLD:.0%} confidence threshold. The report "
            "therefore does not name a main arrhythmia type; the detailed waveform "
            "examples below should be reviewed manually.",
            s["body"],
        ))
    else:
        story.append(Paragraph(
            (
                "No alerts were triggered during this session. No analysed beat "
                "exceeded the abnormal-probability threshold."
                if improved_mode else
                "No alerts were triggered during this session. The ECG signal was "
                "classified as within normal limits by the autoencoder model."
            ),
            s["body"],
        ))

    if model_output_rows:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("Model Output Summary", s["h2"]))
        story.append(Paragraph(
            "Each row is one distinct model output. Beat counts are calculated "
            "after repeated R-peak alerts are removed. The binary decision answers "
            "\"is this beat abnormal?\" The subtype confidence separately describes "
            "the predicted abnormal type.",
            s["body"],
        ))
        story.append(Spacer(1, 2 * mm))
        cls_rows = [[
            "Model output",
            "Beats",
            "Binary decision",
            "Subtype conf.",
            "RR",
            "Input",
        ]]
        for row in model_output_rows:
            example = row["example"]
            cls = _alert_classification(example) or {}
            ctx = example.get("context", {}) or {}
            probability = float(example.get("anomaly_score") or 0.0)
            threshold = float(example.get("threshold") or 0.0)
            if ctx.get("score_type") == "abnormal_probability":
                decision = f"prob. {probability:.1%} > thr. {threshold:.1%}"
            else:
                decision = f"error {probability:.4f} > thr. {threshold:.4f}"
            rr_seconds = ctx.get("rr_seconds")
            input_mode = (
                ctx.get("classification_mode")
                or cls.get("mode")
                or "not recorded"
            )
            cls_rows.append([
                Paragraph(_pdf_text(row["class_name"]), s["small"]),
                str(row["count"]),
                decision,
                f"{float(cls.get('confidence') or 0.0):.1%}",
                f"{float(rr_seconds):.3f} s" if rr_seconds is not None else "—",
                Paragraph(_pdf_text(input_mode), s["small"]),
            ])
        cls_tbl = Table(
            cls_rows,
            colWidths=[43 * mm, 18 * mm, 40 * mm, 27 * mm, 18 * mm, 24 * mm],
            repeatRows=1,
        )
        cls_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#fafbfc")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(cls_tbl)

    # ---- Detected Anomaly Waveforms ----
    # Show one representative waveform per distinct model output. Repeated
    # beats remain counted in the summary but do not create duplicate panels.
    if alert_list:
        top_alerts = [
            row["example"]
            for row in model_output_rows
            if (row["example"].get("context", {}) or {}).get("signal")
        ][:6]

        if top_alerts:
            story.append(Paragraph(
                "Unique Anomaly Waveforms and Model Outputs", s["h2"]
            ))
            waveform_explanation = (
                "Each panel shows exactly one representative beat for a distinct "
                "model output. The red band marks the detected R peak. The title "
                "shows abnormal probability, decision threshold, decision margin, "
                "subtype label, and subtype confidence. The lower bars show subtype "
                "probabilities. Repeated beats of the same output are counted in "
                "the summary table and are not plotted again."
                if improved_mode else
                "Each panel shows one representative segment for a distinct model "
                "output, together with reconstruction error, threshold, label, and "
                "classifier probabilities. Repeated outputs are not plotted."
            )
            story.append(Paragraph(waveform_explanation, s["body"]))
            story.append(Spacer(1, 4 * mm))

            # Plot each anomaly waveform and arrange 2 per row in a table
            for i, alert in enumerate(top_alerts, start=1):
                plot_path = settings.REPORT_DIR / f"session_{session_id}_anomaly_{i}.png"
                _plot_anomaly_waveform(alert, plot_path, i)

            # Build a 2-column table with the images
            rows = []
            for i in range(0, len(top_alerts), 2):
                left_img = RLImage(
                    str(settings.REPORT_DIR / f"session_{session_id}_anomaly_{i+1}.png"),
                    width=82 * mm, height=55 * mm,
                )
                if i + 1 < len(top_alerts):
                    right_img = RLImage(
                        str(settings.REPORT_DIR / f"session_{session_id}_anomaly_{i+2}.png"),
                        width=82 * mm, height=55 * mm,
                    )
                else:
                    right_img = ""
                rows.append([left_img, right_img])

            waveforms_tbl = Table(rows, colWidths=[85 * mm, 85 * mm])
            waveforms_tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(waveforms_tbl)
            story.append(Spacer(1, 4 * mm))

    # ---- Clinical summary ----
    story.append(Paragraph("Clinical Summary", s["h2"]))
    anomaly_pct = (sess[9] / max(1, sess[8]) * 100) if sess[8] else 0
    if anomaly_pct < 1:
        assessment = (
            f"Fewer than 1% of analysed {count_unit} exceeded the model threshold. "
            "This research result does not exclude arrhythmia."
        )
    elif anomaly_pct < 10:
        assessment = (
            f"Approximately {anomaly_pct:.1f}% of analysed {count_unit} were "
            "flagged. The waveform and recording quality should be reviewed; the "
            "model output alone is not a diagnosis."
        )
    else:
        assessment = (
            f"{anomaly_pct:.1f}% of analysed {count_unit} were flagged. This may "
            "reflect abnormal morphology, rhythm, noise, or domain shift and "
            "requires qualified review."
        )
    if arrhythmia_rows:
        detected_names = ", ".join(row["class_name"] for row in arrhythmia_rows[:4])
        assessment += (
            f" High-confidence classifier findings in this report include: "
            f"{detected_names}. Only arrhythmia labels with confidence above "
            f"{CONFIDENCE_THRESHOLD:.0%} are included."
        )
    story.append(Paragraph(assessment, s["body"]))

    # ---- Physician notes ----
    story.append(Paragraph("Physician Notes", s["h2"]))
    if notes:
        story.append(Paragraph(notes, s["body"]))
    story.append(Spacer(1, 6 * mm))
    notes_tbl = Table([[""]] * 6, colWidths=[170 * mm], rowHeights=[8 * mm] * 6)
    notes_tbl.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
    ]))
    story.append(notes_tbl)
    story.append(Spacer(1, 6 * mm))

    # ---- Sign-off ----
    signoff = Table([
        ["Physician signature: ____________________",
         "Date: ____________________"],
    ], colWidths=[100 * mm, 65 * mm])
    signoff.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#475569")),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(signoff)

    # ---- Disclaimer ----
    # Keep enough room for the complete disclaimer on the sign-off page.
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This report is generated automatically by a research ECG anomaly-detection "
        "system and is intended for research review only. Model probability and "
        "subtype confidence are not clinical certainty. This report does not "
        "constitute a medical diagnosis and must not replace qualified clinical "
        "judgment.",
        s["small"],
    ))

    doc.build(story)
    return out_path
