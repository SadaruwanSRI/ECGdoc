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

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from app.core.config import settings
from app.db.session import get_db
from app.ml.health_info import get_health_info
from sqlalchemy import text


PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONFIDENCE_THRESHOLD = 0.50


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


def _alert_classification(alert: dict) -> Optional[dict]:
    """Return classification stored on an alert or inside its context."""
    cls = alert.get("classification")
    if cls:
        return cls
    ctx = alert.get("context", {}) or {}
    return ctx.get("classification")


def _high_confidence_alerts(alerts: list[dict],
                            min_confidence: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    selected = []
    for alert in alerts:
        cls = _alert_classification(alert)
        if not cls:
            continue
        confidence = float(cls.get("confidence") or 0.0)
        arrhythmia_class = cls.get("class")
        if confidence >= min_confidence and arrhythmia_class and arrhythmia_class != "N":
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
    """Plot a single anomaly waveform with classification results.

    Shows:
    - ECG signal (red) + reconstruction (blue dashed) + error (shaded)
    - Classification label + confidence in the title
    - Probability bar chart below the waveform
    - Diagnosis text
    """
    ctx = alert.get("context", {}) or {}
    signal = ctx.get("signal", [])
    recon = ctx.get("reconstruction", [])
    fs = ctx.get("fs", 64)
    classification = alert.get("classification")

    if not signal:
        fig, ax = plt.subplots(figsize=(5.2, 2.0), constrained_layout=True)
        ax.text(0.5, 0.5, "Waveform not available",
                ha="center", va="center", fontsize=10, color="#94a3b8")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    t = [i / fs for i in range(len(signal))]

    # Determine figure height based on whether we have classification
    has_clf = classification is not None
    fig_height = 3.2 if has_clf else 2.0

    if has_clf:
        fig, (ax_wave, ax_probs) = plt.subplots(
            2, 1, figsize=(5.2, fig_height),
            gridspec_kw={'height_ratios': [3, 1.2]},
            constrained_layout=True
        )
    else:
        fig, ax_wave = plt.subplots(figsize=(5.2, fig_height), constrained_layout=True)

    # --- Waveform plot ---
    ax_wave.plot(t, signal, color="#dc2626", linewidth=1.0, label="ECG (anomalous)")
    if recon:
        ax_wave.plot(t, recon, color="#0891b2", linewidth=0.9, linestyle="--",
                     label="Reconstruction")
    ax_wave.fill_between(t, signal, recon, color="#fecaca", alpha=0.4,
                         label="Reconstruction error")

    severity = alert.get("severity", "warning")
    score = alert.get("anomaly_score", 0)

    # Title with classification info
    if has_clf:
        cls_name = classification.get("class_name", "Unknown")
        confidence = classification.get("confidence", 0)
        ax_wave.set_title(
            f"#{idx}  •  {severity.upper()}  •  {cls_name} ({confidence:.1%})\n"
            f"Reconstruction error: {score:.4f}",
            fontsize=9, color="#0f172a", loc="left", pad=4
        )
    else:
        ax_wave.set_title(
            f"#{idx}  •  {severity.upper()}  •  score={score:.4f}",
            fontsize=9, color="#0f172a", loc="left", pad=4
        )

    ax_wave.set_xlabel("Time (s)", fontsize=8)
    ax_wave.set_ylabel("Amplitude", fontsize=8)
    ax_wave.grid(True, alpha=0.3, linewidth=0.4)
    ax_wave.tick_params(axis="both", labelsize=7)
    ax_wave.spines["top"].set_visible(False)
    ax_wave.spines["right"].set_visible(False)
    ax_wave.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # --- Probability bar chart (only if classification available) ---
    if has_clf:
        probs = classification.get("probabilities", {})
        top_class = classification.get("class", "")
        classes = sorted(probs.keys(), key=lambda k: probs[k], reverse=True)
        values = [probs[c] * 100 for c in classes]
        colors = ["#dc2626" if c == top_class else "#94a3b8" for c in classes]

        bars = ax_probs.barh(classes, values, color=colors, height=0.6)
        ax_probs.set_xlabel("Confidence (%)", fontsize=8)
        ax_probs.set_xlim(0, 100)
        ax_probs.tick_params(axis="both", labelsize=7)
        ax_probs.spines["top"].set_visible(False)
        ax_probs.spines["right"].set_visible(False)
        ax_probs.invert_yaxis()

        # Add percentage labels on bars
        for bar, val in zip(bars, values):
            ax_probs.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                          f"{val:.1f}%", va="center", fontsize=7, color="#475569")

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
        ["Model", f"{sess[13] or '—'} ({sess[14] or '—'})  •  {sess[15] or '—'}"],
        ["Detection threshold (MAE)", f"{sess[16]:.4f}" if sess[16] else "—"],
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

    # ---- ECG plot ----
    story.append(Paragraph("ECG Recording & Detection Results", s["h2"]))
    story.append(Paragraph(
        "The following plot shows the recorded ECG signal (z-score normalized). "
        "Red-shaded regions indicate samples flagged as anomalous by the autoencoder "
        "(reconstruction error exceeding the trained threshold).",
        s["body"],
    ))
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
        story.append(Paragraph(
            "Reason: this type is shown because the autoencoder first flagged the "
            "ECG segment as anomalous by reconstruction error, and the classifier "
            f"then assigned this arrhythmia label with at least {CONFIDENCE_THRESHOLD:.0%} "
            "confidence. Lower-confidence labels are excluded from this report to "
            "avoid over-interpreting uncertain model output.",
            s["body"],
        ))
        story.append(Paragraph(
            f"<b>Example:</b> time {example['timestamp']}, anomaly score "
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
            "No alerts were triggered during this session. The ECG signal was "
            "classified as within normal limits by the autoencoder model.",
            s["body"],
        ))

    if arrhythmia_rows:
        story.append(Spacer(1, 2 * mm))
        cls_rows = [["Arrhythmia Type", "Detections", "Avg Confidence", "Example"]]
        for row in arrhythmia_rows:
            example = row["example"]
            cls_rows.append([
                row["class_name"],
                str(row["count"]),
                f"{row['avg_confidence']:.1%}",
                f"score {example['anomaly_score']:.4f}, conf {row['max_confidence']:.1%}",
            ])
        cls_tbl = Table(cls_rows, colWidths=[65 * mm, 25 * mm, 35 * mm, 45 * mm])
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
    # Show the top 6 most significant alerts (by anomaly score) with their
    # actual ECG waveform + reconstruction, so the physician can visually
    # inspect what the model flagged as abnormal.
    if alert_list:
        # Prefer high-confidence classifier examples, then fall back to the
        # largest reconstruction errors if no confident class label is present.
        alerts_with_signal = [a for a in _high_confidence_alerts(alert_list)
                              if (a.get("context", {}) or {}).get("signal")]
        if alerts_with_signal:
            alerts_with_signal.sort(
                key=lambda a: (
                    float((_alert_classification(a) or {}).get("confidence") or 0.0),
                    a.get("anomaly_score", 0),
                ),
                reverse=True,
            )
        else:
            alerts_with_signal = [a for a in alert_list
                                  if (a.get("context", {}) or {}).get("signal")]
            alerts_with_signal.sort(key=lambda a: a.get("anomaly_score", 0), reverse=True)
        top_alerts = alerts_with_signal[:6]

        if top_alerts:
            story.append(Paragraph("Detected Anomaly Waveforms", s["h2"]))
            story.append(Paragraph(
                "The following plots show example ECG segments (red) that "
                "triggered alerts, alongside the "
                "autoencoder's reconstruction of what it expected to see "
                "(blue dashed). The shaded region between the two curves "
                "represents the reconstruction error that exceeded the "
                "detection threshold. When available, examples are selected "
                f"from arrhythmia classifications above {CONFIDENCE_THRESHOLD:.0%} confidence.",
                s["body"],
            ))
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
                    width=82 * mm, height=32 * mm,
                )
                if i + 1 < len(top_alerts):
                    right_img = RLImage(
                        str(settings.REPORT_DIR / f"session_{session_id}_anomaly_{i+2}.png"),
                        width=82 * mm, height=32 * mm,
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
        assessment = ("The recorded ECG segment was predominantly within normal "
                      "limits, with fewer than 1% of analyzed samples exceeding the "
                      "anomaly threshold. No clinically significant arrhythmias were "
                      "detected by the model during this session.")
    elif anomaly_pct < 10:
        assessment = (f"Approximately {anomaly_pct:.1f}% of analyzed samples were "
                      "flagged as anomalous. This may indicate intermittent ectopic "
                      "activity or transient arrhythmias. Correlation with clinical "
                      "symptoms and a 12-lead ECG is recommended.")
    else:
        assessment = (f"{anomaly_pct:.1f}% of analyzed samples were flagged as "
                      "anomalous, suggesting sustained abnormal ECG morphology. "
                      "Urgent cardiology review is recommended.")
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
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "This report is generated automatically by an unsupervised deep-learning "
        "anomaly detection system and is intended for decision support only. It does "
        "not constitute a medical diagnosis and should not replace clinical judgment "
        "by a qualified physician.",
        s["small"],
    ))

    doc.build(story)
    return out_path
