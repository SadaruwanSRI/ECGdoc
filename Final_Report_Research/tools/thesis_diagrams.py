"""Vector diagrams for the thesis, using the corrected experiment as authority.

Run from the report directory: python tools/thesis_diagrams.py
Coordinates are in inches; font sizes are chosen for a full-width thesis figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

REPORT = Path(__file__).resolve().parents[1]
OUTPUT = REPORT / "pic" / "generated"
RESULT = REPORT / "experiments" / "corrected_temporal_holdout.json"
NAVY, TEAL, PURPLE = "#233e56", "#176b68", "#66508a"
RUST, GREY, INK = "#9e482c", "#65717b", "#17232d"
LIGHT, LINE = "#f4f7f9", "#c9d2d9"


def corrected_result():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    b, s, w = (result[k] for k in ("test_binary",
        "test_subtype_on_true_supported_arrhythmias", "test_whole_system"))
    assert b["normal_support"] + b["abnormal_support"] == b["support"]
    assert b["normal_support"] + s["support"] == w["support"]
    assert sum(a["temporal_test_beats"] for a in result["protocol"]["split_audit"].values()) == b["support"]
    return result


class Diagram:
    def __init__(self, height):
        plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                             "ps.fonttype": 42})
        self.width, self.height = 7.2, height
        self.fig = plt.figure(figsize=(self.width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, self.width), ylim=(0, height))
        self.ax.axis("off")
        self.texts, self.contained = [], []

    def text(self, x, y, label, size=10.5, color=INK, weight="normal",
             ha="left", va="top", **kwargs):
        t = self.ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                         ha=ha, va=va, linespacing=1.25, **kwargs)
        self.texts.append(t)
        return t

    def heading(self, x, y, label):
        return self.text(x, y, label, size=11.5, color=NAVY, weight="bold")

    def rect(self, x, y, w, h, color=LINE, fill=LIGHT, **kwargs):
        patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=color,
                          linewidth=1.0, **kwargs)
        self.ax.add_patch(patch)
        return patch

    def card(self, x, y, w, h, title, body, color=NAVY, fs=10.5):
        patch = self.rect(x, y, w, h, color=color, fill="white")
        title_text = self.text(x + .13, y + h - .13, title, size=11,
                               color=color, weight="bold")
        body_text = self.text(x + .13, y + h - .43, body, size=fs)
        self.contained.extend([(title_text, patch), (body_text, patch)])

    def node(self, x, y, w, h, label, color=NAVY, fill=LIGHT, fs=10.5):
        patch = self.rect(x, y, w, h, color=color, fill=fill)
        t = self.text(x + w / 2, y + h / 2, label, size=fs, ha="center",
                      va="center", color=INK)
        self.contained.append((t, patch))

    def arrow(self, start, end, color=GREY, dashed=False, shrink_start=3):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
            mutation_scale=11, linewidth=1.15, color=color,
            linestyle="--" if dashed else "-", shrinkA=shrink_start, shrinkB=3))

    def elbow(self, points, color=GREY):
        self.ax.plot(*zip(*points[:-1]), color=color, lw=1.15)
        self.arrow(points[-2], points[-1], color, shrink_start=0)

    def save(self, name):
        # Catch the original failure mode before publishing any diagram.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        boxes = [(t, t.get_window_extent(renderer)) for t in self.texts]
        for t, box in boxes:
            if not self.fig.bbox.contains(box.x0, box.y0) or not self.fig.bbox.contains(box.x1, box.y1):
                raise ValueError(f"{name}: text outside canvas: {t.get_text()}")
        for i, (t, box) in enumerate(boxes):
            for other, obox in boxes[i + 1:]:
                if box.overlaps(obox):
                    raise ValueError(f"{name}: overlapping text: {t.get_text()!r} / {other.get_text()!r}")
        for t, patch in self.contained:
            text_box = t.get_window_extent(renderer)
            card_box = patch.get_window_extent(renderer)
            if not card_box.contains(text_box.x0, text_box.y0) or not card_box.contains(text_box.x1, text_box.y1):
                raise ValueError(f"{name}: text exceeds card: {t.get_text()}")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUTPUT / name, metadata={"Title": name.removesuffix(".pdf")})
        plt.close(self.fig)


def lead_positions():
    d = Diagram(5.35)
    d.heading(.2, 5.16, "A  MLII polarity and torso placement")
    # An anterior-view schematic, not a claim about exact database coordinates.
    outline = [(3.23,4.85),(3.23,4.63),(2.65,4.51),(2.50,4.30),
               (2.62,3.70),(2.78,3.42),(2.88,2.43),(4.32,2.43),
               (4.42,3.42),(4.58,3.70),(4.70,4.30),(4.55,4.51),
               (3.97,4.63),(3.97,4.85)]
    d.ax.add_patch(PathPatch(MplPath(outline), facecolor=LIGHT, edgecolor=LINE, lw=1.5))
    negative, positive, reference = (3.02,4.35), (4.16,2.75), (3.00,2.75)
    for point, color in [(negative,NAVY),(positive,RUST),(reference,GREY)]:
        d.ax.add_patch(Circle(point, .072, facecolor="white", edgecolor=color, lw=1.8))
    d.arrow((3.14,4.20), (4.04,2.91), TEAL)
    d.text(4.02,4.04,"MLII\naxis",ha="center",color=TEAL,size=10)
    d.text(.24,4.56,"Negative (-)\nRA-equivalent location\nRight upper torso",size=10)
    d.elbow([(2.00,4.12),(2.32,4.12),negative],NAVY)
    d.text(4.91,3.52,"Positive (+)\nLL-equivalent location\nLeft lower torso",size=10)
    d.elbow([(4.92,2.99),(4.62,2.99),positive],RUST)
    d.text(.24,3.05,"Reference (RL)\nSensor reference",size=10)
    d.elbow([(2.04,2.75),(2.45,2.75),reference],GREY)
    d.text(3.60,2.23,"Front view: patient's right is on the left of the drawing.",ha="center",size=9.5,color=GREY)
    d.heading(.2,1.88,"B  One channel per input; channel identities differ")
    d.card(.2,.31,3.27,1.28,"Normal autoencoder: NSRDB",
           "First stored channel: ECG1\nAnatomical placement is unspecified.\nUsed to learn a normal ECG reference.",TEAL,10)
    d.card(3.73,.31,3.27,1.28,"Supervised models: MITDB",
           "Select MLII by its header name.\nRecord 114: MLII is the second channel.\nExclude 102 and 104: neither has MLII.",NAVY,10)
    d.save("real_lead_positions.pdf")


def data_preparation():
    normal = json.loads((REPORT / "experiments/nsrdb_autoencoder_training.json").read_text())
    result = corrected_result()
    d = Diagram(6.0)
    d.heading(.2,5.81,"A  Normal-reference data")
    d.heading(3.88,5.81,"B  Labelled MLII beat data")
    d.card(.2,4.47,3.12,1.02,"NSRDB: ECG1",
           f"{normal['healthy_subjects_total']} complete recordings\nUse the first stored channel only.",TEAL,10)
    d.card(3.88,4.47,3.12,1.02,"MITDB: MLII",
           f"{result['protocol']['records']} eligible records\nResample to 128 Hz; locate R peaks.",NAVY,10)
    d.arrow((1.76,4.47),(1.76,4.19),TEAL)
    d.arrow((5.44,4.47),(5.44,4.19),NAVY)
    d.card(.2,2.85,3.12,1.34,"Clean waveform windows",
           f"Non-overlapping 4-second windows\n512 samples per window\n{normal['training_windows']:,} fitting windows\n{normal['validation_windows']:,} validation windows",TEAL,10)
    d.card(3.88,2.85,3.12,1.34,"Prepare one accepted beat",
           "Extract 512 samples around the R peak.\nFilter and z-score within that window.\nMeasure previous and next RR intervals\nand earlier rhythm history.",NAVY,9.7)
    d.arrow((1.76,2.85),(1.76,2.56),TEAL)
    d.arrow((5.44,2.85),(5.44,2.56),NAVY)
    d.card(.2,1.17,3.12,1.39,"Train, validate, then freeze",
           f"{normal['training_subjects']} subjects fit the autoencoder.\n{normal['validation_subjects']} other subjects select its checkpoint.\nApply it to each MLII beat to obtain\n4 reconstruction-error features.",TEAL,10)
    d.card(3.88,1.17,3.12,1.39,"One feature vector per beat",
           "223 waveform measurements\n+ 15 RR timing measurements\n+ 4 autoencoder error measurements\n= 242 values for both tree models",PURPLE,10)
    d.arrow((3.32,1.79),(3.88,1.79),TEAL)
    d.text(.2,.76,"Expert labels are stored separately from the 242 input features.",weight="bold",size=10.5)
    d.text(.2,.44,"Earlier MITDB beats develop the models; later beats evaluate their predictions.\nThe first and last eligible beat are excluded when a neighbouring RR is unavailable.",size=10,color=GREY)
    d.save("real_data_preparation_flow.pdf")


def training_lifecycle(result=None):
    r = result or corrected_result()
    threshold = r["selected_binary"]["threshold_from_64_to_80_validation"]
    d = Diagram(5.35)
    d.heading(.2,5.15,"A  Offline training produces three fixed components")
    d.card(.2,3.51,3.27,1.32,"Normal autoencoder",
           "Train on NSRDB ECG1.\nSelect the checkpoint on other subjects.\nFreeze its weights before MLII modelling.",TEAL,10)
    d.card(3.73,3.51,3.27,1.32,"Two Extra Trees models",
           "Use the same 242 MLII features.\nSelect on earlier validation beats.\nRefit the binary gate and subtype model.",NAVY,10)
    d.elbow([(1.835,3.51),(1.835,3.25),(3.6,3.25),(3.6,3.03)],TEAL)
    d.elbow([(5.365,3.51),(5.365,3.25),(3.6,3.25),(3.6,3.03)],NAVY)
    d.card(.9,2.02,5.4,1.01,"Saved models and decision threshold",
           f"Frozen autoencoder + binary gate + subtype classifier\nValidation-selected abnormality threshold: {threshold:.6f}",PURPLE,10.5)
    d.heading(.2,1.75,"B  Frozen model use")
    d.elbow([(3.6,2.02),(3.6,1.40),(1.835,1.40),(1.835,1.18)],GREY)
    d.elbow([(3.6,2.02),(3.6,1.40),(5.365,1.40),(5.365,1.18)],GREY)
    d.card(.2,.14,3.27,1.04,"Retrospective temporal test",
           "Later MITDB beats from known patients\nCompare predictions with expert labels.",RUST,10)
    d.card(3.73,.14,3.27,1.04,"Session prediction",
           "Incoming MLII signal or database replay\nReturn N or one of five arrhythmia groups.",NAVY,10)
    d.save("architecture_training_live_map.pdf")


def autoencoder_architecture():
    verified = json.loads((REPORT / "experiments/final_verification.json").read_text())
    shapes = verified["actual_tensor_shapes"]
    d = Diagram(6.35)
    d.heading(.25,6.17,"Encoder: shorten the sequence")
    d.heading(4.48,6.17,"Decoder: reconstruct the ECG")
    d.node(.3,5.32,2.2,.55,"Input window\n1 x 512",NAVY)
    d.node(4.7,5.27,2.2,.65,"Linear output: 1 x 256\nResize to 1 x 512",TEAL)
    ys = [4.42,3.62,2.82,2.02]
    for i,y in enumerate(ys,1):
        enc = shapes[f"enc{i}"][1:]
        up = shapes[f"up{5-i}"][1:]
        d.node(.3,y,2.2,.58,f"Encoder {i}: residual conv\n{enc[0]} x {enc[1]}",NAVY,fs=10.5)
        d.node(4.7,y,2.2,.58,f"Decoder {5-i}: up block\n{up[0]} x {up[1]}",TEAL,fs=10.5)
        d.arrow((2.5,y+.29),(4.7,y+.29),TEAL,dashed=True)
        d.text(3.6,y+.29,"skip x 0.5",ha="center",va="center",size=9.5,color=TEAL,
               bbox={"facecolor":"white","edgecolor":"none","pad":2})
    d.arrow((1.4,5.32),(1.4,5.00),NAVY)
    d.arrow((5.8,5.00),(5.8,5.27),TEAL)
    for upper,lower in zip(ys,ys[1:]):
        d.arrow((1.4,upper),(1.4,lower+.58),NAVY)
        d.arrow((5.8,lower+.58),(5.8,upper),TEAL)
    d.node(2.35,.95,2.5,.68,"Dilated bottleneck: 256 x 32\nDilation rates: 1, 2, 4, 8",PURPLE,fs=10.5)
    d.elbow([(1.4,2.02),(1.4,1.29),(2.35,1.29)],NAVY)
    d.elbow([(4.85,1.29),(5.8,1.29),(5.8,2.02)],TEAL)
    d.text(.25,.63,"Dimensions are channels x time samples; the batch dimension is omitted.",size=10)
    d.text(.25,.37,"Dashed arrows carry encoder features to matching decoder lengths.\nEach up block resizes to its skip length; final interpolation restores 512 samples.",size=9.7,color=GREY)
    d.save("architecture_autoencoder_layers.pdf")


def method_workflow(result=None):
    r = result or corrected_result()
    d = Diagram(4.8)
    d.heading(.2,4.60,"Chronological split within each MITDB record")
    d.text(.2,4.29,"Percentages refer to accepted-beat order, not elapsed recording time.",size=10.2,color=GREY)
    x0, width, bh = .35, 6.5, .61
    cut64, cut80 = x0 + width*.64, x0 + width*.8
    d.heading(.2,3.90,"A  Select model settings")
    y=3.00
    d.node(x0,y,width*.64,bh,"Fit candidate models\nFirst 64%",NAVY,fill="#eef3f7",fs=11)
    d.node(cut64,y,width*.16,bh,"Validate\n64-80%",TEAL,fill="#e9f3f1",fs=10)
    d.node(cut80,y,width*.2,bh,"Set aside\nFinal 20%",GREY,fill="#f5f5f5",fs=10)
    for boundary in (cut64,cut80):
        d.rect(boundary-.065,y,.065,bh,color=GREY,fill="white",hatch="////",zorder=3)
    d.text(.35,2.83,"Validation chooses the model candidates and the abnormality threshold.",size=10.1)
    d.heading(.2,2.40,"B  Refit, freeze, then evaluate")
    y=1.51
    d.node(x0,y,width*.8,bh,
           f"Refit selected models on the earlier 80% (minus the test gap)\nBinary: {r['dataset_counts']['development']:,} beats; subtype: {r['dataset_counts']['subtype_development']:,} abnormal beats",TEAL,fill="#e9f3f1",fs=10.5)
    d.node(cut80,y,width*.2,bh,f"Final test\n{r['test_binary']['support']:,} beats",RUST,fill="#faf0eb",fs=10)
    d.rect(cut80-.065,y,.065,bh,color=GREY,fill="white",hatch="////",zorder=3)
    d.text(.35,1.33,"The model and threshold remain fixed while scoring the final test segment.",size=10.1)
    d.text(.2,.93,"Hatched markers: 16 beats excluded immediately before each marked boundary.\nMarkers are enlarged for visibility. The 64% gap is restored during final refitting.",size=9.8,color=GREY)
    d.text(.2,.39,"Interpretation: later beats from known patients; the test had been seen in earlier\nproject development, so this is a retrospective evaluation.",size=9.8,color=RUST)
    d.fig.set_size_inches(7.2, 4.2)
    d.save("method_final_workflow.pdf")


def testing_protocol(result=None):
    r = result or corrected_result()
    binary, subtype, whole = (r[k] for k in ("test_binary",
        "test_subtype_on_true_supported_arrhythmias","test_whole_system"))
    other = binary["support"] - whole["support"]
    d = Diagram(6.1)
    d.heading(.2,5.96,"One temporal test set; three evaluation questions")
    d.card(.2,4.85,6.8,.82,f"{binary['support']:,} later beats from {r['protocol']['independent_subjects']} patient clusters",
           f"{binary['normal_support']:,} normal + {subtype['support']:,} supported abnormal + {other:,} other abnormal",NAVY,10.5)
    d.ax.plot([.55,.55],[4.85,1.225],color=GREY,lw=1.15)
    rows=[
        (3.4,NAVY,"A  Binary detection",binary["support"],
         f"All normal and abnormal beats, including {other:,} other abnormal labels.\nQuestion: does the gate distinguish normal from abnormal?"),
        (2.05,PURPLE,"B  Subtype classification",subtype["support"],
         "True supported abnormal beats; score independently of the binary gate.\nQuestion: does the subtype model choose the correct arrhythmia group?"),
        (.70,TEAL,"C  Complete system",whole["support"],
         "Normal and supported abnormal beats; apply the gate, then subtype.\nQuestion: does the full hierarchy return the correct final class?"),
    ]
    for y,color,title,count,body in rows:
        d.arrow((.55,y+.525),(1.02,y+.525),color)
        d.card(1.02,y,5.98,1.05,f"{title}: {count:,} beats",body,color,10.2)
    d.text(.2,.51,"Supported arrhythmias: PVC, PAC, LBBB, RBBB and AFib.",size=10)
    d.text(.2,.25,"The three groups overlap; each answers a separate evaluation question.",size=9.7,color=GREY)
    d.fig.set_size_inches(7.2, 4.3)
    d.save("final_testing_protocol.pdf")


def main():
    result = corrected_result()
    lead_positions()
    data_preparation()
    training_lifecycle(result)
    autoencoder_architecture()
    method_workflow(result)
    testing_protocol(result)
    print("Generated six thesis diagrams from the corrected protocol.")


if __name__ == "__main__":
    main()
