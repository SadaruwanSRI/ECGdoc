"""Reproduce a real-data teaching case without changing trained model artifacts.

The isolated AE update is a new demonstration, not a recovered historical step.
The supervised forests are refitted in memory using the original development
rows and seeds, and checked against the saved experiment. No model is installed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb
from sklearn.ensemble import ExtraTreesClassifier

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
sys.path[:0] = [str(ROOT / "backend"), str(Path(__file__).resolve().parent)]
from app.ml.data import bandpass_filter, resample_to
from app.ml.beat_preparation import prepare_beat, base_rr_at
from app.ml.feature_system import build_features, extend_rr_features
from app.ml.model import build_model
from app.ml.classifier import mitbih_symbol_is_anomaly
from app.ml.train_hierarchical import _rhythm_lookup
from final_temporal_holdout import temporal_split, record_class_weights, subject_groups

DATA = ROOT / "backend/app/storage/datasets"
EXP = REPORT / "experiments"
FIG = REPORT / "pic/generated"
AE_PATH = ROOT / "backend/app/storage/models/model_nsrdb_primary_healthy.pt"
FOREST_PATH = EXP / "corrected_temporal_holdout.joblib"
CLASS_NAMES = ["N", "PVC", "PAC", "LBBB", "RBBB", "AFib"]
NAVY, TEAL, RED, GREY = "#17324D", "#168C8C", "#AD4738", "#63717C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def feature_name(j):
    if j < 64: return f"whole-window mean bin {j}"
    if j < 144: return f"central mean bin {j-64}"
    if j < 204: return f"central derivative bin {j-144}"
    if j < 212: return f"spectral band {j-204}"
    if j < 223:
        return ["mean", "standard deviation", "minimum", "maximum", "range",
                "mean absolute amplitude", "RMS", "derivative standard deviation",
                "maximum absolute derivative", "skewness", "excess kurtosis"][j-212]
    return ["previous RR", "next RR", "local mean RR", "previous/mean RR",
            "next/mean RR", "RR coefficient of variation", "instantaneous heart rate",
            "RR median", "RR IQR", "RR MAD", "RR RMSSD", "RR pNN50", "RR range ratio",
            "RR trend", "latest RR change", "AE MAE", "AE MSE", "AE central MAE",
            "AE residual-change MAE"][j-223]


def node_audit(estimator, features, labels, weights, target):
    tree = estimator.tree_
    reached = np.arange(len(features))
    node = 0
    path = []
    while tree.children_left[node] != -1:
        j, threshold = int(tree.feature[node]), float(tree.threshold[node])
        left = reached[features[reached, j] <= threshold]
        right = reached[features[reached, j] > threshold]
        lc, rc = tree.children_left[node], tree.children_right[node]
        total, wl, wr = weights[reached].sum(), weights[left].sum(), weights[right].sum()
        assert len(reached) == tree.n_node_samples[node]
        np.testing.assert_allclose(total, tree.weighted_n_node_samples[node], rtol=1e-10)
        shares = np.array([weights[reached][labels[reached] == c].sum()
                           for c in np.unique(labels)]) / total
        nonzero = shares[shares > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        np.testing.assert_allclose(entropy, tree.impurity[node], atol=1e-10)
        goes_left = bool(target[j] <= threshold)
        path.append({"node": node, "feature_index": j, "feature": feature_name(j),
                     "threshold": threshold, "input_value": float(target[j]),
                     "branch": "left" if goes_left else "right", "count": len(reached),
                     "weight": total, "class_shares": shares, "entropy": entropy,
                     "left_count": len(left), "right_count": len(right),
                     "left_weight": wl, "right_weight": wr,
                     "left_entropy": tree.impurity[lc], "right_entropy": tree.impurity[rc],
                     "gain": entropy - wl/total*tree.impurity[lc] - wr/total*tree.impurity[rc]})
        node = int(lc if goes_left else rc)
        reached = left if goes_left else right
    return {"path": path, "leaf": node, "leaf_count": len(reached),
            "leaf_score": estimator.predict_proba(target[None])[0]}


def write_path_tables(audit):
    destination = REPORT / "content/final/generated"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("binary", "subtype"):
        lines = [r"{\small", r"\begin{longtable}{@{}r>{\raggedright\arraybackslash}p{4.0cm}rrl@{}}",
                 rf"\caption{{Complete route of the real test PVC through the first saved {name} tree.}}",
                 rf"\label{{tab:real-{name}-path}}\\", r"\toprule",
                 r"Node & Feature (zero-based index) & Beat value & Split value & Branch\\",
                 r"\midrule", r"\endfirsthead", r"\toprule",
                 r"Node & Feature (zero-based index) & Beat value & Split value & Branch\\",
                 r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
        for row in audit["supervised_training"][name]["first_tree"]["path"]:
            lines.append(f"{row['node']} & {row['feature_index']}: {row['feature']} & "
                         f"${row['input_value']:.6f}$ & ${row['threshold']:.6f}$ & {row['branch']}" + r"\\")
        lines.extend([r"\end{longtable}", "}", ""])
        (destination / f"real_{name}_path.tex").write_text("\n".join(lines),encoding="utf-8")


def main():
    start = time.time()
    original_hashes = {"autoencoder": digest(AE_PATH), "forest": digest(FOREST_PATH)}
    torch.set_num_threads(4)
    result = json.loads((EXP / "corrected_temporal_holdout.json").read_text())
    ae_history = json.loads((EXP / "nsrdb_autoencoder_training.json").read_text())
    with np.load(DATA / "mitdb/corrected_beat_local_v1.npz") as stored:
        data = {key: stored[key] for key in stored.files}
    first = np.load(DATA / "mitdb/corrected_local_features_v1.npz")["features"]
    residuals = np.load(DATA / "mitdb/corrected_local_ae_features_v1.npz")["features"]
    features = np.concatenate([first, data["rr15"][:, 7:], residuals], axis=1).astype(np.float32)
    split = temporal_split(data["records"])
    pred = np.load(EXP / "corrected_test_predictions.npz")
    test = split["test"]
    np.testing.assert_array_equal(data["peaks"][test], pred["peaks"])
    chosen = []
    for c in range(6):
        candidates = np.where((pred["truth_class"] == c) & (pred["pred_system"] == c))[0]
        if c == 1:
            fitting = split["selection_train"]
            fitting_records = np.unique(data["records"][fitting[data["classes"][fitting] == 1]])
            candidates = candidates[np.isin(data["records"][test[candidates]], fitting_records)]
        chosen.append(int(test[candidates[0]]))
    pvc = chosen[1]
    rec = str(data["records"][pvc])
    train_candidates = split["selection_train"]
    train_candidates = train_candidates[(data["records"][train_candidates] == rec)
                                         & (data["classes"][train_candidates] == 1)]
    train_pvc = int(train_candidates[len(train_candidates)//2])
    bundle = joblib.load(FOREST_PATH)
    binary, subtype = bundle["binary_model"], bundle["class_model"]
    checkpoint = torch.load(AE_PATH, map_location="cpu", weights_only=False)
    ae = build_model(512, skip_scale=.5)
    ae.load_state_dict(checkpoint["state_dict"])
    ae.eval()
    selected_indices = np.array([train_pvc] + chosen)
    inputs = data["windows"][selected_indices]
    with torch.no_grad():
        reconstruction = ae(torch.from_numpy(inputs).unsqueeze(1)).squeeze(1).numpy()
    e = inputs - reconstruction
    rebuilt_ae = np.stack([np.abs(e).mean(1), (e**2).mean(1),
                          np.abs(e[:, 104:344]).mean(1), np.abs(np.diff(e, axis=1)).mean(1)], axis=1)
    np.testing.assert_allclose(rebuilt_ae, residuals[selected_indices], atol=2e-6)
    rebuilt = np.concatenate([build_features(inputs, data["rr15"][selected_indices]), rebuilt_ae], axis=1)
    np.testing.assert_allclose(rebuilt, features[selected_indices], atol=2e-6)
    np.testing.assert_allclose(binary.predict_proba(rebuilt), binary.predict_proba(features[selected_indices]), atol=1e-12)
    np.testing.assert_allclose(subtype.predict_proba(rebuilt), subtype.predict_proba(features[selected_indices]), atol=1e-12)
    print("Verified selected feature rows against saved model inputs", flush=True)

    # Read the actual locally stored WFDB source; do not change its cache.
    wf = wfdb.rdrecord(str(DATA / "mitdb" / rec))
    channel = wf.sig_name.index("MLII")
    raw128 = resample_to(wf.p_signal[:, channel].astype(np.float32), wf.fs, 128)
    ann = wfdb.rdann(str(DATA / "mitdb" / rec), "atr")
    all_peaks = (ann.sample * 128 / wf.fs).astype(int)
    rhythm_at = _rhythm_lookup(all_peaks, getattr(ann, "aux_note", []))
    eligible_peaks = np.array([p for p, sym in zip(all_peaks, ann.symbol)
                              if mitbih_symbol_is_anomaly(sym, rhythm_at(int(p))) is not None
                              and p >= 200 and p + 312 <= len(raw128)])
    peak = int(data["peaks"][pvc])
    pos = int(np.flatnonzero(eligible_peaks == peak)[0])
    raw_window = raw128[peak-200:peak+312]
    filtered = bandpass_filter(raw_window.astype(np.float64), 128, .5, 50.)
    np.testing.assert_array_equal(prepare_beat(raw_window), data["windows"][pvc])
    rr_history = np.diff(eligible_peaks[max(0,pos-20):pos+1]) / 128.
    rr = extend_rr_features(base_rr_at(eligible_peaks, pos), rr_history)
    np.testing.assert_array_equal(rr, data["rr15"][pvc])

    cases = []
    for idx in selected_indices:
        local = np.flatnonzero(data["records"] == data["records"][idx])
        ordinal = int(np.flatnonzero(local == idx)[0])
        probability = float(binary.predict_proba(features[idx:idx+1])[0,1])
        q = subtype.predict_proba(features[idx:idx+1])[0]
        output = 0 if probability <= bundle["binary_threshold"] else int(subtype.classes_[np.argmax(q)])
        cases.append({"dataset_index": int(idx), "record": str(data["records"][idx]),
                      "accepted_index_zero_based": ordinal, "record_accepted_count": len(local),
                      "cut64": int(.64*len(local)), "cut80": int(.8*len(local)),
                      "peak128": int(data["peaks"][idx]), "seconds": float(data["peaks"][idx]/128),
                      "truth": CLASS_NAMES[data["classes"][idx]], "output": CLASS_NAMES[output],
                      "binary_score": probability, "subtype_scores": q,
                      "rr15": data["rr15"][idx], "residual4": residuals[idx],
                      "feature242": features[idx], "role": "candidate fitting" if idx==train_pvc else "temporal test"})
    pvc_case = cases[2]
    source_positions = np.flatnonzero(all_peaks == peak)
    source = {"record": rec, "channel": channel, "lead": "MLII", "source_fs": wf.fs,
              "original_annotation_sample": int(ann.sample[source_positions[0]]),
              "symbol": ann.symbol[source_positions[0]], "peak128": peak,
              "previous_peak128": int(eligible_peaks[pos-1]), "next_peak128": int(eligible_peaks[pos+1]),
              "window_start128": peak-200, "window_last128": peak+311,
              "filtered_mean": float(filtered.mean()), "filtered_std": float(filtered.std()),
              "raw_at_peak": float(raw_window[200]), "filtered_at_peak": float(filtered[200]),
              "standardised_at_peak": float(inputs[2,200]), "history20": rr_history,
              "first_pool_values": inputs[2,:8], "first_pool_mean": float(inputs[2,:8].mean())}

    # Original final fitting is reproducible; all new fits remain in memory.
    supervised = {}
    for name, model, labels, population, candidate, seed in [
        ("binary", binary, data["binary"], split["development"], result["selected_binary"]["candidate"], 20260901),
        ("subtype", subtype, data["classes"], split["development"][data["classes"][split["development"]]>0], result["selected_subtype"], 20260902),
    ]:
        weights = record_class_weights(labels[population], subject_groups(data["records"][population]), candidate["record_power"])
        print(f"Reproducing {name} forest: {len(population)} real development beats", flush=True)
        fitted = ExtraTreesClassifier(n_estimators=candidate["trees"], max_depth=candidate["max_depth"],
                    min_samples_leaf=candidate["min_samples_leaf"], max_features=candidate["max_features"],
                    criterion="entropy", random_state=seed, n_jobs=4)
        fitted.fit(features[population], labels[population], sample_weight=weights)
        for actual, saved in zip(fitted.estimators_, model.estimators_):
            np.testing.assert_array_equal(actual.tree_.feature, saved.tree_.feature)
            np.testing.assert_array_equal(actual.tree_.threshold, saved.tree_.threshold)
            np.testing.assert_array_equal(actual.tree_.value, saved.tree_.value)
        np.testing.assert_allclose(fitted.predict_proba(features[test]), model.predict_proba(features[test]), atol=1e-12)
        audit = node_audit(model.estimators_[0], features[population], labels[population], weights, features[pvc])
        loc = int(np.flatnonzero(population == train_pvc)[0])
        groups = subject_groups(data["records"][population])
        c = labels[train_pvc]
        cell = (labels[population] == c) & (groups == groups[loc])
        counts = {"class_count": int(np.sum(labels[population]==c)),
                  "class_subjects": int(len(np.unique(groups[labels[population]==c]))),
                  "cell_count": int(cell.sum()), "normalised_weight": float(weights[loc]),
                  "alpha": candidate["record_power"]}
        leaf_scores = np.array([tree.predict_proba(features[pvc:pvc+1])[0] for tree in model.estimators_])
        supervised[name] = {"exact_forest_reproduction": True, "seed": seed,
                           "training_rows": len(population), "training_example_weight": counts,
                           "first_tree": audit, "sum_tree_scores": leaf_scores.sum(0),
                           "mean_tree_scores": leaf_scores.mean(0)}
        del fitted
        gc.collect()
        print(f"{name}: all saved splits and leaf values reproduced", flush=True)

    # New one-step demonstration on a real fitting subject, separate from saved AE.
    healthy = np.load(DATA / "nsrdb-primary-full/16272_primary_full.npy", mmap_mode="r")
    normal_raw = np.load(DATA / "nsrdb/16272.npy", mmap_mode="r")
    normal_filtered = bandpass_filter(np.asarray(normal_raw[:230400+512]),128)
    # Archived accepted windows retain their historical scaling. In particular,
    # they are not all unit variance. Verify the source waveform up to its
    # archived affine scaling instead of silently renormalising training input.
    source_design = np.column_stack([normal_filtered[:512], np.ones(512)])
    affine = np.linalg.lstsq(source_design, healthy[0], rcond=None)[0]
    np.testing.assert_allclose(source_design @ affine, healthy[0], atol=2e-6)
    demonstration_seed = 20260927
    torch.manual_seed(demonstration_seed)
    demo = build_model(512, skip_scale=.5)
    demo.train()
    batch = torch.from_numpy(np.array(healthy[:256])).unsqueeze(1)
    noisy = batch + torch.randn_like(batch)*.03
    optimiser = torch.optim.Adam(demo.parameters(), lr=.001)
    dropout_state = torch.get_rng_state()
    before = demo(noisy)
    loss = torch.nn.functional.mse_loss(before,batch)
    loss.backward()
    parameter_before = float(demo.final.bias[0].detach())
    gradient = float(demo.final.bias.grad[0])
    optimiser.step()
    parameter_after = float(demo.final.bias[0].detach())
    # Compare the same noisy batch with the same dropout masks.
    torch.set_rng_state(dropout_state)
    with torch.no_grad():
        after = demo(noisy)
        after_loss = float(torch.nn.functional.mse_loss(after,batch))
        healthy_saved_reconstruction = ae(batch[:1]).squeeze().numpy()
    demonstration = {"kind": "new isolated Adam update; not historical training replay",
        "seed": demonstration_seed, "record": "16272", "channel": "ECG1",
        "shown_window_cached_index": 0, "shown_source_samples": [0,511],
        "batch_cached_indices": [0,255], "batch_size": 256,
        "cached_window_mean": float(healthy[0].mean()),
        "cached_window_std": float(healthy[0].std()),
        "source_filtered_affine_gain": float(affine[0]),
        "source_filtered_affine_offset": float(affine[1]),
        "source_affine_max_error": float(np.max(np.abs(source_design @ affine - healthy[0]))),
        "loss_before": float(loss.detach()), "loss_after_same_dropout": after_loss,
        "shown_window_loss_before": float(torch.mean((before[0]-batch[0])**2).detach()),
        "shown_window_loss_after": float(torch.mean((after[0]-batch[0])**2)),
        "parameter": "final.bias[0]", "parameter_before": parameter_before,
        "gradient": gradient, "parameter_after": parameter_after,
        "saved_checkpoint_shown_window_mse": float(np.mean((healthy[0]-healthy_saved_reconstruction)**2)),
        "first_clean_values": batch[0,0,:6].numpy(), "first_noisy_values": noisy[0,0,:6].numpy()}
    print("Isolated real-window Adam demonstration complete", flush=True)

    # Actual ECG graphics with source identifiers and declared selection rule.
    t = np.arange(512)/128.
    fig, axes = plt.subplots(3,1,figsize=(8.2,7.0),sharex=True,layout="constrained")
    axes[0].plot(t,batch[0,0].numpy(),c=NAVY,lw=1.3,label="clean target")
    axes[0].plot(t,noisy[0,0].numpy(),c=TEAL,lw=.8,alpha=.7,label="noisy training input")
    axes[0].set_title("A. Real NSRDB 16272 / ECG1, first four seconds",loc="left",weight="bold")
    axes[1].plot(t,batch[0,0].numpy(),c=NAVY,lw=1.1,label="clean target")
    axes[1].plot(t,before[0,0].detach().numpy(),c=GREY,lw=.8,label="before one update")
    axes[1].plot(t,after[0,0].numpy(),c=RED,lw=.9,label="after one update")
    axes[1].set_title("B. New demonstration: one Adam step from random weights",loc="left",weight="bold")
    axes[2].plot(t,batch[0,0].numpy(),c=NAVY,lw=1.1,label="clean input")
    axes[2].plot(t,healthy_saved_reconstruction,c=TEAL,lw=1.0,label="saved two-epoch model")
    axes[2].set_title("C. Original selected checkpoint on the same real window",loc="left",weight="bold")
    for ax in axes:
        ax.set_ylabel("archived normalised units")
        ax.legend(fontsize=8,loc="upper right",ncol=2)
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("time from recording start (s)")
    save(fig,"method_real_ae_training")

    fig, axes = plt.subplots(3,1,figsize=(8.2,7.0),sharex=True,layout="constrained")
    offset=(np.arange(512)-200)/128.
    for ax, row, title in zip(axes,[0,2,1],["A. Earlier PVC used for supervised fitting", "B. Later PVC reserved for temporal testing", "C. Later normal beat reserved for temporal testing"]):
        ax.plot(offset,inputs[row],c=NAVY,lw=1.2,label="real prepared MLII")
        ax.plot(offset,reconstruction[row],c=TEAL,lw=1,alpha=.85,label="frozen AE reconstruction")
        ax.axvline(0,c=RED,ls="--",lw=.9)
        ca=cases[row]
        ax.set_title(f"{title}\nrecord {ca['record']}, R peak at {ca['seconds']:.3f} s",loc="left",fontsize=10,weight="bold")
        ax.set_ylabel("standardised amplitude")
        ax.legend(fontsize=8,loc="upper right")
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("time relative to the target R peak (s)")
    save(fig,"method_real_train_test_segments")

    fig, axes=plt.subplots(3,1,figsize=(8.2,7.0),layout="constrained")
    axes[0].plot(offset,raw_window,c=GREY,lw=1,label="resampled raw MLII")
    axes[0].plot(offset,filtered,c=NAVY,lw=1,label="window-local filtered MLII")
    axes[0].axvline(0,c=RED,ls="--",lw=.9)
    axes[0].set(title=f"A. Test PVC: record {rec}, peak {peak} at 128 Hz",ylabel="mV",xlabel="time relative to R peak (s)")
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].plot(offset,inputs[2],c=NAVY,lw=1,label="prepared input")
    axes[1].plot(offset,reconstruction[2],c=TEAL,lw=1,label="AE reconstruction")
    axes[1].fill_between(offset,0,np.abs(e[2]),color=RED,alpha=.22,label="absolute residual")
    axes[1].set(title="B. Fixed reconstruction produces four measured residual features",ylabel="standardised amplitude",xlabel="time relative to R peak (s)")
    axes[1].legend(fontsize=8,ncol=3)
    scores=pvc_case["subtype_scores"]
    axes[2].bar(CLASS_NAMES[1:],scores,color=[RED,TEAL,NAVY,GREY,"#8774A0"])
    for j,val in enumerate(scores): axes[2].text(j,val+.015,f"{val:.4f}",ha="center",fontsize=9)
    axes[2].set(title=f"C. Binary score {pvc_case['binary_score']:.6f} > {bundle['binary_threshold']:.6f}: run subtype forest",ylabel="mean leaf score",ylim=(0,1.14))
    for ax in axes: ax.grid(alpha=.15,axis="y")
    save(fig,"method_real_prediction_walkthrough")

    fig, axes=plt.subplots(3,2,figsize=(8.4,8.0),sharex=True,layout="constrained")
    for ax,row in zip(axes.ravel(),range(1,7)):
        ca=cases[row]
        ax.plot(offset,inputs[row],c=NAVY,lw=1.0)
        ax.axvline(0,c=RED,ls="--",lw=.8)
        q=ca["subtype_scores"]
        score=1-ca["binary_score"] if ca["output"]=="N" else max(q)
        ax.set_title(f"{ca['truth']} label / {ca['output']} prediction\nrecord {ca['record']}, {ca['seconds']:.3f} s",loc="left",fontsize=10,weight="bold")
        ax.text(.02,.03,f"p(abnormal)={ca['binary_score']:.4f}; displayed={score:.4f}",transform=ax.transAxes,fontsize=8,bbox=dict(fc="white",ec="none",alpha=.85))
        ax.set_ylabel("standardised amplitude")
        ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel("time from target R peak (s)")
    save(fig,"method_real_six_outputs")

    assert original_hashes == {"autoencoder": digest(AE_PATH), "forest": digest(FOREST_PATH)}
    audit={"artifact_hashes_unchanged": original_hashes, "selection_rule": "first correctly classified test beat of each class in saved test order; PVC additionally requires a record with candidate-fitting PVCs; training PVC is middle candidate-fitting PVC in that record",
           "autoencoder_demonstration": demonstration, "original_ae_history": ae_history["history"],
           "cases": cases, "pvc_source_preparation": source, "supervised_training": supervised,
           "binary_candidates": result["binary_selection"], "subtype_candidates": result["subtype_selection"],
           "threshold": bundle["binary_threshold"], "elapsed_seconds": time.time()-start}
    (EXP / "methodology_real_case.json").write_text(json.dumps(audit,indent=2,default=serial),encoding="utf-8")
    write_path_tables(audit)
    print(json.dumps({"ae_demo": demonstration, "pvc": {k:v for k,v in pvc_case.items() if k not in ("feature242",)},
                      "source": source, "elapsed_seconds":audit["elapsed_seconds"]},default=serial,indent=2),flush=True)


if __name__ == "__main__":
    main()
