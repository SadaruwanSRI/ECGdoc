"""Fixed post-fit check of complete locally cached INCART I01 and I02 records.

Selection is by local availability, not performance. No threshold optimisation
or model fitting occurs here. This is a limited external-source check.
"""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
import wfdb
from corrected_dataset import extract, ROOT
from final_temporal_holdout import binary_metrics, multiclass_metrics, subtype_metrics
from app.ml.feature_system import build_features
from app.ml.model import build_model

OUT = ROOT / "Final_Report_Research/experiments"
torch.set_num_threads(4)
model_path = OUT / "corrected_temporal_holdout.joblib"
bundle = joblib.load(model_path)
ae_path = ROOT / "backend/app/storage/models/model_nsrdb_primary_healthy.pt"
checkpoint = torch.load(ae_path, map_location="cpu", weights_only=False)
ae = build_model(512, skip_scale=float(checkpoint["config"]["skip_scale"]))
ae.load_state_dict(checkpoint["state_dict"])
ae.eval()
pieces, headers = [], {}
for record in ("I01", "I02"):
    piece = extract(record, "incartdb", "II")
    pieces.append(piece)
    header = wfdb.rdheader(record, pn_dir="incartdb")
    headers[record] = header.comments
    print(f"[external] {record}: {len(piece['windows'])} beats", flush=True)
data = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
residual_rows = []
with torch.no_grad():
    for start in range(0, len(data["windows"]), 256):
        x = torch.from_numpy(data["windows"][start:start+256]).unsqueeze(1)
        residual = x-ae(x)
        absolute = residual.abs()
        residual_rows.append(torch.stack([absolute.mean((1,2)), residual.square().mean((1,2)),
            absolute[:,:,104:344].mean((1,2)), torch.diff(residual,dim=2).abs().mean((1,2))],dim=1).numpy())
features = np.concatenate([build_features(data["windows"],data["rr15"]),np.concatenate(residual_rows)],axis=1)
p = bundle["binary_model"].predict_proba(features)[:,1]
binary = p > bundle["binary_threshold"]
subtype = bundle["class_model"].predict(features)
system = np.where(binary,subtype,0)
supported = data["classes"] >= 0
abnormal = data["classes"] > 0
result = {"protocol": {"records": ["I01","I02"], "selection": "all complete locally cached INCART lead-II records at audit start",
          "threshold_tuned": False, "lead": "II", "mode": "offline annotated beats", "headers":headers,
          "limitation": "convenience subset; source patients may repeat across records; no population generalisation claim"},
          "threshold":bundle["binary_threshold"], "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
          "test_binary":binary_metrics(data["binary"],p,binary),
          "test_whole_system":multiclass_metrics(data["classes"][supported],system[supported]),
          "test_subtype":subtype_metrics(data["classes"][abnormal],subtype[abnormal]),
          "per_record": {r:binary_metrics(data["binary"][data["records"]==r],p[data["records"]==r],binary[data["records"]==r]) for r in ("I01","I02")}}
(OUT / "corrected_external_incart.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
np.savez_compressed(OUT / "corrected_external_predictions.npz",records=data["records"],peaks=data["peaks"],
                    truth_binary=data["binary"],truth_class=data["classes"],pred_binary=binary,
                    pred_subtype=subtype,pred_system=system,probability=p)
print(json.dumps(result,indent=2))
