"""Independently recompute saved predictions, metrics, intervals and AE shapes."""
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"backend"))
from app.ml.model import build_model

OUT=ROOT/"Final_Report_Research/experiments"
result=json.loads((OUT/"corrected_temporal_holdout.json").read_text())
pred=np.load(OUT/"corrected_test_predictions.npz")
supported=pred["truth_class"]>=0
matrix=confusion_matrix(pred["truth_class"][supported],pred["pred_system"][supported],labels=np.arange(6))
assert matrix.tolist()==result["test_whole_system"]["confusion_matrix"]
assert accuracy_score(pred["truth_class"][supported],pred["pred_system"][supported])==result["headline"]["whole_system_accuracy"]
assert np.isclose(f1_score(pred["truth_class"][supported],pred["pred_system"][supported],labels=np.arange(6),average="macro"),result["headline"]["whole_system_macro_f1"])
records=pred["records"].astype("<U7")
records[np.isin(records,["201","202"]) ]="201-202"
scores=np.asarray([np.mean(pred["truth_class"][(records==r)&supported]==pred["pred_system"][(records==r)&supported]) for r in np.unique(records)])
rng=np.random.default_rng(20260801)
replicates=scores[rng.integers(0,len(scores),(10000,len(scores)))].mean(axis=1)
np.testing.assert_allclose(np.percentile(replicates,[2.5,97.5]),result["headline"]["whole_system_subject_clustered_ci_95"],atol=1e-12)
model=joblib.load(OUT/"corrected_temporal_holdout.joblib")
data=np.load(ROOT/"backend/app/storage/datasets/mitdb/corrected_beat_local_v1.npz")
first=np.load(ROOT/"backend/app/storage/datasets/mitdb/corrected_local_features_v1.npz")["features"]
ae_features=np.load(ROOT/"backend/app/storage/datasets/mitdb/corrected_local_ae_features_v1.npz")["features"]
indices=[]
for r in np.unique(data["records"]):
    local=np.where(data["records"]==r)[0]
    indices.extend(local[int(.8*len(local)):])
indices=np.asarray(indices)
x=np.concatenate([first[indices],data["rr15"][indices,7:],ae_features[indices]],axis=1)
np.testing.assert_array_equal(model["binary_model"].predict_proba(x)[:,1],pred["probability"])
np.testing.assert_array_equal(model["class_model"].predict(x),pred["pred_subtype"])
torch.set_num_threads(2)
ae=build_model(512,skip_scale=.5).eval()
shapes={}
def hook(name):
    def capture(module,inputs,output): shapes[name]=list(output.shape)
    return capture
for name in ("enc1","enc2","enc3","enc4","up1","up2","up3","up4","final"):
    getattr(ae,name).register_forward_hook(hook(name))
with torch.no_grad(): shapes["output"]=list(ae(torch.zeros(1,1,512)).shape)
audit={"prediction_reproduction":"passed", "confusion_and_macro_f1":"passed", "bootstrap_reproduction":"passed",
       "parameter_count":ae.count_parameters(),"actual_tensor_shapes":shapes,
       "model_sha256":hashlib.sha256((OUT/"corrected_temporal_holdout.joblib").read_bytes()).hexdigest(),
       "test_predictions_sha256":hashlib.sha256((OUT/"corrected_test_predictions.npz").read_bytes()).hexdigest()}
(OUT/"final_verification.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit,indent=2))
