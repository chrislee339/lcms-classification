"""Nested cross-validation result on 38 patients (S11 always in a training set).

Selection: for each outer fold, each candidate's inner score is the AUC-ROC of its best checkpoint (y_prob) on the
inner-test predictions pooled over the 5 inner folds (each non-test patient predicted once), computed per seed and
averaged over the 3 seeds. Highest score wins; ties go to the lower mean log loss.
Outer estimate: for each outer fold, the chosen candidate's outer-fold predictions (from evaluate_checkpoints.py) are
assembled into 76 samples x 3 seeds and scored with single_model_stats.py (single model, mean of the 3 seeds) and
patient_stats.py (three-seed ensemble), each in its own process so the bootstrap starts from seed 2026.
The full-spectrum 512 x 512 ResNet-34 configuration (fullspec_resnet34_512_withS11rot) is scored the same way for
reference.
Outputs in <out dir>: nested_withS11rot_predictions.csv, picks.csv, inner_scores.csv, inner_scores_per_seed.csv,
single_<label>.csv and ensemble_<label>.csv for the nested estimate and the reference, inputs_sha256.txt.

usage: python3 nested_result.py <nested dir> <predictions dir> <stats code dir> <splits json> <out dir>
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

nested_dir, eval_dir, stats_dir, splits_path, out_dir = sys.argv[1:6]
CANDIDATES = ([f"arch_resnet{v}_256_withS11rot" for v in (18, 34, 50, 101, 152)]
              + ["fullspec_resnet34_512_withS11rot", "fullspec_resnet34_1024_withS11rot"]
              + [f"subregion_resnet34_{r}_withS11rot" for r in (256, 512, 1024)])
FLAT = "fullspec_resnet34_512_withS11rot"
SEEDS, OUTER = (42, 69, 137), range(6)
NAME = "nested_withS11rot"
os.makedirs(out_dir, exist_ok=True)
patient = lambda s: s.str.replace(r"[AB]\.png$", "", regex=True)  # noqa: E731

# ---- load and check the 180 inner units
files = sorted(glob.glob(os.path.join(nested_dir, "preds", "*__outer*_seed*.csv")))
assert len(files) == len(CANDIDATES) * 6 * 3, len(files)
with open(os.path.join(out_dir, "inputs_sha256.txt"), "w") as fh:
    for f in files:
        fh.write(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {os.path.basename(f)}\n")
inner = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
inner["patient"] = patient(inner.filename)
splits = json.load(open(splits_path))
inner_splits = json.load(open(os.path.join(nested_dir, "kfold", "inner_splits.json")))

for (c, k, s), g in inner.groupby(["candidate", "outer", "seed"]):
    assert c in CANDIDATES and s in SEEDS, (c, s)
    pool = set(splits[str(k)]["train"]) | set(splits[str(k)]["val"])
    assert g.filename.is_unique and set(g.patient) == pool and len(g) == 2 * len(pool), (c, k, s)
    assert not set(g.patient) & set(splits[str(k)]["test"]), (c, k, s, "outer test patient in an inner fold")
    for j, gj in g.groupby("inner_fold"):
        assert set(gj.patient) == set(inner_splits[str(k)][str(j)]["test"]), (c, k, s, j)
assert inner.groupby(["candidate", "outer", "seed"]).ngroups == 180

# ---- inner scores and selection
p = inner.y_prob.clip(1e-15, 1 - 1e-15)
inner["nll"] = -(inner.y_true * np.log(p) + (1 - inner.y_true) * np.log(1 - p))
per_seed = (inner.groupby(["outer", "candidate", "seed"])
            .apply(lambda g: pd.Series({"inner_auc": roc_auc_score(g.y_true, g.y_prob), "inner_logloss": g.nll.mean()}))
            .reset_index())
scores = per_seed.groupby(["outer", "candidate"])[["inner_auc", "inner_logloss"]].mean().reset_index()
scores = scores.sort_values(["outer", "inner_auc", "inner_logloss"], ascending=[True, False, True])
scores["rank"] = scores.groupby("outer").cumcount() + 1
picks = scores[scores["rank"] == 1].reset_index(drop=True)
runner = scores[scores["rank"] == 2].set_index("outer")
picks["tie_on_auc"] = [abs(r.inner_auc - runner.at[r.outer, "inner_auc"]) < 1e-12 for r in picks.itertuples()]
per_seed.to_csv(os.path.join(out_dir, "inner_scores_per_seed.csv"), index=False)
scores.to_csv(os.path.join(out_dir, "inner_scores.csv"), index=False)

# ---- outer estimate from the chosen candidates' existing outer predictions
parts = []
for r in picks.itertuples():
    o = pd.read_csv(os.path.join(eval_dir, f"{r.candidate}_predictions.csv"))
    o = o[o.fold == r.outer]
    assert set(patient(o.filename)) == set(splits[str(r.outer)]["test"]) and set(o.seed) == set(SEEDS), r
    parts.append(o)
nested = pd.concat(parts, ignore_index=True)
assert len(nested) == 228 and nested.groupby("seed").filename.nunique().eq(76).all(), len(nested)
nested.to_csv(os.path.join(out_dir, f"{NAME}_predictions.csv"), index=False)

flat = pd.read_csv(os.path.join(eval_dir, f"{FLAT}_predictions.csv"))
fold_rows = []
for r in picks.itertuples():
    row = {"outer": r.outer, "n_test_patients": len(splits[str(r.outer)]["test"]), "chosen": r.candidate,
           "inner_auc": r.inner_auc, "inner_logloss": r.inner_logloss, "tie_on_auc": r.tie_on_auc,
           "flat_inner_auc": scores.set_index(["outer", "candidate"]).at[(r.outer, FLAT), "inner_auc"],
           "flat_rank": int(scores.set_index(["outer", "candidate"]).at[(r.outer, FLAT), "rank"])}
    for label, df in (("chosen", nested), ("flat", flat)):
        f = df[df.fold == r.outer]
        row[f"{label}_outer_correct"] = int(((f.y_prob > 0.5) == f.y_true).sum())  # summed over the 3 seeds
    fold_rows.append(row)
folds = pd.DataFrame(fold_rows)
folds.to_csv(os.path.join(out_dir, "picks.csv"), index=False)

# ---- statistics, each scoring in its own process (bootstrap seed 2026 from the start)
py = sys.executable
nested_csv = os.path.join(out_dir, f"{NAME}_predictions.csv")
for label, path in ((NAME, nested_csv), (FLAT, os.path.join(eval_dir, f"{FLAT}_predictions.csv"))):
    subprocess.run([py, os.path.join(stats_dir, "single_model_stats.py"),
                    os.path.join(out_dir, f"single_{label}.csv"), path], check=True)
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); d, n, o = sys.argv[2:5]; "
            "sys.argv = ['patient_stats.py', d, o + '_unused']; import patient_stats as ps; "
            "import pandas as pd; pd.DataFrame([ps.summarize(n, 'y_prob')]).to_csv(o, index=False)")
    subprocess.run([py, "-c", code, stats_dir, os.path.dirname(path), label,
                    os.path.join(out_dir, f"ensemble_{label}.csv")], check=True)

print("\nselection by outer fold:")
print(folds.to_string(index=False))
print("\ntimes chosen:", picks.candidate.value_counts().to_dict())
print(f"distinct configurations chosen: {picks.candidate.nunique()} of {len(CANDIDATES)}")
