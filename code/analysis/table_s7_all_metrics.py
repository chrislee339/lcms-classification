"""Table S7: paired tests for all five metrics (accuracy, AUC-ROC, precision, recall and F1), plus the top-3
checkpoint average's accuracy and AUC-ROC, for full spectrum 512 x 512 with per-sample vs global normalization
(fullspec_resnet34_512_withS11rot vs fullspec_resnet34_512_globalnorm_withS11rot).

Uses patient_stats.py's own load(), metrics(), patient_index() and SEED, and the loop of its compare(), with the metric
list extended after accuracy and AUC-ROC. The random stream is consumed in the same order, so the accuracy and AUC-ROC
rows reproduce stats38rot_comparisons.csv (from patient_stats.py --rot) exactly; the script asserts that. One
comparison per metric, so the Holm p-value equals the raw permutation p-value.
Reads results/analysis/predictions/ and results/analysis/stats/stats38rot_comparisons.csv under $LCMS_BASE (default:
the repository root).

usage: python table_s7_all_metrics.py <out csv>
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parents[2]))
out_csv = sys.argv[1]
sys.argv = ["patient_stats.py", f"{BASE}/results/analysis/predictions", out_csv + "_unused", "--rot"]
ns = {"__name__": "patient_stats"}
exec(compile(open(Path(__file__).with_name("patient_stats.py")).read().split("\ndef holm")[0], "patient_stats.py",
             "exec"), ns)
load, metrics, patient_index = ns["load"], ns["metrics"], ns["patient_index"]
N_BOOT, N_PERM, SEED = ns["N_BOOT"], ns["N_PERM"], ns["SEED"]
assert N_BOOT == N_PERM == 10_000


def compare(a_name, b_name, col, which):
    rng = np.random.default_rng(SEED)
    a, b = load(a_name, col), load(b_name, col)
    assert (a.filename.values == b.filename.values).all()
    y = a.y.values
    idx = patient_index(a)
    rows = []
    for k in which:
        d_obs = metrics(y, b.prob.values)[k] - metrics(y, a.prob.values)[k]
        boots = []
        for _ in range(N_BOOT):
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            boots.append(metrics(y[pick], b.prob.values[pick])[k] - metrics(y[pick], a.prob.values[pick])[k])
        perm = []
        for _ in range(N_PERM):
            pa, pb = a.prob.values.copy(), b.prob.values.copy()
            for i in np.flatnonzero(rng.random(len(idx)) < 0.5):
                pa[idx[i]], pb[idx[i]] = b.prob.values[idx[i]], a.prob.values[idx[i]]
            perm.append(metrics(y, pb)[k] - metrics(y, pa)[k])
        p = (np.sum(np.abs(perm) >= abs(d_obs) - 1e-12) + 1) / (N_PERM + 1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"a": a_name, "b": b_name, "model": col, "metric": k, "b_minus_a": d_obs, "lo": lo, "hi": hi,
                     "p_perm": p, "family": "S7 axes and intensity", "p_holm": p})
    return rows


A, B = "fullspec_resnet34_512_withS11rot", "fullspec_resnet34_512_globalnorm_withS11rot"
res = pd.DataFrame(compare(A, B, "y_prob", ("accuracy", "auc", "precision", "recall", "f1")))
stored = pd.read_csv(f"{BASE}/results/analysis/stats/stats38rot_comparisons.csv")
stored = stored[(stored.family == "S7 axes and intensity")]
for col in ("y_prob", "y_prob_ens"):
    for k in ("accuracy", "auc"):
        s = stored[(stored.model == col) & (stored.metric == k)].iloc[0]
        if col == "y_prob":
            r = res[res.metric == k].iloc[0]
            assert np.allclose([r.b_minus_a, r.lo, r.hi, r.p_perm], [s.b_minus_a, s.lo, s.hi, s.p_perm]), (k, r, s)
res = pd.concat([res, stored[stored.model == "y_prob_ens"][res.columns]], ignore_index=True)
res.to_csv(out_csv, index=False)
print(res[["model", "metric", "b_minus_a", "lo", "hi", "p_perm"]].round(4).to_string())
print("accuracy and AUC-ROC (single model) reproduce the stored S7 rows exactly")
