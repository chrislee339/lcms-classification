"""Single-model paired comparison, CNN vs each classical baseline.

Matches single_model_stats.py: for each seed, metric(CNN seed s) - metric(baseline seed s) on the
pooled samples, averaged over the three seeds. Interval: patient-level cluster bootstrap (same resampled
patients for every seed). p-value: permutation that swaps the two models' predictions patient by
patient (the same swap applied in every seed). Holm across the baselines given, per metric.
N_RESAMPLE (default 10,000) sets the number of bootstrap and permutation resamples.

usage: python3 single_model_compare.py <cnn predictions> <out csv> <baseline predictions> [...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N = int(os.environ.get("N_RESAMPLE", 10_000))
rng = np.random.default_rng(2026)


def wide(path):
    p = pd.read_csv(path)
    p["patient"] = p.filename.str.replace(r"[AB]\.png$", "", regex=True)
    w = p.pivot_table(index=["patient", "filename", "y_true"], columns="seed", values="y_prob").reset_index()
    return w.sort_values(["patient", "y_true"]).reset_index(drop=True)


def seed_mean(y, P, metric):
    f = (lambda p: np.mean((p > 0.5) == y)) if metric == "accuracy" else (lambda p: roc_auc_score(y, p))
    return np.mean([f(p) for p in P])


cnn = wide(sys.argv[1])
rows = []
for path in sys.argv[3:]:
    base = wide(path)
    assert (base.filename.values == cnn.filename.values).all()
    seeds = [c for c in cnn.columns if isinstance(c, (int, np.integer))]
    y = cnn.y_true.values
    A, B = cnn[seeds].values.T, base[seeds].values.T            # (seeds, samples)
    idx = list(cnn.groupby("patient").indices.values())
    for metric in ("accuracy", "auc"):
        obs = seed_mean(y, A, metric) - seed_mean(y, B, metric)
        boots = []
        for _ in range(N):
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            boots.append(seed_mean(y[pick], A[:, pick], metric) - seed_mean(y[pick], B[:, pick], metric))
        perm = []
        for _ in range(N):
            a, b = A.copy(), B.copy()
            for i in np.flatnonzero(rng.random(len(idx)) < 0.5):
                a[:, idx[i]], b[:, idx[i]] = B[:, idx[i]], A[:, idx[i]]
            perm.append(seed_mean(y, a, metric) - seed_mean(y, b, metric))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"cnn": os.path.basename(sys.argv[1]), "baseline": os.path.basename(path), "metric": metric,
                     "cnn_minus_baseline": obs, "lo": lo, "hi": hi,
                     "p_perm": (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (N + 1)})
res = pd.DataFrame(rows)
res["p_holm"] = np.nan
for metric, g in res.groupby("metric"):
    order, running = g.p_perm.sort_values().index, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * res.at[i, "p_perm"]))
        res.at[i, "p_holm"] = running
res.to_csv(sys.argv[2], index=False)
print(res.round(3).to_string(index=False))
