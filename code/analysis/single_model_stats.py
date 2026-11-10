"""Single-model statistics: one trained model's performance on all samples.

For each configuration and seed: pool that seed's out-of-fold predictions over all samples and compute
accuracy, AUC, precision, recall, F1. The estimate is the mean over the three
seeds; the per-seed range is reported too. Interval: patient-level cluster bootstrap (both samples of a patient
together); in each resample the per-seed metrics are recomputed on the same resampled patients and averaged over
seeds. Done for the best checkpoint (y_prob) and the run's top-3 checkpoint average (y_prob_ens). The resampling
restarts from seed 2026 for every file and model, so a result does not depend on the other files in the call.
N_RESAMPLE (default 10,000) sets the number of bootstrap resamples.

usage: python3 single_model_stats.py <out csv> <predictions file> [<predictions file> ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = int(os.environ.get("N_RESAMPLE", 10_000))
METRICS = ("accuracy", "auc", "precision", "recall", "f1")


def metric_values(y, prob):
    pred = prob > 0.5
    precision = np.sum(pred & (y == 1)) / pred.sum() if pred.sum() else 0.0
    recall = np.mean(pred[y == 1])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return np.array([np.mean(pred == y), roc_auc_score(y, prob), precision, recall, f1])


def per_seed(y, probs, pick=None):
    """probs: (n_seeds, n_samples). Returns (n_seeds, 5) metrics on the selected samples."""
    if pick is not None:
        y, probs = y[pick], probs[:, pick]
    return np.array([metric_values(y, p) for p in probs])


rows = []
for path in sys.argv[2:]:
    name = os.path.basename(path).replace("_predictions.csv", "")
    p = pd.read_csv(path)
    p["patient"] = p.filename.str.replace(r"[AB]\.png$", "", regex=True)
    for col in ("y_prob", "y_prob_ens"):
        rng = np.random.default_rng(2026)
        wide = p.pivot_table(index=["patient", "filename", "y_true"], columns="seed", values=col).reset_index()
        assert wide.notna().all().all(), f"{name}: missing seed predictions"
        wide = wide.sort_values(["patient", "y_true"]).reset_index(drop=True)
        seeds = [c for c in wide.columns if isinstance(c, (int, np.integer))]
        y, probs = wide.y_true.values, wide[seeds].values.T
        idx = list(wide.groupby("patient").indices.values())
        observed = per_seed(y, probs)
        boots = np.array([per_seed(y, probs, np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])).mean(0)
                          for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        row = {"configuration": name, "model": "best checkpoint" if col == "y_prob" else "top-3 checkpoints",
               "n_patients": len(idx), "n_samples": len(y), "n_seeds": len(seeds)}
        for k, m in enumerate(METRICS):
            row.update({m: observed[:, k].mean(), f"{m}_lo": lo[k], f"{m}_hi": hi[k],
                        f"{m}_seed_min": observed[:, k].min(), f"{m}_seed_max": observed[:, k].max()})
        rows.append(row)
        print(f"{name:42s} {row['model']:18s} acc {row['accuracy']:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  "
              f"auc {row['auc']:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]", flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
