"""Patient-level statistics on the out-of-fold predictions of each configuration.

Unit: the out-of-fold samples (2 per patient). For each sample the tumor probability is averaged over the
three seeds, then thresholded at 0.5. Intervals: patient-level cluster bootstrap (both samples of a
patient resampled together). Paired comparisons: patient-level bootstrap interval for the difference and a
paired permutation test that swaps the two configurations' predictions patient by patient. Holm
correction within each comparison family.
Metrics: accuracy, AUC-ROC, precision, recall, F1. Each summary and
each comparison restarts the resampling from SEED, so its result does not depend on what else is run.
N_RESAMPLE (default 10,000) sets the number of bootstrap and permutation resamples.
Writes <out prefix>_summary.csv (one row per *_predictions.csv in the folder and model) and
<out prefix>_comparisons.csv.

usage: python3 patient_stats.py <predictions dir> <out prefix> [--37 | --rot]
  --37   use the 37-patient (S11-excluded) prediction files and names
  --rot  use the 38-patient files in which S11 is always in a training set (names ending _withS11rot)
         (without either flag: the 38-patient files with names ending _withS11)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = N_PERM = int(os.environ.get("N_RESAMPLE", 10_000))  # small values only for testing the code
SEED = 2026

pred_dir, out_prefix = sys.argv[1], sys.argv[2]
suffix = "" if "--37" in sys.argv else "_withS11rot" if "--rot" in sys.argv else "_withS11"


def load(name, col):
    """Seed-averaged probability per sample, ordered by patient then class."""
    p = pd.read_csv(os.path.join(pred_dir, f"{name}_predictions.csv"))
    assert p.groupby("filename").seed.nunique().eq(3).all(), f"{name}: not every sample has 3 seeds"
    s = p.groupby("filename").agg(y=("y_true", "first"), prob=(col, "mean")).reset_index()
    s["patient"] = s.filename.str.replace(r"[AB]\.png$", "", regex=True)
    return s.sort_values(["patient", "y"]).reset_index(drop=True)


def metrics(y, prob):
    pred = prob > 0.5
    precision = np.sum(pred & (y == 1)) / pred.sum() if pred.sum() else 0.0
    recall = np.mean(pred[y == 1])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": np.mean(pred == y), "auc": roc_auc_score(y, prob),
            "precision": precision, "recall": recall, "f1": f1}


def patient_index(s):
    groups = s.groupby("patient").indices
    return [groups[p] for p in sorted(groups)]


rng = np.random.default_rng(SEED)


def bootstrap(s_list, fn):
    """Percentile 95% interval of fn(*resampled frames), resampling patients."""
    idx = patient_index(s_list[0])
    out = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        out.append(fn(*[s.iloc[pick] for s in s_list]))
    return np.percentile(out, [2.5, 97.5])


def summarize(name, col):
    global rng
    rng = np.random.default_rng(SEED)
    s = load(name, col)
    m = metrics(s.y.values, s.prob.values)
    row = {"configuration": name, "model": col, "n_patients": s.patient.nunique(), "n_samples": len(s)}
    for k, v in m.items():
        lo, hi = bootstrap([s], lambda a, k=k: metrics(a.y.values, a.prob.values)[k])
        row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
    return row


def compare(a_name, b_name, col):
    global rng
    rng = np.random.default_rng(SEED)
    a, b = load(a_name, col), load(b_name, col)
    assert (a.filename.values == b.filename.values).all()
    y = a.y.values
    rows = []
    for k in ("accuracy", "auc"):
        d_obs = metrics(y, b.prob.values)[k] - metrics(y, a.prob.values)[k]
        idx = patient_index(a)
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
        rows.append({"a": a_name, "b": b_name, "model": col, "metric": k, "b_minus_a": d_obs,
                     "lo": lo, "hi": hi, "p_perm": p})
    return rows


def holm(df):
    df = df.copy()
    df["p_holm"] = np.nan
    for (fam, model, metric), g in df.groupby(["family", "model", "metric"]):
        order = g.p_perm.sort_values().index
        m, running = len(order), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * df.at[i, "p_perm"]))
            df.at[i, "p_holm"] = running
    return df


A = [f"arch_resnet{v}_256{suffix}" for v in (18, 34, 50, 101, 152)]
FULL = {256: f"arch_resnet34_256{suffix}" if suffix else "fullspec_resnet34_256",
        512: f"fullspec_resnet34_512{suffix}", 1024: f"fullspec_resnet34_1024{suffix}"}
SUB = {r: f"subregion_resnet34_{r}{suffix}" for r in (256, 512, 1024)}
available = {os.path.basename(f).replace("_predictions.csv", "") for f in glob.glob(f"{pred_dir}/*_predictions.csv")}

families = {
    "architecture": [(a, b) for i, a in enumerate(A) for b in A[i + 1:]],
    "full-spectrum binning": [(FULL[256], FULL[512]), (FULL[256], FULL[1024]), (FULL[512], FULL[1024])],
    "subregion binning": [(SUB[256], SUB[512]), (SUB[256], SUB[1024]), (SUB[512], SUB[1024])],
    "full vs subregion": [(FULL[r], SUB[r]) for r in (256, 512, 1024)],
    "S7 axes and intensity": [(FULL[512], f"fullspec_resnet34_512_globalnorm{suffix}")],
}

if __name__ == "__main__":  # cnn_vs_classical.py and nested_result*.py import this module for its functions only
    configs = sorted(available)
    summary = pd.DataFrame([summarize(n, c) for n in configs for c in ("y_prob", "y_prob_ens")])
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)

    comp = []
    for fam, pairs in families.items():
        for a, b in pairs:
            if a in available and b in available:
                for c in ("y_prob", "y_prob_ens"):
                    comp += [dict(r, family=fam) for r in compare(a, b, c)]
    comp = holm(pd.DataFrame(comp))
    comp.to_csv(f"{out_prefix}_comparisons.csv", index=False)
    print(f"wrote {out_prefix}_summary.csv ({len(summary)} rows) and _comparisons.csv ({len(comp)} rows)")
