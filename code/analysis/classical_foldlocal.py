"""Classical baselines with fold-local preprocessing and tuning.

Reuses code/pick_peaks_classical.py, unchanged, for peak picking, binning, the 20% prevalence filter, the
log10(x+1) transform and the classifier settings. What differs from pick_peaks_classical.py:

1. Fold-local features. pick_peaks_classical.py builds the (m/z, RT) grid and the 20% filter once from all
   samples, so test samples help decide which features exist. Here they are built from each fold's
   training samples only (train + val patients, as in pick_peaks_classical.py), and test samples are mapped onto
   that grid.
2. Tuning. Besides the fixed settings of pick_peaks_classical.py (RF 500 trees, sqrt features; PLS-DA 5
   components), each classifier is tuned by 5-fold patient-grouped inner cross-validation inside the training set,
   with the feature grid rebuilt inside every inner split too. Selection by mean inner AUC.
3. Both cohorts: 38 patients (primary) and 37 (S11 excluded, sensitivity).
4. Check first: pick_peaks_classical.py's exact procedure on its cached 74-sample features must reproduce its
   results.

Inputs: --mzml, the folder of mzML files (S11's two runs are extracted from it); --feature-cache, the feature cache
written by pick_peaks_classical.py --feature-cache; --reference-detailed, the classical_baselines_detailed.csv of that
run (results/classical_baselines_detailed.csv); --splits37 and --splits38, data/kfold/patient_splits.json and
data/kfold/patient_splits_withS11.json.
Outputs (small CSVs, in --out): per-sample predictions in the same format as the CNN
(fold, seed, filename, y_true, y_prob, y_prob_ens), classical_<rf|plsda>_<fixed|tuned>_<37|38>_predictions.csv;
a per-fold log of feature counts and chosen settings, classical_folds_<37|38>.csv; the reproduction check,
reproduction_check.csv; and S11's extracted features, s11_features.json.

usage: python3 classical_foldlocal.py --mzml DIR --feature-cache JSON --reference-detailed CSV --splits37 JSON
       --splits38 JSON --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pick_peaks_classical as ppc  # noqa: E402

SEEDS = (42, 69, 137)
MZ_BIN, RT_BIN, MIN_FREQ = 0.01, 5.0, 0.2  # pick_peaks_classical.py defaults
RF_GRID = [{"max_features": mf, "min_samples_leaf": leaf} for mf in ("sqrt", 0.1, 0.3) for leaf in (1, 3)]
PLS_GRID = [1, 2, 3, 4, 5, 8, 10]
log = ppc.log


def grid_keys(per_file, ids):
    """pick_peaks_classical.py's binning and prevalence filter, applied to the given samples only."""
    counts = {}
    for sid in ids:
        seen = {(int(round(mz / MZ_BIN)), int(round(rt / RT_BIN))) for mz, rt, _ in per_file[sid]}
        for k in seen:
            counts[k] = counts.get(k, 0) + 1
    min_n = max(2, int(MIN_FREQ * len(ids)))
    return sorted(k for k, c in counts.items() if c >= min_n), len(counts)


def matrix(per_file, ids, keys):
    """Max intensity per kept bin (as in pick_peaks_classical.py), then log10(x + 1)."""
    col = {k: j for j, k in enumerate(keys)}
    X = np.zeros((len(ids), len(keys)), dtype=np.float32)
    for i, sid in enumerate(ids):
        for mz, rt, inten in per_file[sid]:
            j = col.get((int(round(mz / MZ_BIN)), int(round(rt / RT_BIN))))
            if j is not None:
                X[i, j] = max(X[i, j], float(inten))
    return np.log10(X + 1.0)


def fit_predict(kind, params, X_tr, y_tr, X_te, seed):
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1, **params)
        return clf.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    scaler = StandardScaler().fit(X_tr)
    n_comp = min(params["n_components"], X_tr.shape[0] - 1, X_tr.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=False).fit(scaler.transform(X_tr), y_tr.astype(float))
    return pls.predict(scaler.transform(X_te)).ravel()


def fold_local(per_file, train_ids, test_ids, kind, params, seed):
    keys, n_candidates = grid_keys(per_file, train_ids)
    y_tr = np.array([ppc.label_from_sample_id(s) for s in train_ids])
    score = fit_predict(kind, params, matrix(per_file, train_ids, keys), y_tr,
                        matrix(per_file, test_ids, keys), seed)
    return score, len(keys), n_candidates


def tune(per_file, train_ids, kind, seed):
    grid = RF_GRID if kind == "rf" else [{"n_components": c} for c in PLS_GRID]
    groups = [ppc.patient_from_sample_id(s) for s in train_ids]
    y = np.array([ppc.label_from_sample_id(s) for s in train_ids])
    best, best_auc = None, -1.0
    for params in grid:
        aucs = []
        for tr, va in GroupKFold(n_splits=5).split(train_ids, y, groups):
            score, _, _ = fold_local(per_file, [train_ids[i] for i in tr], [train_ids[i] for i in va],
                                     kind, params, seed)
            aucs.append(roc_auc_score(y[va], score))
        if np.mean(aucs) > best_auc:
            best, best_auc = params, float(np.mean(aucs))
    return best, best_auc


def run_cohort(per_file, splits, cohort, out):
    ids = sorted(per_file)
    pred_rows, log_rows = [], []
    for kind, fixed in (("rf", {"max_features": "sqrt"}), ("plsda", {"n_components": 5})):
        for setting in ("fixed", "tuned"):
            for seed in SEEDS:
                for fold in range(6):
                    train_pat = set(splits[fold]["train"]) | set(splits[fold]["val"])
                    train_ids = [s for s in ids if ppc.patient_from_sample_id(s) in train_pat]
                    test_ids = [s for s in ids if ppc.patient_from_sample_id(s) in set(splits[fold]["test"])]
                    params, inner_auc = (fixed, np.nan) if setting == "fixed" else tune(per_file, train_ids, kind, seed)
                    score, n_feat, n_cand = fold_local(per_file, train_ids, test_ids, kind, params, seed)
                    # PLS-DA scores are unbounded regression outputs, thresholded at 0.5 as in pick_peaks_classical.py
                    pred_rows += [{"classifier": f"{kind}_{setting}", "fold": fold, "seed": seed,
                                   "filename": f"{s}.png", "y_true": ppc.label_from_sample_id(s),
                                   "y_prob": p, "y_prob_ens": p} for s, p in zip(test_ids, score)]
                    log_rows.append({"classifier": f"{kind}_{setting}", "seed": seed, "fold": fold,
                                     "n_train": len(train_ids), "n_test": len(test_ids), "n_candidate_bins": n_cand,
                                     "n_features": n_feat, "params": json.dumps(params), "inner_auc": inner_auc})
                    log(f"  [{cohort} {kind}_{setting} seed={seed} fold={fold}] {n_feat} features, {params}")
    preds = pd.DataFrame(pred_rows)
    for clf, g in preds.groupby("classifier"):
        g.drop(columns="classifier").to_csv(out / f"classical_{clf}_{cohort}_predictions.csv", index=False)
    pd.DataFrame(log_rows).to_csv(out / f"classical_folds_{cohort}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mzml", required=True)
    ap.add_argument("--feature-cache", required=True)
    ap.add_argument("--reference-detailed", required=True)
    ap.add_argument("--splits37", required=True)
    ap.add_argument("--splits38", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    per_file = {k: [tuple(t) for t in v] for k, v in json.load(open(args.feature_cache)).items()}
    log(f"feature cache: {len(per_file)} samples")
    load = lambda p: {int(k): {s: set(v) for s, v in d.items()} for k, d in json.load(open(p)).items()}
    splits37, splits38 = load(args.splits37), load(args.splits38)

    # Check: pick_peaks_classical.py's own (global) procedure on the cached features reproduces its results
    X, sids, _ = ppc.build_feature_matrix(per_file, mz_bin_da=MZ_BIN, rt_bin_s=RT_BIN, min_freq=MIN_FREQ)
    rep = pd.concat([ppc.run_cv(X, sids, splits37, classifier=c) for c in ("rf", "plsda")], ignore_index=True)
    old = pd.read_csv(args.reference_detailed)
    m = rep.merge(old, on=["classifier", "seed", "fold"], suffixes=("", "_reference"))
    diff = max((m[c] - m[c + "_reference"]).abs().max() for c in ("accuracy", "roc_auc"))
    pd.DataFrame([{"rows_compared": len(m), "max_abs_difference": diff}]).to_csv(out / "reproduction_check.csv", index=False)
    log(f"reproduction of pick_peaks_classical.py: {len(m)} rows, max |difference| {diff:.2e}")

    # S11's two runs: extract with pick_peaks_classical.py's own function and settings
    for sid in ("S11A", "S11B"):
        if sid not in per_file:
            _, feats, msg = ppc._worker((sid, Path(args.mzml) / f"{sid}.mzML"))
            assert feats is not None, msg
            per_file[sid] = feats
            log(f"  {sid}: {msg}")
    json.dump({k: [list(t) for t in per_file[k]] for k in ("S11A", "S11B")}, open(out / "s11_features.json", "w"))

    run_cohort({k: v for k, v in per_file.items() if not k.startswith("S11")}, splits37, "37", out)
    run_cohort(per_file, splits38, "38", out)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
