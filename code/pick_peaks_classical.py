"""Classical metabolomics baseline: peak picking + RF/PLS-DA.

Pipeline:
  1. Per-file FeatureFindingMetabo (pyOpenMS) extracts (m/z, RT, intensity)
     features. Parallelized over files.
  2. Pool features into a global (m/z, RT) grid for cross-sample alignment
     (0.01 Da m/z, 5 s RT bins by default). Each (m/z bin, RT bin) becomes a
     candidate feature; bins seen in fewer than `min_freq` of samples are
     dropped.
  3. Log10(intensity + 1) transform.
  4. Patient-stratified 6-fold CV using `patient_splits.json` (same splits
     as CNN headline). 3 seeds x 6 folds x 2 classifiers = 36 fits.
  5. Random Forest and PLS-DA classifiers (sklearn).
  6. Outputs:
       <out>/feature_matrix.csv
       <out>/classical_baselines_detailed.csv  (one row per (clf, seed, fold))
       <out>/classical_baselines_summary.csv   (mean+/-std per classifier)
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EXCLUDE_FILES = {"Fetuin_final", "S2B-5x-0302"}
EXCLUDE_PATIENTS = {"S11"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_features_one(mzml_path, noise_threshold=1000.0, mass_error_ppm=10.0):
    """Run FeatureFindingMetabo on one mzML; return list of (mz, rt, intensity)."""
    from pyopenms import (
        MSExperiment, MzMLFile, MassTraceDetection, ElutionPeakDetection,
        FeatureFindingMetabo, FeatureMap,
    )

    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)
    exp.sortSpectra(True)

    mtd = MassTraceDetection()
    p = mtd.getDefaults()
    p.setValue("noise_threshold_int", float(noise_threshold))
    p.setValue("mass_error_ppm", float(mass_error_ppm))
    p.setValue("reestimate_mt_sd", "true")
    mtd.setParameters(p)
    mass_traces = []
    mtd.run(exp, mass_traces, 0)

    epd = ElutionPeakDetection()
    p = epd.getDefaults()
    p.setValue("width_filtering", "fixed")
    epd.setParameters(p)
    mass_traces_split = []
    epd.detectPeaks(mass_traces, mass_traces_split)

    ffm = FeatureFindingMetabo()
    p = ffm.getDefaults()
    p.setValue("isotope_filtering_model", "none")
    p.setValue("remove_single_traces", "true")
    p.setValue("mz_scoring_by_elements", "false")
    p.setValue("report_convex_hulls", "false")
    ffm.setParameters(p)
    features = FeatureMap()
    chromatograms = []
    ffm.run(mass_traces_split, features, chromatograms)

    return [(f.getMZ(), f.getRT(), f.getIntensity()) for f in features]


def _worker(args):
    sample_id, path = args
    t0 = time.time()
    try:
        feats = extract_features_one(path)
    except Exception as e:
        return sample_id, None, f"ERROR: {e}"
    return sample_id, feats, f"{len(feats)} features in {time.time()-t0:.1f}s"


def build_feature_matrix(per_file_features, mz_bin_da=0.01, rt_bin_s=5.0,
                         min_freq=0.2):
    """Pool features into a global (mz, rt) grid; build samples x features matrix.

    Returns (X, sample_ids, feature_keys, intensity_summary).
    """
    sample_ids = sorted(per_file_features.keys())
    n = len(sample_ids)
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    bins = {}
    for sid in sample_ids:
        for mz, rt, intensity in per_file_features[sid]:
            key = (int(round(mz / mz_bin_da)), int(round(rt / rt_bin_s)))
            d = bins.setdefault(key, {})
            d[sid] = max(d.get(sid, 0.0), float(intensity))

    min_n = max(2, int(min_freq * n))
    feature_keys = sorted(k for k, sd in bins.items() if len(sd) >= min_n)
    log(f"  Total candidate bins: {len(bins):,}; "
        f"after >= {min_n}/{n}-sample filter: {len(feature_keys):,}")

    X = np.zeros((n, len(feature_keys)), dtype=np.float32)
    for j, k in enumerate(feature_keys):
        for sid, val in bins[k].items():
            X[sid_to_idx[sid], j] = val


    X = np.log10(X + 1.0)
    return X, sample_ids, feature_keys


def label_from_sample_id(sid):
    return 1 if sid.endswith("B") else 0


def patient_from_sample_id(sid):
    return sid[:-1]


def run_cv(X, sample_ids, splits, classifier, seeds=(42, 69, 137)):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score,
    )
    from sklearn.preprocessing import StandardScaler

    rows = []
    y = np.array([label_from_sample_id(s) for s in sample_ids])
    patients = np.array([patient_from_sample_id(s) for s in sample_ids])

    for seed in seeds:
        for fold in range(6):
            train_pat = splits[fold]["train"] | splits[fold]["val"]
            test_pat = splits[fold]["test"]

            train_mask = np.array([p in train_pat for p in patients])
            test_mask = np.array([p in test_pat for p in patients])

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask], y[test_mask]

            if classifier == "rf":
                clf = RandomForestClassifier(
                    n_estimators=500, max_features="sqrt",
                    random_state=seed, n_jobs=-1,
                )
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                y_score = clf.predict_proba(X_te)[:, 1]
            elif classifier == "plsda":

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                n_comp = min(5, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
                pls = PLSRegression(n_components=n_comp, scale=False)
                pls.fit(X_tr_s, y_tr.astype(np.float64))
                y_score = pls.predict(X_te_s).ravel()
                y_pred = (y_score > 0.5).astype(int)
            else:
                raise ValueError(classifier)

            rows.append({
                "classifier": classifier,
                "seed": seed,
                "fold": fold,
                "accuracy": accuracy_score(y_te, y_pred),
                "roc_auc": (roc_auc_score(y_te, y_score)
                            if len(set(y_te.tolist())) > 1 else float("nan")),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1_score": f1_score(y_te, y_pred, zero_division=0),
                "n_test": int(test_mask.sum()),
            })
            log(f"    [{classifier} seed={seed} fold={fold}] "
                f"acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['roc_auc']:.3f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mzml", required=True,
                    help="Directory with mzML files (flat or classA/classB subdirs)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--patient-splits", required=True,
                    help="patient_splits.json (S11-excluded)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--feature-cache", default=None)
    ap.add_argument("--mz-bin-da", type=float, default=0.01)
    ap.add_argument("--rt-bin-s", type=float, default=5.0)
    ap.add_argument("--min-freq", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, 16))


    mzml_dir = Path(args.mzml)
    candidates = []
    for sub in (mzml_dir, mzml_dir / "classA", mzml_dir / "classB"):
        if sub.is_dir():
            candidates.extend(sub.glob("*.mzML"))
    samples = {}
    for p in candidates:
        sid = p.stem
        if sid in EXCLUDE_FILES:
            continue
        if "-" in sid:
            continue
        patient = patient_from_sample_id(sid)
        if patient in EXCLUDE_PATIENTS:
            continue
        samples[sid] = p
    log(f"Found {len(samples)} mzML files (S11 excluded)")
    if not samples:
        raise SystemExit(f"No mzML files found under {mzml_dir}")


    cache_path = args.feature_cache and Path(args.feature_cache)
    per_file = None
    if cache_path and cache_path.exists():
        with cache_path.open() as f:
            raw = json.load(f)
        per_file = {k: [tuple(t) for t in v] for k, v in raw.items()}

        if set(per_file.keys()) >= set(samples.keys()):
            per_file = {k: per_file[k] for k in samples.keys()}
            log(f"Loaded cached features for {len(per_file)} samples from {cache_path}")
        else:
            log(f"Cache incomplete; re-extracting")
            per_file = None

    if per_file is None:
        log(f"Extracting features ({workers} workers)...")
        with Pool(workers) as pool:
            per_file = {}
            for i, (sid, feats, msg) in enumerate(
                    pool.imap_unordered(_worker, list(samples.items())), 1):
                if feats is None:
                    log(f"  [{i}/{len(samples)}] {sid}: {msg}")
                    continue
                per_file[sid] = feats
                log(f"  [{i}/{len(samples)}] {sid}: {msg}")
        if cache_path:
            with cache_path.open("w") as f:
                json.dump({k: [list(t) for t in v] for k, v in per_file.items()}, f)
            log(f"Cached features to {cache_path}")


    log("Building feature matrix...")
    X, sample_ids, feature_keys = build_feature_matrix(
        per_file, mz_bin_da=args.mz_bin_da, rt_bin_s=args.rt_bin_s,
        min_freq=args.min_freq,
    )
    log(f"  Feature matrix shape: {X.shape}")
    fm_df = pd.DataFrame(X, index=sample_ids,
                         columns=[f"mz{k[0]*args.mz_bin_da:.4f}_rt{k[1]*args.rt_bin_s:.1f}"
                                  for k in feature_keys])
    fm_df.to_csv(out_dir / "feature_matrix.csv")
    log(f"  Wrote {out_dir/'feature_matrix.csv'}")


    with open(args.patient_splits) as f:
        raw = json.load(f)
    splits = {int(k): {s: set(v) for s, v in d.items()} for k, d in raw.items()}


    log("Running RF + PLS-DA over 3 seeds x 6 folds...")
    rf_df = run_cv(X, sample_ids, splits, classifier="rf")
    pls_df = run_cv(X, sample_ids, splits, classifier="plsda")
    detailed = pd.concat([rf_df, pls_df], ignore_index=True)
    detailed.to_csv(out_dir / "classical_baselines_detailed.csv", index=False)
    log(f"  Wrote {out_dir/'classical_baselines_detailed.csv'}")


    summary_rows = []
    for clf, sub in detailed.groupby("classifier"):
        row = {"classifier": clf, "n_obs": len(sub)}
        for metric in ("accuracy", "roc_auc", "precision", "recall", "f1_score"):
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "classical_baselines_summary.csv", index=False)

    log("\n=== Classical baselines summary ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
