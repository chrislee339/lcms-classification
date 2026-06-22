"""Paired (seed, fold) comparison: per-sample-norm vs global-norm.

Output:
  results/globalnorm_comparison.csv  — one row per metric with both means,
                                        bootstrap CI on Δ, and Wilcoxon p.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

import os
REPO_ROOT = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
RES = REPO_ROOT / "results"
BASE = pd.read_csv(RES / "fullspec_resnet34_512_detailed.csv")
GLOB = pd.read_csv(RES / "fullspec_resnet34_512_globalnorm_detailed.csv")

merged = BASE.merge(GLOB, on=["seed", "fold"], suffixes=("_base", "_glob"))
assert len(merged) == 18, f"expected 18 (seed,fold) pairs, got {len(merged)}"

METRICS = ["accuracy", "roc_auc", "precision", "recall", "f1_score",
           "ens_accuracy", "ens_roc_auc"]
N_BOOT = 10000
rng = np.random.default_rng(42)

rows = []
for m in METRICS:
    base = merged[f"{m}_base"].values
    glob = merged[f"{m}_glob"].values
    diff = glob - base
    boot = np.empty(N_BOOT)
    n = len(diff)
    for i in range(N_BOOT):
        boot[i] = diff[rng.integers(0, n, n)].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    try:
        w_stat, w_p = wilcoxon(glob, base, zero_method="wilcox")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    rows.append({
        "metric": m,
        "baseline_mean": base.mean(),
        "globalnorm_mean": glob.mean(),
        "delta": diff.mean(),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "wilcoxon_p": w_p,
    })

out = pd.DataFrame(rows)
out.to_csv(RES / "globalnorm_comparison.csv", index=False)

print("=" * 80)
print("Per-sample (baseline) vs global-norm head-to-head — paired (seed, fold)")
print("=" * 80)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nN pairs: {len(merged)}")
