"""Paired comparison: CNN headline vs RF / PLS-DA on the same (seed, fold).

Reads:
  results/fullspec_resnet34_512_detailed.csv          (CNN headline)
  results/classical_baselines_detailed.csv            (from HPC)

Writes:
  results/classical_comparison.csv                    (one row per
                                                       baseline x metric)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

import os
REPO_ROOT = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
RES = REPO_ROOT / "results"
CNN = pd.read_csv(RES / "fullspec_resnet34_512_detailed.csv")
CLS = pd.read_csv(RES / "classical_baselines_detailed.csv")

METRICS = ["accuracy", "roc_auc", "precision", "recall", "f1_score"]
N_BOOT = 10000
rng = np.random.default_rng(42)

rows = []
for clf in CLS["classifier"].unique():
    sub = CLS[CLS["classifier"] == clf]
    merged = CNN.merge(sub, on=["seed", "fold"], suffixes=("_cnn", "_cls"))
    assert len(merged) == 18, (
        f"expected 18 (seed,fold) pairs for {clf}, got {len(merged)}"
    )
    for m in METRICS:
        cnn = merged[f"{m}_cnn"].values
        cls = merged[f"{m}_cls"].values
        diff = cls - cnn
        boot = np.empty(N_BOOT)
        n = len(diff)
        for i in range(N_BOOT):
            boot[i] = diff[rng.integers(0, n, n)].mean()
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        try:
            _, w_p = wilcoxon(cls, cnn, zero_method="wilcox")
        except ValueError:
            w_p = float("nan")
        rows.append({
            "classifier": clf,
            "metric": m,
            "cnn_mean": cnn.mean(),
            "classical_mean": cls.mean(),
            "delta": diff.mean(),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "wilcoxon_p": w_p,
        })

out = pd.DataFrame(rows)
out.to_csv(RES / "classical_comparison.csv", index=False)

print("=" * 90)
print("CNN headline vs classical baselines — paired (seed, fold), n=18 per row")
print("=" * 90)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
