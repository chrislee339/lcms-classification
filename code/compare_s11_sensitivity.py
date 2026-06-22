"""Build the S11 sensitivity comparison table for Supplementary Table S5.

For each configuration that was retrained with S11 included, pair the
fold-level metrics by (seed, fold) and report:
  - mean accuracy / AUC under both cohorts
  - paired mean difference (S11-included minus S11-excluded)
  - 95% bootstrap CI on the paired difference
  - Wilcoxon signed-rank p-value (raw)

Output: results/s11_sensitivity_comparison.csv
"""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import os
REPO_ROOT = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
RESULTS = REPO_ROOT / "results"

PAIRS = [
    ("Headline (ResNet-34 / 512x512)", "fullspec_resnet34_512", "fullspec_resnet34_512_withS11"),
    ("ResNet-18 / 256x256",  "arch_resnet18_256",  "arch_resnet18_256_withS11"),
    ("ResNet-34 / 256x256",  "arch_resnet34_256",  "arch_resnet34_256_withS11"),
    ("ResNet-50 / 256x256",  "arch_resnet50_256",  "arch_resnet50_256_withS11"),
    ("ResNet-101 / 256x256", "arch_resnet101_256", "arch_resnet101_256_withS11"),
    ("ResNet-152 / 256x256", "arch_resnet152_256", "arch_resnet152_256_withS11"),
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}_detailed.csv")


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> None:
    rows = []
    for label, excl, incl in PAIRS:
        try:
            de = load(excl)
            di = load(incl)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")
            continue

        for metric in ["accuracy", "roc_auc"]:
            merged = de[["seed", "fold", metric]].merge(
                di[["seed", "fold", metric]],
                on=["seed", "fold"],
                suffixes=("_excl", "_incl"),
            )
            diffs = merged[f"{metric}_incl"].values - merged[f"{metric}_excl"].values
            mean_excl = float(merged[f"{metric}_excl"].mean())
            mean_incl = float(merged[f"{metric}_incl"].mean())
            mean_d = float(diffs.mean())
            lo, hi = bootstrap_ci(diffs)
            try:
                p = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
            except ValueError:
                p = float("nan")
            rows.append({
                "configuration": label,
                "metric": metric,
                "mean_S11_excluded": round(mean_excl, 4),
                "mean_S11_included": round(mean_incl, 4),
                "mean_diff_incl_minus_excl": round(mean_d, 4),
                "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
                "wilcoxon_p": round(p, 4) if not np.isnan(p) else float("nan"),
                "n_pairs": len(diffs),
            })

    out = pd.DataFrame(rows)
    out_path = RESULTS / "s11_sensitivity_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 12)
    print(out.to_string(index=False))


    print("\nArchitecture ranking (mean accuracy across seeds and folds):")
    for cohort, suffix in [("S11-excluded", ""), ("S11-included", "_withS11")]:
        ranks = []
        for label, _, _ in PAIRS[1:]:
            arch = label.split(" /")[0]
            name = f"arch_{arch.lower().replace('-','')}_256{suffix}"
            try:
                df = load(name)
                ranks.append((arch, float(df["accuracy"].mean())))
            except FileNotFoundError:
                pass
        ranks.sort(key=lambda r: -r[1])
        line = "  " + cohort + ": " + " > ".join(
            f"{a}({m:.3f})" for a, m in ranks)
        print(line)


if __name__ == "__main__":
    main()
