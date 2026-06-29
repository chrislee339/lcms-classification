"""
Bootstrap CIs and Holm-corrected Wilcoxon p-values
on paired ablation differences (architecture and resolution).

For each pair of configurations, treats fold-level metrics paired by
(seed, fold) and computes:
  - mean paired difference (and SD)
  - bootstrap 95% CI for the paired mean difference (BCa, 10,000 resamples)
  - Wilcoxon signed-rank statistic & raw p-value
  - Holm-corrected p-value across all comparisons in a family (architecture
    pairs and resolution pairs are corrected separately)

Outputs CSVs to results/ for the
sensitivity-analysis tables.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import os
REPO_ROOT = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parent.parent))
RESULTS_DIR = REPO_ROOT / "results"

ARCH_CONFIGS = [
    ("ResNet-18",  "arch_resnet18_256"),
    ("ResNet-34",  "arch_resnet34_256"),
    ("ResNet-50",  "arch_resnet50_256"),
    ("ResNet-101", "arch_resnet101_256"),
    ("ResNet-152", "arch_resnet152_256"),
]
RES_CONFIGS = [
    ("Full-256",  "fullspec_resnet34_256"),
    ("Full-512",  "fullspec_resnet34_512"),
    ("Full-1024", "fullspec_resnet34_1024"),
]
SUB_CONFIGS = [
    ("Sub-256",  "subregion_resnet34_256"),
    ("Sub-512",  "subregion_resnet34_512"),
    ("Sub-1024", "subregion_resnet34_1024"),
]


def load_detailed(key: str) -> pd.DataFrame:
    p = RESULTS_DIR / f"{key}_detailed.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def paired_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> np.ndarray:
    """Align by (seed, fold) and return (a_metric - b_metric) paired diffs."""
    keys = ["seed", "fold"]
    merged = df_a[keys + [metric]].merge(
        df_b[keys + [metric]], on=keys, suffixes=("_a", "_b")
    )
    return merged[f"{metric}_a"].values - merged[f"{metric}_b"].values


def bootstrap_mean_ci(diffs: np.ndarray, n_boot: int = 10_000,
                      alpha: float = 0.05, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = min(1.0, (n - rank) * p[idx])
        if adj_p < running_max:
            adj_p = running_max
        else:
            running_max = adj_p
        adj[idx] = adj_p
    return adj.tolist()


def run_family(configs: list[tuple[str, str]], metric: str,
               family_name: str) -> pd.DataFrame:
    """Run all pairwise comparisons within a family for one metric."""
    detailed = {}
    for label, key in configs:
        df = load_detailed(key)
        if df is None:
            print(f"  WARNING: missing detailed CSV for {key}; skipping")
            continue
        detailed[label] = df

    rows = []
    raw_ps = []
    pair_index = []
    for (la, kla), (lb, klb) in combinations(configs, 2):
        if la not in detailed or lb not in detailed:
            continue
        diffs = paired_metric(detailed[la], detailed[lb], metric)
        if len(diffs) == 0:
            continue
        mean_d = float(diffs.mean())
        sd_d = float(diffs.std(ddof=1))
        lo, hi = bootstrap_mean_ci(diffs)
        try:
            stat, pval = stats.wilcoxon(diffs, zero_method="wilcox",
                                        alternative="two-sided")
            pval = float(pval)
        except ValueError:
            stat, pval = float("nan"), 1.0
        rows.append({
            "family": family_name,
            "metric": metric,
            "config_a": la,
            "config_b": lb,
            "n_pairs": len(diffs),
            "mean_diff": mean_d,
            "sd_diff": sd_d,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "wilcoxon_p_raw": pval,
        })
        raw_ps.append(pval)
        pair_index.append(len(rows) - 1)

    if raw_ps:
        adj = holm_correct(raw_ps)
        for i, idx in enumerate(pair_index):
            rows[idx]["wilcoxon_p_holm"] = adj[i]
    return pd.DataFrame(rows)


def main():
    out_frames = []
    for family_name, configs in [
        ("architecture", ARCH_CONFIGS),
        ("resolution_full", RES_CONFIGS),
        ("resolution_sub", SUB_CONFIGS),
    ]:
        for metric in ["accuracy", "roc_auc", "f1_score", "recall"]:
            df = run_family(configs, metric, family_name)
            if not df.empty:
                out_frames.append(df)
    full = pd.concat(out_frames, ignore_index=True)

    out_csv = RESULTS_DIR / "ablation_bootstrap_holm.csv"
    full.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(full)} rows).")


    for fam in full["family"].unique():
        sub = full[(full["family"] == fam) & (full["metric"] == "accuracy")]
        print(f"\n=== {fam} (accuracy) ===")
        cols = ["config_a", "config_b", "mean_diff", "ci95_lo", "ci95_hi",
                "wilcoxon_p_raw", "wilcoxon_p_holm"]
        view = sub[cols].copy()
        for c in ["mean_diff", "ci95_lo", "ci95_hi"]:
            view[c] = view[c].round(4)
        for c in ["wilcoxon_p_raw", "wilcoxon_p_holm"]:
            view[c] = view[c].round(3)
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
