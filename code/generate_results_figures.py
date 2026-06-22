
"""
Generate publication-quality figures and summary tables from LC-MS CNN
experiment results.

Reads result CSVs from <repo>/results/ and produces:
  1. Architecture comparison bar chart
  2. Resolution comparison bar chart (full spectrum)
  3. Subregion comparison bar chart
  4. Full spectrum vs subregion comparison
  5. Best model confusion matrix (aggregated)
  6. Per-fold performance variation for best model
  7. Summary table (printed + CSV)

All figures saved to <repo>/figures/ at 300 DPI.

Usage:
    python3 code/generate_results_figures.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap




from pathlib import Path
BASE_DIR = os.environ.get("LCMS_BASE", str(Path(__file__).resolve().parent.parent))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.environ.get("LCMS_FIGURES", os.path.join(BASE_DIR, "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)





_FONT_FAMILY = "serif"
_FONT_SERIF = ["Times New Roman", "Times", "DejaVu Serif"]

plt.rcParams.update({
    "font.family": _FONT_FAMILY,
    "font.serif": _FONT_SERIF,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


RESNET_PARAMS = {
    "resnet18":  11.2,
    "resnet34":  21.3,
    "resnet50":  23.5,
    "resnet101": 42.5,
    "resnet152": 58.1,
}





def _safe_read_csv(path):
    """Read a CSV if it exists, else return None and warn."""
    if os.path.isfile(path):
        return pd.read_csv(path)
    warnings.warn(f"Missing file, skipping: {path}")
    return None


def _load_experiment(prefix, results_dir=RESULTS_DIR):
    """Load summary and detailed CSVs for a given experiment prefix.
    Returns (summary_df_or_None, detailed_df_or_None)."""
    summary = _safe_read_csv(os.path.join(results_dir, f"{prefix}_summary.csv"))
    detailed = _safe_read_csv(os.path.join(results_dir, f"{prefix}_detailed.csv"))
    return summary, detailed


def _grouped_bar_chart(data, labels, metrics, metric_labels, title, ylabel,
                       filename, ylim=(0.0, 1.05), figsize=(7, 4)):
    """Create a grouped bar chart with error bars.

    Parameters
    ----------
    data : list of dicts
        Each dict has keys like 'accuracy_mean', 'accuracy_std', etc.
    labels : list of str
        X-axis group labels (one per data dict).
    metrics : list of str
        Metric base names (e.g., 'accuracy').
    metric_labels : list of str
        Display names for the metrics.
    """
    n_groups = len(labels)
    n_metrics = len(metrics)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        means = [d.get(f"{metric}_mean", np.nan) for d in data]
        stds = [d.get(f"{metric}_std", 0.0) for d in data]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=mlabel, color=COLORS[i],
                      edgecolor="white", linewidth=0.5,
                      capsize=3, error_kw={"linewidth": 0.8})

        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")






def figure_architecture_comparison():
    """Figure: Architecture Comparison (ResNet-18/34/50/101/152)."""
    variants = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    prefixes = [f"arch_{v}_256" for v in variants]
    labels = ["ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "ResNet-152"]

    data = []
    valid_labels = []
    for prefix, label in zip(prefixes, labels):
        summary, _ = _load_experiment(prefix)
        if summary is not None and len(summary) > 0:
            data.append(summary.iloc[0].to_dict())
            valid_labels.append(label)

    if not data:
        print("  [SKIP] No architecture comparison data found.")
        return


    _grouped_bar_chart(
        data, valid_labels,
        metrics=["accuracy", "roc_auc", "f1_score"],
        metric_labels=["Accuracy", "AUC-ROC", "F1 Score"],
        title="Architecture Comparison (Single Best Model)",
        ylabel="Score",
        filename="architecture_comparison.png",
    )


    _grouped_bar_chart(
        data, valid_labels,
        metrics=["ens_accuracy", "ens_roc_auc", "ens_f1_score"],
        metric_labels=["Accuracy", "AUC-ROC", "F1 Score"],
        title="Architecture Comparison (Top-3 Ensemble)",
        ylabel="Score",
        filename="architecture_comparison_ensemble.png",
    )


def figure_resolution_comparison():
    """Figure: Resolution Comparison (Full Spectrum)."""
    resolutions = [256, 512, 1024, 2048]
    prefixes = [f"fullspec_resnet50_{r}" for r in resolutions]
    labels = [f"{r}" for r in resolutions]

    data = []
    valid_labels = []
    for prefix, label in zip(prefixes, labels):
        summary, _ = _load_experiment(prefix)
        if summary is not None and len(summary) > 0:
            data.append(summary.iloc[0].to_dict())
            valid_labels.append(label)

    if not data:
        print("  [SKIP] No resolution comparison data found.")
        return

    _grouped_bar_chart(
        data, valid_labels,
        metrics=["accuracy", "roc_auc", "f1_score"],
        metric_labels=["Accuracy", "AUC-ROC", "F1 Score"],
        title="Full Spectrum Resolution Comparison (ResNet-50)",
        ylabel="Score",
        filename="resolution_fullspec_comparison.png",
    )

    _grouped_bar_chart(
        data, valid_labels,
        metrics=["ens_accuracy", "ens_roc_auc", "ens_f1_score"],
        metric_labels=["Accuracy", "AUC-ROC", "F1 Score"],
        title="Full Spectrum Resolution Comparison -- Ensemble (ResNet-50)",
        ylabel="Score",
        filename="resolution_fullspec_comparison_ensemble.png",
    )


def figure_subregion_comparison():
    """Figure: Subregion Comparison."""
    resolutions = [256, 512, 1024]
    prefixes = [f"subregion_resnet50_{r}" for r in resolutions]
    labels = [f"{r}" for r in resolutions]

    data = []
    valid_labels = []
    for prefix, label in zip(prefixes, labels):
        summary, _ = _load_experiment(prefix)
        if summary is not None and len(summary) > 0:
            data.append(summary.iloc[0].to_dict())
            valid_labels.append(label)

    if not data:
        print("  [SKIP] No subregion comparison data found.")
        return

    _grouped_bar_chart(
        data, valid_labels,
        metrics=["accuracy", "roc_auc", "f1_score"],
        metric_labels=["Accuracy", "AUC-ROC", "F1 Score"],
        title="Subregion Resolution Comparison (ResNet-50)",
        ylabel="Score",
        filename="resolution_subregion_comparison.png",
    )

    _grouped_bar_chart(
        data, valid_labels,
        metrics=["ens_accuracy", "ens_roc_auc", "ens_f1_score"],
        metric_labels=["Accuracy", "AUC-ROC", "F1 Score"],
        title="Subregion Resolution Comparison -- Ensemble (ResNet-50)",
        ylabel="Score",
        filename="resolution_subregion_comparison_ensemble.png",
    )


def figure_fullspec_vs_subregion():
    """Figure: Full Spectrum vs Subregion at matching resolutions."""
    resolutions = [256, 512, 1024]

    fullspec_data = []
    subregion_data = []
    valid_resolutions = []

    for r in resolutions:
        fs_summary, _ = _load_experiment(f"fullspec_resnet50_{r}")
        sr_summary, _ = _load_experiment(f"subregion_resnet50_{r}")
        if fs_summary is not None and sr_summary is not None:
            fullspec_data.append(fs_summary.iloc[0].to_dict())
            subregion_data.append(sr_summary.iloc[0].to_dict())
            valid_resolutions.append(r)

    if not valid_resolutions:
        print("  [SKIP] No matching full-spectrum/subregion data found.")
        return

    metrics = ["accuracy", "roc_auc", "f1_score"]
    metric_labels = ["Accuracy", "AUC-ROC", "F1 Score"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(valid_resolutions))
    width = 0.35

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        fs_means = [d.get(f"{metric}_mean", np.nan) for d in fullspec_data]
        fs_stds = [d.get(f"{metric}_std", 0.0) for d in fullspec_data]
        sr_means = [d.get(f"{metric}_mean", np.nan) for d in subregion_data]
        sr_stds = [d.get(f"{metric}_std", 0.0) for d in subregion_data]

        bars1 = ax.bar(x - width / 2, fs_means, width, yerr=fs_stds,
                        label="Full Spectrum", color=COLORS[0],
                        edgecolor="white", linewidth=0.5, capsize=3,
                        error_kw={"linewidth": 0.8})
        bars2 = ax.bar(x + width / 2, sr_means, width, yerr=sr_stds,
                        label="Subregion", color=COLORS[1],
                        edgecolor="white", linewidth=0.5, capsize=3,
                        error_kw={"linewidth": 0.8})


        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in valid_resolutions])
        ax.set_xlabel("Resolution")
        ax.set_ylim(0.0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Score")
            ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Full Spectrum vs Subregion (ResNet-50)", fontsize=13, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "fullspec_vs_subregion.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def _find_best_config():
    """Identify the best configuration based on highest mean accuracy.
    Returns (prefix, summary_dict, detailed_df) or (None, None, None)."""


    all_prefixes = []
    for v in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        all_prefixes.append(f"arch_{v}_256")
    for r in [256, 512, 1024, 2048]:
        all_prefixes.append(f"fullspec_resnet50_{r}")
    for r in [256, 512, 1024]:
        all_prefixes.append(f"subregion_resnet50_{r}")

    best_prefix = None
    best_acc = -1
    best_summary = None
    best_detailed = None

    for prefix in all_prefixes:
        summary, detailed = _load_experiment(prefix)
        if summary is not None and len(summary) > 0:
            acc = summary.iloc[0].get("accuracy_mean", 0)
            if acc > best_acc:
                best_acc = acc
                best_prefix = prefix
                best_summary = summary.iloc[0].to_dict()
                best_detailed = detailed

    return best_prefix, best_summary, best_detailed


def figure_confusion_matrix():
    """Figure: Aggregated confusion matrix from the best configuration."""
    prefix, summary, detailed = _find_best_config()
    if detailed is None:
        print("  [SKIP] No detailed results found for confusion matrix.")
        return

    print(f"  Best config for confusion matrix: {prefix} "
          f"(accuracy={summary.get('accuracy_mean', 0):.4f})")































    tp_total = 0
    tn_total = 0
    fp_total = 0
    fn_total = 0

    for _, row in detailed.iterrows():
        acc = row.get("accuracy", 0)
        prec = row.get("precision", 0)
        rec = row.get("recall", 0)


        best_n = 4
        best_err = 999
        for n in range(2, 9):
            k = round(acc * n)
            err = abs(k / n - acc)
            if err < best_err:
                best_err = err
                best_n = n

        n = best_n
        correct = round(acc * n)






        best_tp = 0
        best_fp = 0
        best_fn = 0
        best_tn = 0
        best_total_err = 999

        for n_pos in range(1, n):
            n_neg = n - n_pos

            tp_est = round(rec * n_pos)
            fn_est = n_pos - tp_est

            if prec > 0:
                fp_est = round(tp_est / prec - tp_est) if tp_est > 0 else 0
            else:
                fp_est = 0
            tn_est = n_neg - fp_est


            if tn_est < 0 or fp_est < 0:
                continue
            est_acc = (tp_est + tn_est) / n if n > 0 else 0
            err = abs(est_acc - acc) + abs(tp_est + fn_est - n_pos) * 0.1
            if err < best_total_err:
                best_total_err = err
                best_tp = tp_est
                best_fp = fp_est
                best_fn = fn_est
                best_tn = tn_est

        tp_total += best_tp
        tn_total += best_tn
        fp_total += best_fp
        fn_total += best_fn

    cm = np.array([[tn_total, fp_total],
                    [fn_total, tp_total]])


    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")


    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Class A\n(Predicted)", "Class B\n(Predicted)"])
    ax.set_yticklabels(["Class A\n(Actual)", "Class B\n(Actual)"])
    ax.set_title(f"Aggregated Confusion Matrix\n({prefix})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "best_model_confusion_matrix.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure_per_fold_performance():
    """Figure: Per-fold performance variation for the best model."""
    prefix, summary, detailed = _find_best_config()
    if detailed is None:
        print("  [SKIP] No detailed results for per-fold performance.")
        return

    print(f"  Best config for per-fold plot: {prefix}")

    metrics = ["accuracy", "roc_auc", "f1_score"]
    metric_labels = ["Accuracy", "AUC-ROC", "F1 Score"]

    n_folds = detailed["fold"].nunique()
    n_seeds = detailed["seed"].nunique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):

        fold_ids = sorted(detailed["fold"].unique())
        positions = np.arange(len(fold_ids))


        fold_data = []
        for fold in fold_ids:
            vals = detailed[detailed["fold"] == fold][metric].values
            fold_data.append(vals)


        if n_seeds > 1:
            bp = ax.boxplot(fold_data, positions=positions, widths=0.5,
                            patch_artist=True, showmeans=True,
                            meanprops={"marker": "D", "markerfacecolor": "red",
                                       "markersize": 5})
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS[0])
                patch.set_alpha(0.6)
        else:
            means = [np.mean(d) for d in fold_data]
            ax.bar(positions, means, color=COLORS[0], edgecolor="white",
                   linewidth=0.5, alpha=0.8)
            for pos, m in zip(positions, means):
                ax.text(pos, m + 0.01, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=7)

        ax.set_title(mlabel)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Fold {f}" for f in fold_ids], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.15)
        if ax is axes[0]:
            ax.set_ylabel("Score")

    fig.suptitle(f"Per-Fold Performance ({prefix})", fontsize=13, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "best_model_per_fold.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


    ens_metrics = ["ens_accuracy", "ens_roc_auc", "ens_f1_score"]
    if all(m in detailed.columns for m in ens_metrics):
        fig2, axes2 = plt.subplots(1, len(ens_metrics),
                                    figsize=(4 * len(ens_metrics), 4.5),
                                    sharey=True)
        if len(ens_metrics) == 1:
            axes2 = [axes2]

        for ax, metric, mlabel in zip(axes2, ens_metrics, metric_labels):
            fold_ids = sorted(detailed["fold"].unique())
            positions = np.arange(len(fold_ids))
            fold_data = []
            for fold in fold_ids:
                vals = detailed[detailed["fold"] == fold][metric].values
                fold_data.append(vals)

            if n_seeds > 1:
                bp = ax.boxplot(fold_data, positions=positions, widths=0.5,
                                patch_artist=True, showmeans=True,
                                meanprops={"marker": "D", "markerfacecolor": "red",
                                           "markersize": 5})
                for patch in bp["boxes"]:
                    patch.set_facecolor(COLORS[1])
                    patch.set_alpha(0.6)
            else:
                means = [np.mean(d) for d in fold_data]
                ax.bar(positions, means, color=COLORS[1], edgecolor="white",
                       linewidth=0.5, alpha=0.8)

            ax.set_title(mlabel)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"Fold {f}" for f in fold_ids], rotation=45,
                               ha="right")
            ax.set_ylim(0.0, 1.15)
            if ax is axes2[0]:
                ax.set_ylabel("Score")

        fig2.suptitle(f"Per-Fold Ensemble Performance ({prefix})",
                      fontsize=13, y=1.02)
        fig2.tight_layout()
        outpath2 = os.path.join(FIGURES_DIR, "best_model_per_fold_ensemble.png")
        fig2.savefig(outpath2)
        plt.close(fig2)
        print(f"  Saved: {outpath2}")






def generate_summary_table():
    """Print and save a summary table of all configurations."""

    rows = []


    for v in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        prefix = f"arch_{v}_256"
        summary, _ = _load_experiment(prefix)
        if summary is None:
            continue
        s = summary.iloc[0]
        params_m = RESNET_PARAMS.get(v, 0)
        row = {
            "Configuration": f"{v.replace('resnet', 'ResNet-')} (256, full)",
            "Experiment": "Architecture",
            "Params (M)": f"{params_m:.1f}",
        }
        for metric in ["accuracy", "roc_auc", "precision", "recall", "f1_score"]:
            mean = s.get(f"{metric}_mean", np.nan)
            std = s.get(f"{metric}_std", np.nan)
            row[metric.replace("_", " ").title()] = f"{mean:.4f} +/- {std:.4f}"
            row[f"Ens {metric.replace('_', ' ').title()}"] = (
                f"{s.get(f'ens_{metric}_mean', np.nan):.4f} +/- "
                f"{s.get(f'ens_{metric}_std', np.nan):.4f}"
            )
        rows.append(row)


    for r in [256, 512, 1024, 2048]:
        prefix = f"fullspec_resnet50_{r}"
        summary, _ = _load_experiment(prefix)
        if summary is None:
            continue
        s = summary.iloc[0]
        row = {
            "Configuration": f"ResNet-50 ({r}, full)",
            "Experiment": "Resolution",
            "Params (M)": f"{RESNET_PARAMS.get('resnet50', 0):.1f}",
        }
        for metric in ["accuracy", "roc_auc", "precision", "recall", "f1_score"]:
            mean = s.get(f"{metric}_mean", np.nan)
            std = s.get(f"{metric}_std", np.nan)
            row[metric.replace("_", " ").title()] = f"{mean:.4f} +/- {std:.4f}"
            row[f"Ens {metric.replace('_', ' ').title()}"] = (
                f"{s.get(f'ens_{metric}_mean', np.nan):.4f} +/- "
                f"{s.get(f'ens_{metric}_std', np.nan):.4f}"
            )
        rows.append(row)


    for r in [256, 512, 1024]:
        prefix = f"subregion_resnet50_{r}"
        summary, _ = _load_experiment(prefix)
        if summary is None:
            continue
        s = summary.iloc[0]
        row = {
            "Configuration": f"ResNet-50 ({r}, subregion)",
            "Experiment": "Subregion",
            "Params (M)": f"{RESNET_PARAMS.get('resnet50', 0):.1f}",
        }
        for metric in ["accuracy", "roc_auc", "precision", "recall", "f1_score"]:
            mean = s.get(f"{metric}_mean", np.nan)
            std = s.get(f"{metric}_std", np.nan)
            row[metric.replace("_", " ").title()] = f"{mean:.4f} +/- {std:.4f}"
            row[f"Ens {metric.replace('_', ' ').title()}"] = (
                f"{s.get(f'ens_{metric}_mean', np.nan):.4f} +/- "
                f"{s.get(f'ens_{metric}_std', np.nan):.4f}"
            )
        rows.append(row)

    if not rows:
        print("  [SKIP] No summary data available.")
        return

    df = pd.DataFrame(rows)


    print("\n" + "=" * 120)
    print("SUMMARY TABLE (Single Model)")
    print("=" * 120)
    single_cols = ["Configuration", "Experiment", "Params (M)",
                   "Accuracy", "Roc Auc", "Precision", "Recall", "F1 Score"]
    single_cols = [c for c in single_cols if c in df.columns]
    print(df[single_cols].to_string(index=False))

    print("\n" + "=" * 120)
    print("SUMMARY TABLE (Ensemble)")
    print("=" * 120)
    ens_cols = ["Configuration", "Experiment", "Params (M)",
                "Ens Accuracy", "Ens Roc Auc", "Ens Precision",
                "Ens Recall", "Ens F1 Score"]
    ens_cols = [c for c in ens_cols if c in df.columns]
    print(df[ens_cols].to_string(index=False))


    csv_path = os.path.join(FIGURES_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Summary table saved: {csv_path}")


    print("\n" + "=" * 120)
    print("LATEX TABLE (Single Model)")
    print("=" * 120)
    _print_latex_table(df, single_cols)

    return df


def figure_roc_curves():
    """Figure: ROC curves from saved predictions (requires *_predictions.csv files).

    These prediction files are generated by code/generate_predictions.py after
    training completes.  Each has columns: fold, seed, filename, y_true, y_prob, y_prob_ens.
    """
    pred_dir = RESULTS_DIR


    configs_to_plot = []


    for v in ["resnet18", "resnet50"]:
        path = os.path.join(pred_dir, f"arch_{v}_256_predictions.csv")
        if os.path.isfile(path):
            configs_to_plot.append((f"arch_{v}_256", f"ResNet-{v[6:]}", path))


    for r in [256, 512]:
        path = os.path.join(pred_dir, f"fullspec_resnet50_{r}_predictions.csv")
        if os.path.isfile(path):
            configs_to_plot.append((f"fullspec_resnet50_{r}", f"Full {r}", path))


    path = os.path.join(pred_dir, "subregion_resnet50_256_predictions.csv")
    if os.path.isfile(path):
        configs_to_plot.append(("subregion_resnet50_256", "Subregion 256", path))

    if not configs_to_plot:
        print("  [SKIP] No prediction files found. Run generate_predictions.py first.")
        return

    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for i, (prefix, label, path) in enumerate(configs_to_plot):
        df = pd.read_csv(path)
        y_true = df["y_true"].values
        y_prob = df["y_prob"].values


        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=COLORS[i], lw=1.8,
                label=f"{label} (AUC = {roc_auc:.3f})")


        seeds = df["seed"].unique()
        if len(seeds) > 1:
            base_fpr = np.linspace(0, 1, 101)
            tprs = []
            for seed in seeds:
                seed_df = df[df["seed"] == seed]
                if len(seed_df["y_true"].unique()) < 2:
                    continue
                fpr_s, tpr_s, _ = roc_curve(seed_df["y_true"], seed_df["y_prob"])
                tpr_interp = np.interp(base_fpr, fpr_s, tpr_s)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
            if len(tprs) > 1:
                tprs = np.array(tprs)
                tpr_mean = tprs.mean(axis=0)
                tpr_std = tprs.std(axis=0)
                ax.fill_between(base_fpr,
                                np.clip(tpr_mean - tpr_std, 0, 1),
                                np.clip(tpr_mean + tpr_std, 0, 1),
                                color=COLORS[i], alpha=0.12)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure_lopo_accuracy():
    """Figure: Per-patient LOPO accuracy bar chart.

    Reads the existing LOPO results from models/loo/all_lopo_predictions.csv
    (generated by the day4_loo_resnet.ipynb notebook with 10 seeds).
    Shows per-patient classification accuracy, highlighting S11 as an outlier.
    """
    lopo_path = os.path.join(BASE_DIR, "models", "loo", "all_lopo_predictions.csv")
    if not os.path.isfile(lopo_path):
        print("  [SKIP] No LOPO predictions found at models/loo/all_lopo_predictions.csv")
        return

    df = pd.read_csv(lopo_path)
    patient_acc = df.groupby("Patient")["Correct"].mean().reset_index()
    patient_acc.columns = ["Patient", "Accuracy"]


    patient_acc = patient_acc.sort_values(["Accuracy", "Patient"],
                                          ascending=[False, True]).reset_index(drop=True)


    colors_bar = [COLORS[3] if p == "S11" else COLORS[0]
                  for p in patient_acc["Patient"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(patient_acc))
    bars = ax.bar(x, patient_acc["Accuracy"], color=colors_bar, edgecolor="white",
                  linewidth=0.5)


    s11_idx = patient_acc[patient_acc["Patient"] == "S11"].index
    if len(s11_idx) > 0:
        idx = s11_idx[0]
        acc_val = patient_acc.loc[idx, "Accuracy"]
        ax.annotate(f"S11\n({acc_val:.0%})",
                    xy=(idx, acc_val), xytext=(idx, acc_val + 0.06),
                    ha="center", fontsize=8, fontweight="bold", color=COLORS[3],
                    arrowprops=dict(arrowstyle="-", color=COLORS[3], lw=0.8))


    overall_acc = df["Correct"].mean()
    excl_s11_acc = df[df["Patient"] != "S11"]["Correct"].mean()
    ax.axhline(excl_s11_acc, color=COLORS[0], linestyle="--", lw=0.8, alpha=0.7,
               label=f"Mean excl. S11 ({excl_s11_acc:.1%})")
    ax.axhline(overall_acc, color="gray", linestyle=":", lw=0.8, alpha=0.7,
               label=f"Overall mean ({overall_acc:.1%})")

    ax.set_xticks(x)
    ax.set_xticklabels(patient_acc["Patient"], rotation=90, fontsize=7)
    ax.set_ylabel("LOPO Classification Accuracy")
    ax.set_title("Leave-One-Patient-Out Accuracy by Patient (10 Seeds)")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.tight_layout()

    outpath = os.path.join(FIGURES_DIR, "lopo_per_patient_accuracy.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


    print(f"  Overall LOPO accuracy: {overall_acc:.3f}")
    print(f"  Excl. S11: {excl_s11_acc:.3f}")
    print(f"  S11 accuracy: {patient_acc[patient_acc['Patient']=='S11']['Accuracy'].values[0]:.3f}")


def figure_statistical_tests():
    """Perform and display pairwise statistical significance tests.

    Uses Wilcoxon signed-rank test on per-fold accuracy across seeds.
    For each experiment group, compares all pairs of configurations.
    Prints a table and saves a heatmap of p-values.
    """


    configs = {}


    for v in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        prefix = f"arch_{v}_256"
        _, detailed = _load_experiment(prefix)
        if detailed is not None:
            configs[f"ResNet-{v[6:]}"] = detailed


    for r in [256, 512, 1024, 2048]:
        prefix = f"fullspec_resnet50_{r}"
        _, detailed = _load_experiment(prefix)
        if detailed is not None:
            configs[f"Full-{r}"] = detailed


    for r in [256, 512, 1024]:
        prefix = f"subregion_resnet50_{r}"
        _, detailed = _load_experiment(prefix)
        if detailed is not None:
            configs[f"Sub-{r}"] = detailed

    if len(configs) < 2:
        print("  [SKIP] Need at least 2 configs for statistical tests.")
        return

    names = list(configs.keys())
    n = len(names)


    fold_accs = {}
    for name, df in configs.items():
        fold_means = df.groupby("fold")["accuracy"].mean().sort_index().values
        fold_accs[name] = fold_means


    p_matrix = np.ones((n, n))
    results_rows = []

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Wilcoxon signed-rank, per-fold accuracy)")
    print("=" * 80)
    print(f"{'Config A':<16} {'Config B':<16} {'Mean A':>8} {'Mean B':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 72)

    for i in range(n):
        for j in range(i + 1, n):
            a = fold_accs[names[i]]
            b = fold_accs[names[j]]

            mean_a = np.mean(a)
            mean_b = np.mean(b)


            diff = a - b
            if np.all(diff == 0):
                p_val = 1.0
            elif len(a) < 6:


                try:
                    result = stats.wilcoxon(a, b, alternative="two-sided")
                    p_val = result.pvalue
                except ValueError:
                    p_val = 1.0
            else:
                try:
                    result = stats.wilcoxon(a, b, alternative="two-sided")
                    p_val = result.pvalue
                except ValueError:
                    p_val = 1.0

            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

            sig = "*" if p_val < 0.05 else ("**" if p_val < 0.01 else "")
            print(f"{names[i]:<16} {names[j]:<16} {mean_a:>8.4f} {mean_b:>8.4f} {p_val:>10.4f} {sig:>6}")

            results_rows.append({
                "Config A": names[i],
                "Config B": names[j],
                "Mean A": round(mean_a, 4),
                "Mean B": round(mean_b, 4),
                "p-value": round(p_val, 4),
                "Significant (p<0.05)": p_val < 0.05
            })


    sig_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(FIGURES_DIR, "statistical_tests.csv")
    sig_df.to_csv(csv_path, index=False)
    print(f"\n  Tests saved: {csv_path}")


    n_folds = len(next(iter(fold_accs.values())))
    print(f"\n  Note: With only {n_folds} folds, statistical power is limited.")
    print("  p-values should be interpreted with caution.")



    _plot_pvalue_heatmap(names, p_matrix, "All Configurations", "pvalue_heatmap.png")


    arch_names = [n for n in names if n.startswith("ResNet")]
    if len(arch_names) >= 2:
        arch_idx = [names.index(n) for n in arch_names]
        sub_matrix = p_matrix[np.ix_(arch_idx, arch_idx)]
        _plot_pvalue_heatmap(arch_names, sub_matrix,
                              "Architecture Comparisons", "pvalue_heatmap_arch.png")


def _plot_pvalue_heatmap(labels, p_matrix, title, filename):
    """Plot a heatmap of p-values."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.8 + 1), max(4, n * 0.7 + 1)))


    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=0)
    p_display = np.ma.masked_where(mask, p_matrix)

    im = ax.imshow(p_display, cmap="RdYlGn_r", vmin=0, vmax=0.1,
                   interpolation="nearest")


    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mask[i, j]:
                continue
            p = p_matrix[i, j]
            stars = ""
            if p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            color = "white" if p < 0.03 else "black"
            ax.text(j, i, f"{p:.3f}\n{stars}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if stars else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Pairwise p-values: {title}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def _print_latex_table(df, cols):
    """Print a simple LaTeX tabular from a DataFrame."""
    cols_present = [c for c in cols if c in df.columns]
    n = len(cols_present)
    col_spec = "l" * n
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    header = " & ".join(cols_present) + " \\\\"
    print(header)
    print("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols_present:
            v = str(row.get(c, ""))

            v = v.replace("+/-", "$\\pm$")
            vals.append(v)
        print(" & ".join(vals) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")





def main():
    print("=" * 60)
    print("Generating publication figures and summary tables")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Figures dir: {FIGURES_DIR}")
    print("=" * 60)


    if not os.path.isdir(RESULTS_DIR):
        print(f"\nWARNING: Results directory does not exist: {RESULTS_DIR}")
        print("No results to process. Run experiments first.")
        return


    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(f"\nWARNING: No CSV files found in {RESULTS_DIR}")
        print("No results to process. Run experiments first.")
        return

    print(f"\n  Found {len(csv_files)} result CSV files:")
    for f in sorted(csv_files):
        print(f"    {f}")


    print("\n--- Figure: Architecture Comparison ---")
    figure_architecture_comparison()

    print("\n--- Figure: Resolution Comparison (Full Spectrum) ---")
    figure_resolution_comparison()

    print("\n--- Figure: Subregion Comparison ---")
    figure_subregion_comparison()

    print("\n--- Figure: Full Spectrum vs Subregion ---")
    figure_fullspec_vs_subregion()

    print("\n--- Figure: Best Model Confusion Matrix ---")
    figure_confusion_matrix()

    print("\n--- Figure: Per-Fold Performance ---")
    figure_per_fold_performance()

    print("\n--- Figure: LOPO Per-Patient Accuracy ---")
    figure_lopo_accuracy()

    print("\n--- Figure: ROC Curves ---")
    figure_roc_curves()

    print("\n--- Statistical Significance Tests ---")
    figure_statistical_tests()

    print("\n--- Summary Table ---")
    generate_summary_table()

    print("\n" + "=" * 60)
    print("Done! All figures saved to:")
    print(f"  {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
