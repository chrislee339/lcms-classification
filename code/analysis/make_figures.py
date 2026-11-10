"""Draw the paper's figures from the 38-patient results (S11 always in a training set).

Plotting only: no training, no new statistics beyond per-fold metrics, confusion counts and ROC curves recomputed
from the saved out-of-fold predictions.

Main text:  fig2_pipeline, fig3_gradcam_means, fig4_occlusion, fig5_roc_*, fig6_configurations_*
Supplement: figS1_lopo, figS2_s16_gradcam, figS3_per_fold_*, figS4_confusion_*
(Figure 1 is not drawn here.)
Figures showing one CNN come in two variants: *_nested (nested cross-validation predictions) and *_full512
(full spectrum ResNet-34 512 x 512); fig6_configurations_nested marks the nested estimate instead of full 512.

Inputs (INPUTS below, under $LCMS_BASE, default: the repository root): the outputs of evaluate_checkpoints.py,
nested_result.py, single_model_stats.py, occlusion.py and gradcam_stability.py under results/analysis/, the printed
output of occlusion.py in logs/occlusion.txt, results/lopo_predictions.csv, data/mzml/S33A.mzML and the images in
data/raw/spectra_filter_*. Writes PNG and PDF files and manifest.json (input hashes and plotted numbers) to <out dir>.

usage: python make_figures.py <out dir> [figure name ...]
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

BASE = Path(os.environ.get("LCMS_BASE", Path(__file__).resolve().parents[2]))
INPUTS = {
    "predictions": BASE / "results/analysis/predictions/fullspec_resnet34_512_withS11rot_predictions.csv",
    "nested_predictions": BASE / "results/analysis/stats/nested/nested_withS11rot_predictions.csv",
    "nested_single": BASE / "results/analysis/stats/nested/single_nested_withS11rot.csv",
    "nested_picks": BASE / "results/analysis/stats/nested/picks.csv",
    "single38": BASE / "results/analysis/stats/single38rot.csv",
    "single_custom": BASE / "results/analysis/stats/single_custom_cnn38rot.csv",
    "occlusion_stats": BASE / "results/analysis/occlusion/occlusion_stats.csv",
    "occlusion_log": BASE / "logs/occlusion.txt",
    "mean_maps": BASE / "results/analysis/gradcam/mean_maps.npz",
    "maps": BASE / "results/analysis/gradcam/maps.npz",
    "maps_meta": BASE / "results/analysis/gradcam/maps_meta.csv",
    "lopo": BASE / "results/lopo_predictions.csv",
    "mzml_s33a": BASE / "data/mzml/S33A.mzML",
    "img_dir": BASE / "data/raw",
}
MM = 1 / 25.4
OKABE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
         "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
SEED_STYLE = {42: (OKABE["blue"], "o"), 69: (OKABE["orange"], "s"), 137: (OKABE["green"], "^")}
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
})
OUT = Path(sys.argv[1])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def letter_title(ax, letter):
    """Panel letter as a left-aligned title, so constrained layout keeps it clear of other text."""
    ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")


def map_axes(ax):
    """Image-coordinate ticks: per-run axes, labelled with the median run's range (Section 2.3)."""
    n = 224
    ax.set_xticks([0, n / 2, n - 1], ["650", "1,825", "3,000"])
    ax.set_yticks([0, n / 2, n - 1], ["68", "37", "5"])
    ax.set_xlabel("m/z")
    ax.set_ylabel("Retention time (min)")
    for s in ax.spines.values():
        s.set_visible(True)


VARIANTS = {"nested": "nested_predictions", "full512": "predictions"}


def preds(variant):
    p = pd.read_csv(INPUTS[VARIANTS[variant]])
    assert len(p) == 228 and p.groupby("seed").size().eq(76).all()
    return p


# ---------------------------------------------------------------- Figure 2
def raw_grid(exp, nb):
    """Bin every scan the way the image code (code/bin_spectra_per_sample.py) bins the filtered ones: per-run m/z and
    RT axes, mean normalized log10 intensity per bin, empty bins 0. Panels b and c show the run before the peak-count
    filter."""
    spectra = [s for s in exp if s.size() > 0]
    lim = np.array([(m.min(), m.max(), np.log10(i + 1e-6).min(), np.log10(i + 1e-6).max())
                    for m, i in (s.get_peaks() for s in spectra)])
    mz_min, mz_max, log_min, log_max = lim[:, 0].min(), lim[:, 1].max(), lim[:, 2].min(), lim[:, 3].max()
    rt = np.array([s.getRT() for s in spectra])
    yi = ((rt - rt.min()) / (rt.max() - rt.min()) * (nb - 1)).astype(int)
    total, count = np.zeros((nb, nb)), np.zeros((nb, nb))
    for s, y in zip(spectra, yi):
        m, i = s.get_peaks()
        xi = ((m - mz_min) / (mz_max - mz_min) * (nb - 1)).astype(int)
        total[y] += np.bincount(xi, weights=(np.log10(i + 1e-6) - log_min) / (log_max - log_min), minlength=nb)
        count[y] += np.bincount(xi, minlength=nb)
    grid = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return grid, (mz_min, mz_max), (rt.min() / 60, rt.max() / 60)


def boxed(ax):
    for s in ax.spines.values():
        s.set_visible(True)


def fig2_pipeline():
    """Figure 2, the processing of one run (S33A): a-c show every scan of the run, d-f the images the CNN sees,
    after the peak-count filter has removed the sparse scans (all MS2)."""
    from pyopenms import MSExperiment, MzMLFile
    exp = MSExperiment()
    MzMLFile().load(str(INPUTS["mzml_s33a"]), exp)
    rts = np.array([s.getRT() / 60 for s in exp])
    tic = np.array([s.get_peaks()[1].sum() for s in exp])
    levels = pd.Series([s.getMSLevel() for s in exp]).value_counts().sort_index()
    nb = 1024
    grid, (mz0, mz1), (rt0, rt1) = raw_grid(exp, nb)

    fig = plt.figure(figsize=(170 * MM, 185 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.75, 1.3, 1.0], hspace=0.42,
                          left=0.08, right=0.98, top=0.96, bottom=0.05)
    ax = fig.add_subplot(gs[0])
    ax.plot(rts, tic, color="tab:red", lw=0.6)
    ax.fill_between(rts, tic, color="tab:red", alpha=0.15, lw=0)
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Total Ion Current")
    ax.grid(alpha=0.3, lw=0.5)
    boxed(ax)
    ax.set_title("(a) LC-MS Data Acquisition (Patient S33A)", fontweight="bold")

    mid = gs[1].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.25)
    ax = fig.add_subplot(mid[0], projection="3d")
    X, Y = np.meshgrid(np.linspace(mz0, mz1, nb), np.linspace(rt0, rt1, nb))
    ax.plot_surface(X, Y, grid, cmap="viridis", rstride=2, cstride=2, linewidth=0, antialiased=False,
                    rasterized=True)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xlabel("m/z")
    ax.set_ylabel("RT (min)")
    ax.set_zlabel("Normalized log(intensity)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_title("(b) 3D Spectral Landscape (raw)", fontweight="bold")

    ax = fig.add_subplot(mid[1])
    ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", interpolation="nearest",
              extent=[mz0, mz1, rt0, rt1])
    ax.set_xlabel("m/z")
    ax.set_ylabel("RT (min)")
    boxed(ax)
    ax.set_title("(c) Top-Down View (raw)", fontweight="bold")

    bot = gs[2].subgridspec(1, 3, wspace=0.3)
    for i, (n, letter) in enumerate(((1024, "d"), (512, "e"), (256, "f"))):
        ax = fig.add_subplot(bot[i])
        ax.imshow(Image.open(INPUTS["img_dir"] / f"spectra_filter_{n}/S33A.png").convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("m/z →")
        if i == 0:
            ax.set_ylabel("RT →")  # the images are saved with origin="lower", so RT increases upward
        boxed(ax)
        ax.set_title(f"({letter}) {n} × {n}", fontweight="bold")
    save(fig, "fig2_pipeline")
    return {"scans": len(rts), "ms_levels": levels.to_dict(), "grid_bins": nb,
            "mz_range": [round(float(mz0), 2), round(float(mz1), 2)], "rt_range_min": [round(rt0, 2), round(rt1, 2)]}


# ---------------------------------------------------------------- Figure 3
def fig3_gradcam_means():
    """Figure 3, mean GradCAM maps of each class and their difference (jet maps on 0-1 with their own colorbars,
    RdBu difference), with the median run's m/z and RT ticks."""
    mm = np.load(INPUTS["mean_maps"])
    A, B = mm["all_A"], mm["all_B"]
    D = B - A
    fig, axs = plt.subplots(1, 3, figsize=(170 * MM, 62 * MM), gridspec_kw={"wspace": 0.75, "left": 0.07,
                            "right": 0.89, "bottom": 0.2, "top": 0.78})
    dmax = np.abs(D).max()
    for ax, m, title, kw in ((axs[0], A, "Class A (Negative)\nMean Attention", {"cmap": "jet", "vmin": 0, "vmax": 1}),
                             (axs[1], B, "Class B (Positive)\nMean Attention", {"cmap": "jet", "vmin": 0, "vmax": 1}),
                             (axs[2], D, "Difference\n(Class B − Class A)", {"cmap": "RdBu_r", "vmin": -dmax, "vmax": dmax})):
        im = ax.imshow(m, interpolation="nearest", **kw)
        ax.set_title(title)
        map_axes(ax)
        fig.colorbar(im, cax=ax.inset_axes([1.06, 0, 0.07, 1]))
    for ax in axs[1:]:
        ax.set_ylabel("")
    fig.suptitle("GradCAM Attention Comparison", fontweight="bold", fontsize=10)
    save(fig, "fig3_gradcam_means")


# ---------------------------------------------------------------- Figure 4
def occlusion_brightness():
    """Mean test-image intensity per quadrant, as printed by occlusion.py (saved in logs/occlusion.txt)."""
    lines = INPUTS["occlusion_log"].read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Mean intensity per quadrant"))
    vals = {}
    for l in lines[i + 2:i + 6]:
        q, v = l.split()
        vals[q] = float(v)
    assert set(vals) == {"TL", "TR", "BL", "BR"}, vals
    return vals


def fig4_occlusion():
    """Figure 4, occlusion control: (a) accuracy with each quadrant masked, (b) AUC-ROC with only one quadrant shown,
    bars with value labels and 95% patient-level intervals; (c) accuracy drop vs mean quadrant intensity, one point
    per quadrant (no fitted line through the four points)."""
    o = pd.read_csv(INPUTS["occlusion_stats"])
    o = o[o.model == "best checkpoint"].set_index("configuration")
    quads = ["TL", "TR", "BL", "BR"]
    bright = occlusion_brightness()
    fig, axs = plt.subplots(1, 3, figsize=(170 * MM, 72 * MM), layout="constrained")
    for ax, kind, metric, ylabel, title, other in (
            (axs[0], "mask", "accuracy", "Test accuracy", "(a) Accuracy when each\nquadrant is masked", "#90caf9"),
            (axs[1], "only", "auc", "Test AUC-ROC", "(b) AUC when only one\nquadrant is shown", "#ffb74d")):
        rows = ["occlusion_full"] + [f"occlusion_{kind}_{q}" for q in quads]
        names = ["Full"] + [f"{'Mask' if kind == 'mask' else 'Only'} {q}" for q in quads]
        v = o.loc[rows, metric].to_numpy()
        lo, hi = o.loc[rows, f"{metric}_lo"].to_numpy(), o.loc[rows, f"{metric}_hi"].to_numpy()
        x = np.arange(len(names))
        ax.bar(x, v, color=["#2e7d32"] + ["#e53935" if q == "BL" else other for q in quads], edgecolor="black",
               linewidth=0.5, width=0.8, zorder=2)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=3)
        for xi, vi, hii in zip(x, v, hi):
            ax.text(xi, hii + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0.5, ls="--", lw=0.7, color="grey", zorder=1)
        ax.set_xticks(x, names, rotation=35, ha="right", rotation_mode="anchor", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)
        boxed(ax)
    full = o.loc["occlusion_full", "accuracy"]
    ax = axs[2]
    colour = {"TL": "#ffb300", "TR": "#1e88e5", "BL": "#e53935", "BR": "#43a047"}
    nudge = {"TR": (6, -4, "left"), "BR": (-6, 5, "right")}  # TR and BR nearly coincide
    for q in quads:
        drop = full - o.loc[f"occlusion_mask_{q}", "accuracy"]
        ax.scatter(bright[q], drop, s=60, color=colour[q], edgecolor="black", linewidth=0.6, zorder=3, label=q)
        dx, dy, ha = nudge.get(q, (6, 3, "left"))
        ax.annotate(q, (bright[q], drop), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    fontweight="bold", ha=ha)
    ax.set_xlabel("Mean quadrant intensity")
    ax.set_ylabel("Accuracy drop when\nquadrant masked")
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.43, 0.485)
    ax.set_ylim(-0.03, 0.33)
    ax.set_title("(c) Discriminative impact\nvs brightness", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(alpha=0.3, lw=0.5, zorder=0)
    boxed(ax)
    fig.suptitle("GradCAM-region occlusion control (ResNet-34, full-spectrum 512 × 512)", fontweight="bold",
                 fontsize=9)
    save(fig, "fig4_occlusion")
    return {q: round(full - o.loc[f"occlusion_mask_{q}", "accuracy"], 3) for q in quads}, bright


# ---------------------------------------------------------------- Figure 5
def fig5_roc(variant):
    """Figure 5, ROC curves: faint seed curves and a bold mean curve with a shaded band. The band and legend give the
    range over the three seeds, not a SD. The title names the predictions shown (nested CV, or full spectrum
    512 x 512)."""
    p = preds(variant)
    fig, ax = plt.subplots(figsize=(85 * MM, 82 * MM), layout="constrained")
    grid = np.linspace(0, 1, 501)
    tprs, aucs = [], []
    for seed, g in p.groupby("seed"):
        fpr, tpr, _ = roc_curve(g.y_true, g.y_prob)
        auc = roc_auc_score(g.y_true, g.y_prob)
        aucs.append(auc)
        ax.plot(fpr, tpr, color={42: "tab:blue", 69: "tab:orange", 137: "tab:green"}[seed], alpha=0.3, lw=1,
                label=f"Seed {seed} (AUC = {auc:.3f})")
        t = np.interp(grid, fpr, tpr)
        t[0] = 0
        tprs.append(t)
    mean, lo, hi = np.mean(tprs, 0), np.min(tprs, 0), np.max(tprs, 0)
    mean[-1] = 1
    ax.plot(grid, mean, color="#1565c0", lw=2,
            label=f"Mean (AUC = {np.mean(aucs):.3f}, range {min(aucs):.3f}–{max(aucs):.3f})")
    ax.fill_between(grid, lo, hi, color="#1565c0", alpha=0.15, lw=0, label="Range across seeds")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, lw=0.5)
    boxed(ax)
    ax.legend(loc="lower right", fontsize=7)
    title = "Nested Cross-Validation" if variant == "nested" else "ResNet-34, Full-Spectrum 512 × 512"
    ax.set_title(f"ROC Curve: {title}", fontweight="bold")
    save(fig, f"fig5_roc_{variant}")
    return {str(k): round(float(a), 3) for k, a in zip(sorted(p.seed.unique()), aucs)}


# ---------------------------------------------------------------- Figure 6
CONFIGS = [
    ("custom_cnn_256_withS11rot", "Custom CNN"),
    ("arch_resnet18_256_withS11rot", "ResNet-18"), ("arch_resnet34_256_withS11rot", "ResNet-34"),
    ("arch_resnet50_256_withS11rot", "ResNet-50"), ("arch_resnet101_256_withS11rot", "ResNet-101"),
    ("arch_resnet152_256_withS11rot", "ResNet-152"),
    ("fullspec_resnet34_512_withS11rot", "512 × 512"), ("fullspec_resnet34_1024_withS11rot", "1024 × 1024"),
    ("subregion_resnet34_256_withS11rot", "256 × 256"), ("subregion_resnet34_512_withS11rot", "512 × 512"),
    ("subregion_resnet34_1024_withS11rot", "1024 × 1024"),
]
GROUPS = [("Baseline\n(full spectrum, 256 × 256)", 0), ("Architecture\n(full spectrum, 256 × 256)", 1),
          ("Binning\n(ResNet-34, full spectrum)", 6), ("Subregion\n(ResNet-34)", 8)]


def fig6_configurations(variant):
    """Figure 6, single-model accuracy (blue) and AUC-ROC (orange) with 95% intervals for every configuration, as a
    dot plot."""
    s = pd.concat([pd.read_csv(INPUTS["single38"]), pd.read_csv(INPUTS["single_custom"])])
    s = s[s.model == "best checkpoint"].set_index("configuration")
    n = pd.read_csv(INPUTS["nested_single"])
    n = n[n.model == "best checkpoint"].iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 108 * MM), sharey=True,
                            gridspec_kw={"wspace": 0.08, "left": 0.28, "right": 0.98, "top": 0.88, "bottom": 0.12})
    y = -np.arange(len(CONFIGS)) - np.repeat([0, 1.8, 3.6, 5.4], [1, 5, 2, 3])  # room for the two-line group headings
    for ax, metric, xlabel, colour in ((axs[0], "accuracy", "Accuracy", "#42a5f4"), (axs[1], "auc", "AUC-ROC", "#ffa726")):
        for yi, (cfg, _) in zip(y, CONFIGS):
            v, lo, hi = s.loc[cfg, [metric, f"{metric}_lo", f"{metric}_hi"]]
            ref = variant == "full512" and cfg == "fullspec_resnet34_512_withS11rot"
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="D" if ref else "o", ms=6 if ref else 5,
                        mfc=colour, mec="black", mew=0.6, ecolor="black", elinewidth=0.9, capsize=2)
        if variant == "nested":  # nested cross-validation estimate (single model) as a reference line
            ax.axvline(n[metric], ls="--", lw=0.9, color="#424242")
            ax.text(n[metric] - 0.006, y[5] - 0.9, f"Nested CV {n[metric]:.3f}", ha="right", va="center",
                    fontsize=8, color="#424242")
        ax.axvline(0.5, ls="--", lw=0.7, color=OKABE["grey"])
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.set_title(xlabel)
    axs[0].set_yticks(y, [lab for _, lab in CONFIGS])
    axs[0].set_ylim(y.min() - 0.7, y.max() + 1.7)
    axs[1].tick_params(axis="y", left=False)
    for name, a in GROUPS:  # left of the axis, on their own rows, so they never cross the plot
        axs[0].annotate(name, xy=(-0.03, y[a] + 1.05), xycoords=("axes fraction", "data"), ha="right",
                        va="center", fontsize=8, fontweight="bold", annotation_clip=False)
    fig.suptitle("Classification Performance Across Configurations (38 patients)", fontweight="bold", fontsize=10)
    save(fig, f"fig6_configurations_{variant}")
    return {m: round(float(n[m]), 3) for m in ("accuracy", "auc")} if variant == "nested" else None


# ---------------------------------------------------------------- Supplementary Figure S1
def figS1_lopo():
    """Supplementary Figure S1, leave-one-patient-out (LOPO) results over 10 seeds. A: accuracy per patient, S11 in red
    and labelled, the mean of the others dashed, y from 0.5 (chance). B: predicted probabilities of S11 and three
    example patients, with the 0.5 threshold."""
    L = pd.read_csv(INPUTS["lopo"])
    assert len(L) == 760 and L.Seed.nunique() == 10, (len(L), L.Seed.nunique())
    acc = L.groupby("Patient").Correct.mean()
    order = sorted(acc.index, key=lambda pt: (-acc[pt], pt))  # order: accuracy, then name as text
    blue, red = "#4c72b0", "#c44e52"
    fig = plt.figure(figsize=(170 * MM, 135 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.75, wspace=0.12,
                          left=0.08, right=0.98, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(order))
    ax.bar(x, acc[order], color=[red if pt == "S11" else blue for pt in order], width=0.75, zorder=2)
    others = acc.drop("S11").mean()
    ax.axhline(others, ls="--", lw=0.8, color=blue, alpha=0.7, label=f"Mean excl. S11 ({others:.1%})")
    i11 = order.index("S11")
    ax.annotate(f"S11\n({acc['S11']:.0%})", (i11, acc["S11"]), xytext=(i11 - 0.4, 0.83), ha="center", va="bottom",
                color=red, fontweight="bold", fontsize=8, arrowprops={"arrowstyle": "-", "color": red, "lw": 0.8})
    ax.set_xticks(x, order, rotation=90, fontsize=6.5)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("LOPO Classification Accuracy")
    ax.set_xlabel("Patient")
    ax.grid(alpha=0.3, lw=0.5, zorder=0)
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Leave-One-Patient-Out Accuracy by Patient (10 Seeds)")
    fig.text(0.01, 0.965, "A", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.455, "B", fontsize=10, fontweight="bold")
    fig.text(0.53, 0.455, "LOPO Prediction Probabilities: S11 vs Three Example Patients (10 seeds per patient)",
             ha="center", fontsize=8, fontweight="bold")
    rng = np.random.default_rng(0)  # horizontal jitter only, for display
    for i, pt in enumerate(["S11", "S29", "S30", "S32"]):
        ax = fig.add_subplot(gs[1, i])
        g = L[L.Patient == pt].sort_values(["TrueLabel", "Seed"])
        for lab, xpos, c, name in ((0, 0, blue, "Class A (non-cancer)"), (1, 1, red, "Class B (cancer)")):
            v = g[g.TrueLabel == lab].PredictedProb.to_numpy()
            ax.scatter(xpos + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=c, alpha=0.7, zorder=3, label=name)
        ax.axhline(0.5, ls="--", lw=0.7, color="grey", alpha=0.6)
        ax.text(1.45, 0.51, "threshold", ha="right", va="bottom", fontsize=6, color="grey")
        ax.set_xticks([0, 1], ["Class A", "Class B"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{pt}\n(LOPO acc: {acc[pt]:.0%})", color=red if pt == "S11" else blue, fontweight="bold",
                     fontsize=8)
        boxed(ax)
        if i == 2:  # legend placed where it hides no points
            ax.legend(loc="center", bbox_to_anchor=(0.5, 0.73), fontsize=6)
        if i == 0:
            ax.set_ylabel("Predicted Probability (Class B)", fontsize=8)
        else:
            ax.set_yticklabels([])
    save(fig, "figS1_lopo")
    s11b = L[(L.Patient == "S11") & (L.TrueLabel == 1)].PredictedProb
    return {"S11": round(acc["S11"], 3), "S17": round(acc["S17"], 3), "others_mean": round(others, 3),
            "S11B_prob_range": (round(s11b.min(), 2), round(s11b.max(), 2))}


# ---------------------------------------------------------------- Supplementary Figure S2
def figS2_s16_gradcam():
    """Supplementary Figure S2, GradCAM for patient S16: image, overlay and heatmap of S16A (A) and S16B (B), with one
    colorbar. Maps and probability are the mean of the three seeds' models."""
    meta = pd.read_csv(INPUTS["maps_meta"])
    full = np.load(INPUTS["maps"])["full"]
    fig, axs = plt.subplots(2, 3, figsize=(170 * MM, 125 * MM), gridspec_kw={"wspace": 0.06, "hspace": 0.32,
                            "left": 0.05, "right": 0.88, "top": 0.9, "bottom": 0.03})
    for r, (f, true, letter) in enumerate((("S16A.png", "A", "A"), ("S16B.png", "B", "B"))):
        idx = meta.index[meta.filename == f]
        assert len(idx) == 3
        m = full[idx].mean(0)
        prob = meta.loc[idx, "prob"].mean()
        pred = "B" if prob > 0.5 else "A"
        img = Image.open(INPUTS["img_dir"] / "spectra_filter_512" / f).convert("RGB")
        m512 = np.array(Image.fromarray(m.astype(np.float32), mode="F").resize(img.size, Image.BILINEAR))
        axs[r, 0].imshow(img)
        axs[r, 1].imshow(img)
        axs[r, 1].imshow(m512, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        im = axs[r, 2].imshow(m, cmap="jet", vmin=0, vmax=1, interpolation="nearest")
        for c, t in enumerate(("Original", "GradCAM Overlay", "Heatmap")):
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            axs[r, c].set_title(t, fontsize=8)
            for sp in axs[r, c].spines.values():
                sp.set_visible(False)
        y = axs[r, 1].get_position().y1 + 0.045
        fig.text(0.465, y, f"{f} | True: {true} | Pred: {pred} (p={prob:.3f}) | "
                 f"{'CORRECT' if pred == true else 'INCORRECT'}", ha="center", fontsize=8, fontweight="bold")
        fig.text(0.01, y, letter, fontsize=10, fontweight="bold")
    cax = fig.add_axes([0.9, 0.25, 0.018, 0.5])
    fig.colorbar(im, cax=cax, label="GradCAM attention")
    save(fig, "figS2_s16_gradcam")
    return meta[meta.filename.isin(["S16A.png", "S16B.png"])][["filename", "seed", "prob"]].round(3).to_dict("records")


# ---------------------------------------------------------------- Supplementary Figure S3
def chosen_labels():
    """Configuration chosen by the inner cross-validation in each outer fold, labelled as in Table 5."""
    k = pd.read_csv(INPUTS["nested_picks"])
    lab = {}
    for fold, c in zip(k.outer, k.chosen):
        kind, net, res = c.replace("_withS11rot", "").split("_")[:3]
        full = kind == "fullspec" or (kind == "arch" and net == "resnet34")  # arch_resnet34_256 is full spectrum 256
        lab[fold] = f"Subregion {res}" if kind == "subregion" else f"Full {res}" if full else f"ResNet-{net[6:]}, {res}"
    return lab


def figS3_per_fold(variant):
    """Supplementary Figure S3, per-fold accuracy, AUC-ROC and F1 of the three seeds, shown as points (three values per
    fold are too few for a box plot) with the mean of the seeds. Seed colours as in Figure 5; the configuration chosen
    in each outer fold is given in the fold label (nested variant)."""
    p = preds(variant)
    rows = []
    for (seed, fold), g in p.groupby(["seed", "fold"]):
        pred = (g.y_prob > 0.5).astype(int)
        tp = ((pred == 1) & (g.y_true == 1)).sum()
        f1 = 2 * tp / (2 * tp + ((pred == 1) & (g.y_true == 0)).sum() + ((pred == 0) & (g.y_true == 1)).sum())
        rows.append({"seed": seed, "fold": fold, "accuracy": (pred == g.y_true).mean(),
                     "auc": roc_auc_score(g.y_true, g.y_prob), "f1": f1, "n": len(g)})
    pf = pd.DataFrame(rows)
    chosen = chosen_labels() if variant == "nested" else {}
    ticks = [f"Fold {k}" + (f" ({chosen[k].replace('Subregion', 'Sub')})" if k in chosen else "") for k in range(6)]
    style = {42: ("tab:blue", "o"), 69: ("tab:orange", "s"), 137: ("tab:green", "^")}
    with plt.rc_context({"font.family": "DejaVu Serif"}):
        fig, axs = plt.subplots(1, 3, figsize=(170 * MM, 72 * MM), sharey=True, layout="constrained")
        for ax, metric, title in ((axs[0], "accuracy", "Accuracy"), (axs[1], "auc", "AUC-ROC"), (axs[2], "f1", "F1 Score")):
            for k, (seed, (c, mk)) in enumerate(style.items()):
                g = pf[pf.seed == seed]
                ax.scatter(g.fold + (k - 1) * 0.17, g[metric], color=c, marker=mk, s=22, edgecolor="black",
                           linewidth=0.4, zorder=3, label=f"Seed {seed}")
            means = pf.groupby("fold")[metric].mean()
            ax.hlines(means.to_numpy(), means.index - 0.32, means.index + 0.32, color="black", lw=1.2, zorder=2,
                      label="Mean of seeds")
            ax.set_xticks(range(6), ticks, fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_ylim(0.5, 1.05)  # every value is above 0.7; 0.5 is chance
            ax.set_title(title)
            ax.grid(alpha=0.3, lw=0.5)
        axs[0].set_ylabel("Score")
        axs[2].legend(loc="lower left", fontsize=6.5, ncol=2)
        title = "nested cross-validation" if variant == "nested" else "fullspec_resnet34_512"
        fig.suptitle(f"Per-Fold Performance ({title})")
        save(fig, f"figS3_per_fold_{variant}")
    return pf.round(3).to_dict("records")


# ---------------------------------------------------------------- Supplementary Figure S4
def figS4_confusion(variant):
    """Supplementary Figure S4, confusion matrices of the three seeds (raw counts over the 6 folds). The title names the
    predictions shown."""
    p = preds(variant)
    fig, axs = plt.subplots(1, 3, figsize=(170 * MM, 72 * MM), layout="constrained")
    counts = {}
    for ax, (seed, g) in zip(axs, p.groupby("seed")):
        pred = (g.y_prob > 0.5).astype(int)
        cm = np.array([[((g.y_true == t) & (pred == q)).sum() for q in (0, 1)] for t in (0, 1)])
        counts[str(seed)] = cm.tolist()
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=38)
        for t in (0, 1):
            for q in (0, 1):
                ax.text(q, t, cm[t, q], ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if cm[t, q] > 19 else "black")
        ax.set_xticks([0, 1], ["Class A\n(Pred)", "Class B\n(Pred)"], fontsize=7)
        ax.set_yticks([0, 1], ["Class A\n(Actual)", "Class B\n(Actual)"], fontsize=7)
        ax.set_title(f"Seed {seed}\n(Acc = {np.trace(cm) / cm.sum():.1%})", fontweight="bold", fontsize=8)
        boxed(ax)
    title = "Nested Cross-Validation" if variant == "nested" else "ResNet-34, Full-Spectrum 512 × 512"
    fig.suptitle(f"Confusion Matrices: {title}\n(Raw counts across 6 CV folds, {len(p) // 3} samples per seed)",
                 fontweight="bold", fontsize=9)
    save(fig, f"figS4_confusion_{variant}")
    return counts


FIGS = {}
for f in (fig2_pipeline, fig3_gradcam_means, fig4_occlusion, fig5_roc, fig6_configurations,
          figS1_lopo, figS2_s16_gradcam, figS3_per_fold, figS4_confusion):
    if f in (fig5_roc, fig6_configurations, figS3_per_fold, figS4_confusion):
        FIGS.update({f"{f.__name__}_{v}": (lambda f=f, v=v: f(v)) for v in VARIANTS})
    else:
        FIGS[f.__name__] = f

if __name__ == "__main__":
    wanted = sys.argv[2:] or list(FIGS)
    report = {"matplotlib": mpl.__version__, "inputs": {}}
    for k, pth in INPUTS.items():
        if pth.is_file():
            report["inputs"][k] = {"path": str(pth), "sha256": hashlib.sha256(pth.read_bytes()).hexdigest()[:16]}
    for name in wanted:
        report[name] = FIGS[name]()
    (OUT / "manifest.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs",)}, indent=1, default=str)[:4000])
